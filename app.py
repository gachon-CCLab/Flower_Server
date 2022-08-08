# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조

from typing import Dict,Optional, Tuple

import flwr as fl
import tensorflow as tf
import tensorflow_addons as tfa

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical

import wandb
from datetime import datetime
import os
import boto3

import requests, json
import time

# FL 하이퍼파라미터 설정
num_rounds = 5
local_epochs = 10
batch_size = 2048
val_steps = 5

# 참고: https://loosie.tistory.com/210, https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
# aws session 연결
def aws_session(region_name='ap-northeast-2'):
    return boto3.session.Session(aws_access_key_id=os.environ.get('ACCESS_KEY_ID'),
                                aws_secret_access_key=os.environ.get('ACCESS_SECRET_KEY'),
                                region_name=region_name)

# s3에 global model upload
def upload_model_to_bucket(global_model):
    bucket_name = os.environ.get('BUCKET_NAME')
    global latest_gl_model_v, next_gl_model
    
    print(f'gl_model_{next_gl_model}_V.h5 모델 업로드 시작')

    session = aws_session()
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(
        Filename=f'/app/gl_model_{next_gl_model}_V.h5',
        Key=global_model,
    )
    
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{global_model}"
    print(f'gl_model_{next_gl_model}_V.h5 모델 업로드 완료')
    return s3_url

# s3에 저장되어 있는 latest global model download
def model_download():
    bucket_name = os.environ.get('BUCKET_NAME')
    print('bucket_name: ', bucket_name)
    global latest_gl_model_v, next_gl_model
    
    try:
        session = aws_session()
        s3_resource = session.client('s3')
        bucket_list = s3_resource.list_objects(Bucket=bucket_name)
        content_list = bucket_list['Contents']

        # s3 bucket 내 global model 파일 조회
        file_list=[]

        for content in content_list:
            key = content['Key']
            file_list.append(key)

        print('model 있음')
        print('model_file_list: ', file_list)
        gl_model = file_list[len(file_list)-1]
        print('gl_model: ', gl_model)
        gl_model_v = int(file_list[len(file_list)-1].split('_')[2])
        print(f'gl_model: {gl_model}, gl_model_v: {gl_model_v}')

        s3_resource.download_file(bucket_name, f'gl_model_{gl_model_v}_V.h5', f'/app/gl_model_{gl_model_v}_V.h5')

        return gl_model, gl_model_v

    # s3에 global model 없을 경우
    except Exception as e:
        print('error: ', e)

        model_X = 'null'
        gl_model_v = 0
        print(f'gl_model: {model_X}, gl_model_v: {gl_model_v}')
        return model_X, gl_model_v


def fl_server_start(model, y_val):

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation    
    
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tfa.metrics.F1Score(name='f1_score', num_classes=len(y_val[0]), average='micro'),
        tf.keras.metrics.AUC(name='auprc', curve='PR'), # precision-recall curve
        ]

    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=METRICS)


    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit > fraction_eval이여야 함
        # min_available_clients의 수를 실제 연결 client 수 보다 작게 하는게 안정적임
        # => client가 학습 중에 멈추는 현상이 가끔 발생
        fraction_fit=0.6, # 클라이언트 학습 참여 비율
        fraction_eval=0.5, # 클라이언트 평가 참여 비율
        min_fit_clients=4, # 최소 학습 참여 수
        min_eval_clients=4, # 최소 평가 참여 수
        min_available_clients=3, # 최소 클라이언트 연결 필요 수
        eval_fn=get_eval_fn(model), # 모델 평가 결과
        on_fit_config_fn=fit_config, # batchsize, epoch 수
        on_evaluate_config_fn=evaluate_config, # val_step
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("0.0.0.0:8080", config={"num_rounds": num_rounds}, strategy=strategy)

def gl_model_load():

    global num_rounds, latest_gl_model_v
    global x_val, y_val # f1_score 계산을 위해 label 개수 확인

    print('')
    print('latest_gl_model_v', latest_gl_model_v)
    print('')
    
    if os.path.isfile('/app/gl_model_%s_V.h5'%latest_gl_model_v):
        print('load model')
        model = tf.keras.models.load_model('/app/gl_model_%s_V.h5'%latest_gl_model_v)
        fl_server_start(model, y_val)

    else:
        # global model 없을 시 초기 글로벌 모델 생성
        print('basic model making')

        # model 생성
        model = Sequential()

        # Convolutional Block (Conv-Conv-Pool-Dropout)
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Classifying
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        fl_server_start(model, y_val)
        

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    global x_val, y_val

    # print('get_eval_fn x_val length: ', x_val.shape[-1])

    # The `evaluate` function will be called after every round
    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(weights)  # Update model with the latest parameters
        
        # loss, accuracy, precision, recall, auc, auprc = model.evaluate(x_val, y_val)
        loss, accuracy, precision, recall, f1_score, auc, auprc = model.evaluate(x_val, y_val)

        global next_gl_model

        # model save
        model.save("/app/gl_model_%s_V.h5"%next_gl_model)

        # wandb에 log upload        
        wandb.log({'loss':loss,"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, "auprc": auprc, "f1_score": f1_score})

        
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, "auprc": auprc, "f1_score":f1_score}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """

    global batch_size, local_epochs

    config = {
        "batch_size": batch_size,
        # "local_epochs": 1 if rnd < 2 else 2,
        "local_epochs": local_epochs,
        "num_rounds": num_rounds,
    }

    # wandb log upload
    wandb.config.update({"local_epochs": local_epochs, "batch_size": batch_size},allow_val_change=True)

    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    # val_steps = 5 if rnd < 4 else 10
    global val_steps

    # wandb log upload
    wandb.config.update({"val_steps": val_steps},allow_val_change=True)
    
    return {"val_steps": val_steps}


if __name__ == "__main__":
    
    today= datetime.today()
    today_time = today.strftime('%Y-%m-%d %H-%M-%S')

    # global model download
    model, latest_gl_model_v = model_download()

    # 새로 생성되는 글로벌 모델 버전
    next_gl_model = latest_gl_model_v + 1


    # server_status 주소
    inform_SE: str = 'http://10.152.183.18:8000/FLSe/'

    inform_Payload = {
            # 형식
            'S3_bucket': 'fl-flower-model', # 버킷명
            'S3_key': 'gl_model_%s_V.h5'%latest_gl_model_v,  # 모델 가중치 파일 이름
            'play_datetime': today_time, # server 수행 시간
            'FLSeReady': True, # server 준비 상태 on
            'GL_Model_V' : latest_gl_model_v # GL 모델 버전
        }

    while True:
        try:
            # server_status => FL server ready
            r = requests.put(inform_SE+'FLSeUpdate', verify=False, data=json.dumps(inform_Payload))
            if r.status_code == 200:
                break
            else:
                print(r.content)
        except:
            print("Connection refused by the server..")
            time.sleep(5)
            continue
    
    # wandb login and init
    wandb.login(key='6266dbc809b57000d78fb8b163179a0a3d6eeb37')
    wandb.init(entity='ccl-fl', project='fl-server', name= 'server_V%s'%next_gl_model, dir='/',  \
        config={"num_rounds": num_rounds,"local_epochs": local_epochs, "batch_size": batch_size,"val_steps": val_steps, "today_datetime": today_time,
        "Model_V": next_gl_model})
    
    
    # Cifar 10 데이터셋 불러오기
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
    num_classes = 10	

    # global model 평가를 위한 데이터셋
    x_val, y_val = X_test[1000:9000], y_test[1000:9000]

    # y(label) one-hot encoding
    y_val = to_categorical(y_val, num_classes)

    try:
        start_time = time.time()
        # Flower Server 실행
        gl_model_load()
        end_time = time.time()
        excution_time = end_time - start_time
        print('excution_time: ', excution_time)
        # s3 버킷에 global model upload
        upload_model_to_bucket("gl_model_%s_V.h5" %next_gl_model)

        
        # server_status error
    except Exception as e:
        print('error: ', e)
        data_inform = {'FLSeReady': False}
        requests.put(inform_SE+'FLSeUpdate', data=json.dumps(data_inform))
        
    finally:
        print('server close')
      
        # server_status에 model 버전 수정 update request
        res = requests.put(inform_SE + 'FLRoundFin', params={'FLSeReady': 'false'})
        if res.status_code == 200:
            print('global model version upgrade')
            print('global model version: ', res.json()['Server_Status']['GL_Model_V'])

        # wandb 종료
        wandb.finish()