# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조

from typing import Dict,Optional, Tuple

import flwr as fl
import tensorflow as tf
import tensorflow_addons as tfa

from keras.utils.np_utils import to_categorical

import numpy as np

import health_dataset as dataset

import wandb
from datetime import datetime
import os
import boto3

import requests, json
import time

# FL 하이퍼파라미터 설정
num_rounds = 1
local_epochs = 1
batch_size = 32
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
    
    session = aws_session()
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(
        Filename=f'/app/gl_model_{next_gl_model}_V.h5',
        Key=global_model,
    )
    
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{global_model}"

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
        gl_model = file_list[len(file_list)-1]
        gl_model_v = int(file_list[len(file_list)-1].split('_')[2])
        print(f'gl_model: {gl_model}, gl_model_v: {gl_model_v}')

        s3_resource.download_file(bucket_name, f'gl_model_{gl_model_v}_V.h5', f'/app/gl_model_{gl_model_v}_V.h5')

        return gl_model, gl_model_v
        
        # # gl_model이 없으면
        # if file_list == 0:
        #     print('model 없음')
        #     model_X = 'null'
        #     gl_model_v = 0
        #     print(f'gl_model: {gl_model}, gl_model_v: {gl_model_v}')
        #     return model_X, gl_model_v
        # else:
        #     print('model 있음')
        #     gl_model = file_list[len(file_list)-1]
        #     gl_model_v = int(file_list[len(file_list)-1].split('_')[2])
        #     print(f'gl_model: {gl_model}, gl_model_v: {gl_model_v}')

        #     s3_resource.download_file(bucket_name, f'gl_model_{gl_model_v}_V.h5', f'/app/gl_model_{gl_model_v}_V.h5')

        #     return gl_model, gl_model_v
    
    except Exception as e:
        print('error: ', e)

        model_X = 'null'
        gl_model_v = 0
        print(f'gl_model: {model_X}, gl_model_v: {gl_model_v}')
        return model_X, gl_model_v

    
def model_build(model):
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tfa.metrics.F1Score(name='f1_score', num_classes=5, average='micro'),
        tf.keras.metrics.AUC(name='auprc', curve='PR'), # precision-recall curve
        ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    return model

def fl_server_start(model):

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation    
    
    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tfa.metrics.F1Score(name='f1_score', num_classes=5, average='micro'),
        tf.keras.metrics.AUC(name='auprc', curve='PR'), # precision-recall curve
        ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=METRICS)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit > fraction_eval이여야 함
        fraction_fit=0.5, # 클라이언트 학습 참여 비율
        fraction_eval=0.4, # 클라이언트 평가 참여 비율
        min_fit_clients=4, # 최소 학습 참여 수
        min_eval_clients=4, # 최소 평가 참여 수
        min_available_clients=4, # 클라이언트 연결 필요 수
        eval_fn=get_eval_fn(model), # 모델 평가 결과
        on_fit_config_fn=fit_config, # batchsize, epoch 수
        on_evaluate_config_fn=evaluate_config, # val_step
        initial_parameters=fl.common.weights_to_parameters(model.get_weights()),
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server("[::]:8080", config={"num_rounds": num_rounds}, strategy=strategy)

def main() -> None:

    global num_rounds, latest_gl_model_v

    print('')
    print('latest_gl_model_v', latest_gl_model_v)
    print('')

    global x_val, y_val # f1_score 계산을 위해 label 개수 확인
    
    if os.path.isfile('/app/gl_model_%s_V.h5'%latest_gl_model_v):
        print('load model')
        model = tf.keras.models.load_model('/app/gl_model_%s_V.h5'%latest_gl_model_v)
        fl_server_start(model)

    else:
        # global model 없을 시 초기 글로벌 모델 생성
        print('basic model making')

        print('initial_model x_val length: ', x_val.shape[-1])
        print()

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(
                16, activation='relu',
                input_shape=(x_val.shape[-1],)), # input_shape에 x_val.shape[-1] 값을 넣으면 오류남 input_shape을 6으로 인식
                # input_shape=(5,)),
            tf.keras.layers.Dense(len(y_val[0]), activation='sigmoid'),
        ])

        fl_server_start(model)
        

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
        model.save("/app/gl_model_%s_V.h5'%latest_gl_model_v")

        # wandb에 log upload
        # wandb.log({'loss':loss,"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc})
        
        wandb.log({'loss':loss,"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, "auprc": auprc, "f1_score": f1_score})

        
        return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, "auprc": auprc, "f1_score":f1_score}

        # loss, accuracy, precision, recall, auc, f1_score, auprc = model.evaluate(x_val, y_val)
        # return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc, "f1_score": f1_score, "auprc": auprc}

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
    # wandb.config.update({"local_epochs": local_epochs, "batch_size": batch_size},allow_val_change=True)

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

    # server_status 주소
    inform_SE: str = 'http://0.0.0.0:8000/FLSe/'

    # server_status 확인 => 전 global model 버전
    # server_res = requests.get(inform_SE + 'info')
    # latest_gl_model_v = int(server_res.json()['Server_Status']['GL_Model_V'])
    
    next_gl_model = latest_gl_model_v + 1

    inform_Payload = {
            # 형식
            'S3_bucket': 'fl-flower-model', # 버킷명
            'S3_key': 'model_V%s.h5'%latest_gl_model_v,  # 모델 가중치 파일 이름
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
            print("Let me sleep for 5 seconds")
            print("ZZzzzz...")
            time.sleep(5)
            print("Was a nice sleep, now let me continue...")
            continue
    
    # wandb login and init
    wandb.login(key=os.environ.get('WB_KEY'))
    wandb.init(entity='ccl-fl', project='NEWS_Server_high_parameter', name= 'server_V%s'%next_gl_model, dir='/Users/yangsemo/VScode/Flower_Health_Local_NEWS/wandb_server',  \
        config={"num_rounds": num_rounds,"local_epochs": local_epochs, "batch_size": batch_size,"val_steps": val_steps, "today_datetime": today_time,
        "Model_V": next_gl_model})
    
    
    # global model 평가를 위한 dataset 
    df, p_list = dataset.data_load()

    # Use the last 5k training examples as a validation set
    x_val, y_val = df.iloc[:10000,1:6], df.loc[:9999,'label']

    # y(label) one-hot encoding
    y_val = to_categorical(np.array(y_val))
    
    # s3에서 latest global model 가져오기
    # if latest_gl_model_v > 0:
    #     print('model downloading')
    #     model_download()
    #     print('model downloaded')

    try:
        # Flower Server 실행
        main()

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