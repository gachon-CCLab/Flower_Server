# https://github.com/adap/flower/tree/main/examples/advanced_tensorflow 참조

import logging
from typing import Dict,Optional, Tuple

import flwr as fl
import tensorflow as tf

from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

# import wandb
import datetime
import os
import boto3
import requests, json
import time

# TF warning log 필터링
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# FL 하이퍼파라미터 설정
class FL_server_parameter:
    num_rounds = 20
    local_epochs = 10
    batch_size = 32
    val_steps = 5
    latest_gl_model_v = 0 # 이전 글로벌 모델 버전
    next_gl_model_v = 0 # 생성될 글로벌 모델 버전
    start_by_round = 0 # fit aggregation start
    end_by_round = 0 # fit aggregation end
    round = 0 # round 수


server = FL_server_parameter()


# 참고: https://loosie.tistory.com/210, https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
# aws session 연결
def aws_session(region_name='ap-northeast-2'):
    return boto3.session.Session(aws_access_key_id=os.environ.get('ACCESS_KEY_ID'),
                                aws_secret_access_key=os.environ.get('ACCESS_SECRET_KEY'),
                                region_name=region_name)

# s3에 global model upload
def upload_model_to_bucket(global_model):
    bucket_name = os.environ.get('BUCKET_NAME')
    global server
    
    logging.info(f'gl_model_{server.next_gl_model_v}_V.h5 모델 업로드 시작')

    session = aws_session()
    s3_resource = session.resource('s3')
    bucket = s3_resource.Bucket(bucket_name)
    bucket.upload_file(
        Filename=f'/app/gl_model_{server.next_gl_model_v}_V.h5',
        Key=global_model,
    )
    
    s3_url = f"https://{bucket_name}.s3.amazonaws.com/{global_model}"
    logging.info(f'gl_model_{server.next_gl_model_v}_V.h5 모델 업로드 완료')
    return s3_url

# s3에 저장되어 있는 latest global model download
def model_download():

    bucket_name = os.environ.get('BUCKET_NAME')
    # print('bucket_name: ', bucket_name)
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

        logging.info('model 있음')
        logging.info('model_file_list: ', file_list)
        gl_model = file_list[len(file_list)-1]
        # print('gl_model: ', gl_model)
        gl_model_v = int(file_list[len(file_list)-1].split('_')[2])
        logging.info(f'gl_model: {gl_model}, gl_model_v: {gl_model_v}')

        s3_resource.download_file(bucket_name, f'gl_model_{gl_model_v}_V.h5', f'/app/gl_model_{gl_model_v}_V.h5')

        return gl_model, gl_model_v

    # s3에 global model 없을 경우
    except Exception as e:
        logging.error('global model read error: ', e)

        model_X = None
        gl_model_v = 0
        logging.info(f'gl_model: {model_X}, gl_model_v: {gl_model_v}')
        return model_X, gl_model_v


def init_gl_model():
    # model 생성
    init_model = Sequential()

    # Convolutional Block (Conv-Conv-Pool-Dropout)
    init_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    init_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    init_model.add(MaxPool2D(pool_size=(2, 2)))
    init_model.add(Dropout(0.25))

    # Classifying
    init_model.add(Flatten())
    init_model.add(Dense(512, activation='relu'))
    init_model.add(Dropout(0.5))
    init_model.add(Dense(10, activation='softmax'))

    METRICS = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    ]

    init_model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=METRICS)

    return init_model


def main(model) -> None:

    global server

    logging.info(f'latest_gl_model_v: {server.latest_gl_model_v}')

    if not model:

        logging.info('init global model making')
        init_model = init_gl_model()

        fl_server_start(init_model)

    else:
        logging.info('load latest global model')

        fl_server_start(model)


def fl_server_start(model):
    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit > fraction_eval이여야 함
        # min_available_clients의 수를 실제 연결 client 수 보다 작게 하는게 안정적임
        # => client가 학습 중에 멈추는 현상이 가끔 발생
        fraction_fit=1.0,  # 클라이언트 학습 참여 비율
        fraction_evaluate=1.0,  # 클라이언트 평가 참여 비율
        min_fit_clients=5,  # 최소 학습 참여 수
        min_evaluate_clients=5,  # 최소 평가 참여 수
        min_available_clients=5,  # 최소 클라이언트 연결 필요 수
        evaluate_fn=get_eval_fn(model),  # 모델 평가 결과
        on_fit_config_fn=fit_config,  # batchsize, epoch 수
        on_evaluate_config_fn=evaluate_config,  # val_step
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=server.num_rounds),
        strategy=strategy,
    )
        

def get_eval_fn(model):
    """Return an evaluation function for server-side evaluation."""
    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    global server

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        # loss, accuracy, precision, recall, auc, auprc = model.evaluate(x_val, y_val)
        loss, accuracy = model.evaluate(x_val, y_val)

        model.set_weights(parameters)  # Update model with the latest parameters

        if server.round >= 1:
            # fit aggregation end time
            server.end_by_round = time.time() - server.start_by_round
            # round_server_operation_time = str(datetime.timedelta(seconds=server.end_by_round))
            server_time_result = {"round": server.round, "operation_time_by_round": server.end_by_round}
            json_time_result = json.dumps(server_time_result)
            logging.info(f'server_time - {json_time_result}')

            # round 별 glmodel 성능
            server_eval_result = {"round": server.round, "gl_loss": loss, "gl_accuracy": accuracy}
            json_eval_result = json.dumps(server_eval_result)
            logging.info(f'server_performance - {json_eval_result}')
            
        # model save
        model.save('/app/gl_model_%s_V.h5' % server.next_gl_model_v)

        return loss, {"accuracy": accuracy}

    return evaluate


def fit_config(rnd: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    global server

    config = {
        "batch_size": server.batch_size,
        "local_epochs": server.local_epochs,
        "num_rounds": server.num_rounds,
    }

    # increase round
    server.round += 1

    # if server.round > 2:
    # fit aggregation start time
    server.start_by_round = time.time()
    logging.info('server start by round')

    return config


def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    global server

    return {"val_steps": server.val_steps}


if __name__ == "__main__":

    # Cifar 10 데이터셋 불러오기
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
        
    num_classes = 10	

    # global model 평가를 위한 데이터셋
    x_val, y_val = X_test[1000:9000], y_test[1000:9000]

    # y(label) one-hot encoding
    y_val = to_categorical(y_val, num_classes)
    
    today= datetime.datetime.today()
    today_time = today.strftime('%Y-%m-%d %H-%M-%S')

    # global model download
    model, server.latest_gl_model_v = model_download()

    # 새로 생성되는 글로벌 모델 버전
    server.next_gl_model_v = server.latest_gl_model_v + 1


    # server_status 주소
    inform_SE: str = 'http://10.152.183.114:8000/FLSe/'

    inform_Payload = {
            # 형식
            'S3_bucket': 'fl-gl-model', # 버킷명
            'Latest_GL_Model': 'gl_model_%s_V.h5'%server.latest_gl_model_v,  # 모델 가중치 파일 이름
            'Play_datetime': today_time, # server 수행 시간
            'FLSeReady': True, # server 준비 상태 on
            'GL_Model_V' : server.latest_gl_model_v # GL 모델 버전
        }

    while True:
        try:
            # server_status => FL server ready
            r = requests.put(inform_SE+'FLSeUpdate', verify=False, data=json.dumps(inform_Payload))
            if r.status_code == 200:
                break
            else:
                logging.error(r.content)
        except:
            logging.error("Connection refused by the server..")
            time.sleep(5)
            continue
    
    # wandb login and init
    # wandb.login(key='6266dbc809b57000d78fb8b163179a0a3d6eeb37')
    # wandb.init(entity='ccl-fl', project='fl-server-news', name= 'server_V%s'%next_gl_model, dir='/',  \
    #     config={"num_rounds": num_rounds,"local_epochs": local_epochs, "batch_size": batch_size,"val_steps": val_steps, "today_datetime": today_time,
    #     "Model_V": next_gl_model})
    

    try:
        fl_start_time = time.time()

        # Flower Server 실행
        main(model)

        fl_end_time = time.time() - fl_start_time  # 연합학습 종료 시간
        # fl_server_operation_time = str(datetime.timedelta(seconds=fl_end_time)) # 연합학습 종료 시간

        server_all_time_result = {"operation_time": fl_end_time}
        json_all_time_result = json.dumps(server_all_time_result)
        logging.info(f'server_operation_time - {json_all_time_result}')

        logging.info('upload model in s3')
        upload_model_to_bucket("gl_model_%s_V.h5" %server.next_gl_model_v)

        
        # server_status error
    except Exception as e:
        logging.error('error: ', e)
        data_inform = {'FLSeReady': False}
        requests.put(inform_SE+'FLSeUpdate', data=json.dumps(data_inform))
        
    finally:
        logging.info('server close')
      
        # server_status에 model 버전 수정 update request
        res = requests.put(inform_SE + 'FLRoundFin', params={'FLSeReady': 'false'})
        if res.status_code == 200:
            logging.info('global model version upgrade')
            logging.info('global model version: ', res.json()['Server_Status']['GL_Model_V'])

        # wandb 종료
        # wandb.finish()