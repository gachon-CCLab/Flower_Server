# client dataset define
from pyarrow import csv

import numpy as np
import pandas as pd

# 중복 개수 확인 => 환자 정보 개수 확인
from collections import Counter
import operator


def data_load():
    # 파일 경로
    file_path='dataset/NEWS_dataset.csv'
    
    df = csv.read_csv(file_path).to_pandas()

    # 환자 ID 추출
    patient_id = df.iloc[:,0]
    p_counter = Counter(patient_id)

    # value값으로 정렬 => 같은 환자id 개수 확인(내림차순)
    p_counter2 = sorted(p_counter.items(), key=operator.itemgetter(1), reverse=True)

    # patient id list화
    p_list = []
    for i in p_counter2:
        p_list.append(i[0])

    return df, p_list
