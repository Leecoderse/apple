import numpy as np
import pandas as pd

import json
import os

train_df = pd.DataFrame()
val_df = pd.DataFrame()



############################################################
# json 파일 모두 합치기
############################################################

# json파일 데이터프레임으로 통합(훈련용)
path1 = '경로 추가하기'

file_list1 = os.listdir(path1)
file_list_py1 = [file for file in file_list1 if file.endswith('.json')]
file_list_len1 = len(file_list_py1)
dict_list1 = []

for i in file_list_py1:
    try:
        for line in open((path1 + '/' + i), "r", encoding='utf8'):
            dict_list1.append(json.loads(line))
    except:
        pass
        
    df_train = pd.DataFrame(dict_list1)
    df_train = pd.concat([train_df, df_train], ignore_index = True)


# json파일 데이터프레임으로 통합(검증용)
path2 = '경로 추가하기'

file_list2 = os.listdir(path2)
file_list_py2 = [file for file in file_list2 if file.endswith('.json')]
file_list_len2 = len(file_list_py2)
dict_list2 = []

for i in file_list_py2:
    try:
        for line in open((path2 + '/' + i), "r", encoding='utf8'):
            dict_list2.append(json.loads(line))
    except:
        pass
        
    df_valid = pd.DataFrame(dict_list2)
    df_valid = pd.concat([val_df, df_valid], ignore_index = True)



############################################################
# 딕셔너리 꼴의 데이터를 이용하기 편하게 바꾸기
############################################################

# 필요 부분만을 활용한 데이터프레임 생성(딕셔너리 꼴의 데이터에서 필요한 key값을 열로 지정 후 짝인 value값 넣기)
for col in df_train.columns:
    val_df_train = pd.DataFrame()
    
    for i in range(len(df_train.index)):
        col_dict1 = df_train[col][i]
        val_dict1 = {}

        for keys in col_dict1.keys():
            if type(col_dict1.get(keys)) is list:
                val_dict1[keys] = [col_dict1.get(keys)]
            else:
                df_train.loc[i, keys] = col_dict1.get(keys)
        
        # list가 value값인 열이 맨 마지막에 있었기 때문에 가능 (val_df는 새로운 열에서 초기화 됨)
        val_dict1 = pd.DataFrame(val_dict1, index = [i])
        val_df_train = pd.concat([val_df_train, val_dict1])

    df_train = df_train.drop(col, axis = 1)

# 훈련용 데이터프레임 확인
df_train = pd.concat([df_train, val_df_train], axis = 1)

# 딕셔너리 꼴의 데이터에서 필요한 key값을 열로 지정 후 짝인 value값 넣기
for col in df_valid.columns:

    val_df_valid = pd.DataFrame()

    for i in range(len(df_valid.index)):
        col_dict2 = df_valid[col][i]
        val_dict2 = {}

        for keys in col_dict2.keys():
            if type(col_dict2.get(keys)) is list:
                val_dict2[keys] = [col_dict2.get(keys)]
            else:
                df_valid.loc[i, keys] = col_dict2.get(keys)
        
        # list가 value값인 열이 맨 마지막에 있었기 때문에 가능한 것 (val_df는 새로운 열에서 초기화 됨)
        val_dict2 = pd.DataFrame(val_dict2, index = [i])
        val_df_valid = pd.concat([val_df_valid, val_dict2])

    df_valid = df_valid.drop(col, axis = 1)

# 검증용 데이터프레임 확인
df_valid = pd.concat([df_valid, val_df_valid], axis = 1)



############################################################
# 데이터프레임 정리 밎 저장
############################################################

# 데이터프레임 열 정리
df_train = df_train.drop(['description','url','version','year','img_path','label_path'], axis = 1)
df_valid = df_valid.drop(['description','url','version','year','img_path','label_path'], axis = 1)

# csv로 변환하여 저장

df_train.to_csv('train.csv')
df_valid.to_csv('val.csv')


