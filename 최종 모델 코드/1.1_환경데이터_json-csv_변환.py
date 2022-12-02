import numpy as np
import pandas as pd
import json
import os

path_basic = 'json 데이터 들어있는 폴더 경로'
path_list = ['세부 폴더 리스트']
apple_1 = pd.DataFrame()


# json 파일 모두 합치기
for i in range(3,6):
    path = path_basic + path_list[i]

    file_list = os.listdir(path)
    file_list_py = [file for file in file_list if file.endswith('.json')]
    file_list_len = len(file_list_py)

    dict_list = []
    for i in file_list_py:
        try:
            for line in open((path + '/' + i), "r", encoding = 'utf8'):
                dict_list.append(json.loads(line))
        except:
            pass
    
    df = pd.DataFrame(dict_list)
    apple_1 = pd.concat([apple_1, df], ignore_index = True)


# 딕셔너리 꼴의 데이터에서 필요한 key값을 열로 지정 후 짝인 value값 넣기
for col in apple_1.columns:
    val_df = pd.DataFrame()

    for i in range(len(apple_1.index)):
        col_dict = apple_1[col][i]
        val_dict = {}
        for keys in col_dict.keys():
            if type(col_dict.get(keys)) is list:
                val_dict[keys] = [col_dict.get(keys)]
            else:
                apple_1.loc[i, keys] = col_dict.get(keys)
        # list가 value값인 열이 맨 마지막에 있었기 때문에 가능한 것 (val_df는 새로운 열에서 초기화 됨)
        val_dict = pd.DataFrame(val_dict, index = [i])
        val_df = pd.concat([val_df, val_dict])

    apple_1 = apple_1.drop(col, axis = 1)


# 필요없는 열 삭제
apple_1 = apple_1.drop(['description','url','version','year','img_path','label_path'], axis = 1)


# csv로 저장
apple_1.to_csv('apple_valid_arisu.csv')