import pandas as pd

# 각각 csv형태로 저장한 데이터 가져오기(train 8개 파일, val 6개 파일)
df0 = pd.read_csv("train0.csv")
df1 = pd.read_csv("train1.csv")
df2 = pd.read_csv("train2.csv")
df3 = pd.read_csv("train3.csv")
df4 = pd.read_csv("train4.csv")
df5 = pd.read_csv("train5.csv")
df6 = pd.read_csv("train6.csv")
df7 = pd.read_csv("train7.csv")

df8 = pd.read_csv("val0.csv")
df9 = pd.read_csv("val1.csv")
df10 = pd.read_csv("val2.csv")
df11 = pd.read_csv("val3.csv")
df12 = pd.read_csv("val4.csv")
df13 = pd.read_csv("val5.csv")

# 하나의 데이터프레임으로 통합
df_train = pd.concat([df0,df1,df2,df3,df4,df5,df6,df7])
df_valid = pd.concat([df8,df8,df10,df11,df12,df13])

# img_file_name을 기준으로 정렬
df_train = df_train.sort_values('img_file_name')
df_valid = df_valid.sort_values('img_file_name')

# 불필요 열 제거(img_file_name을 기준으로 정렬을 실시하여 기존에 있던 인덱스번호가 Unnamed:0로 저장됨)
df_train = df_train.drop(['Unnamed: 0'], axis = 1)
df_valid = df_valid.drop(['Unnamed: 0'], axis = 1)

# 인덱스 번호 재부여
df_train = df_train.reset_index()
df_valid = df_valid.reset_index()

# 기존 인덱스 번호 열 제거
df_train.drop(['index'], axis = 1)
df_valid.drop(['index'], axis = 1)

# 통합 데이터셋 저장
df_train.to_csv("train_img.csv")
df_valid.to_csv("val_img.csv")