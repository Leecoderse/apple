from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, r2_score, f1_score, precision_score, recall_score

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

# 폰트 설정
import matplotlib.font_manager as fm
!apt-get -qq -y install fonts-nanum > /dev/null
fontpath = '/usr/share/fonts/truetype/nanum/NanumSquareRound.ttf'
font = fm.FontProperties(fname=fontpath, size=10)
fm._rebuild()
mpl.rc('font', family='NanumSquareRound') 
plt.rc('font', family='NanumSquareRound') 
mpl.rc('axes', unicode_minus=False)
plt.rc('axes', unicode_minus=False)



############################################################
# 데이터 불러오기
############################################################

# 사과 환경데이터 가져오기
train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')

# 사과 이미지데이터 가져오기
train_img = pd.read_csv('trainimg.csv')
val_img = pd.read_csv('valimg.csv')



############################################################
# 데이터 전처리
############################################################

# 환경데이터 img_file_name기준으로 재정렬
train = train.sort_values('img_file_name')
val = val.sort_values('img_file_name')

# 인덱스 번호 재부여
train = train.reset_index()
val = val.reset_index()

# 불필요 열 제거
train = train.drop(['index'], axis = 1)
val = val.drop(['index'], axis = 1)

# 불필요 열 제거
train_img = train_img.drop(['Unnamed: 0', 'index'], axis= 1)
val_img = val_img.drop(['Unnamed: 0', 'index'], axis= 1)
train = train.drop(['Unnamed: 0'], axis = 1)
val = val.drop(['Unnamed: 0'], axis = 1)

#데이터 프레임 통합
train = pd.merge(train, train_img , on ='img_file_name')
val = pd.merge(val, val_img, on ='img_file_name')

# 전처리를 위해 훈련용과 검증용 통합
total = pd.concat([train,val], axis = 0, ignore_index = True)

# 필요없는 열 제거(세그멘테이션 좌표, 당도등급 제외 모든 불필요 특성 제거)
total = total.drop(['type','position','sugar_content','img_angle','licenses_id','licenses_name','img_height','img_width','area','segmentation_x','bbox','img_dist','sugar_content_nir','apple_kind','obj_num','tod_attribute','tod_temper','soil_ec','soil_temper','soil_humidty','soil_potential','temperature','humidity','sunshine','sunrise_time','sunset_time','img_attribute','img_time', 'img_file_name'], axis= 1)

# 원핫 인코딩 (segmentation(좌표))
total = pd.get_dummies(data = total, columns = ['segmentation_y'], prefix = 'segmentation')

# 훈련용, 검증용 데이터셋 분리
train = total.iloc[:546,:]
val = total.iloc[546:,:]

# 검증용 인덱스번호 재부여 및 불필요 열 제거
val = val.reset_index()
val = val.drop(['index'], axis = 1)



############################################################
# train_X, train_y 분리
############################################################

train_y = train['sugar_grade']
train_X = train.drop(['sugar_grade'], axis = 1)

val_y = val['sugar_grade']
val_X = val.drop(['sugar_grade'], axis = 1)

# 데이터프레임 형태로 변환
train_y = pd.DataFrame(train_y)
val_y = pd.DataFrame(val_y)



############################################################
# 목표값 라벨 인코딩
############################################################

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoder.fit(train_y)
train_y = encoder.transform(train_y)
val_y = encoder.transform(val_y)



############################################################
# 인공지능 모델 제작 및 학습
############################################################

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators = 120, random_state = 10)
model.fit(train_X, train_y)

pred_y = model.predict(val_X)



############################################################
# 모델 성능 확인
############################################################

# 성적 지표
print('평균제곱오차: %.2f' % mean_squared_error(val_y, pred_y))
print(f'정확도 = {accuracy_score(val_y, pred_y): .4f}\n')
print('f1 score: %.2f' % f1_score(val_y, pred_y, average = 'micro'))
print('정밀도: %.2f' % precision_score(val_y, pred_y, average = 'micro'))
print('재현율: %.2f' % recall_score(val_y, pred_y, average = 'micro'))

# 혼동행렬 히트맵
matrix = confusion_matrix(val_y, pred_y)
등급 = ['A','B','C'] 

sns.heatmap(data = matrix, 
            annot=True,
            annot_kws={"size": 14},
            cmap= 'cool',
            xticklabels= 등급,
            yticklabels= 등급,
            fmt = 'd')
plt.xlabel('예측치', fontsize=14)
plt.ylabel('실측치', fontsize=14)

plt.suptitle("사과 당도 예측", y=1.04, size=18)
plt.show()


