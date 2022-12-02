from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, mean_squared_error, accuracy_score, r2_score

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

# 사과 종류별 train데이터 가져오기
sinano = pd.read_csv('apple_train_sinano.csv')
hongro = pd.read_csv('apple_train_hongro.csv')
arisu = pd.read_csv('apple_train_arisu.csv')
huji = pd.read_csv('apple_train_huji.csv')

# 사과 종류별 validation data 가져오기
sinano1 = pd.read_csv('apple_valid_sinano.csv')
hongro1 = pd.read_csv('apple_valid_hongro.csv')
arisu1 = pd.read_csv('apple_valid_arisu.csv')
huji1 = pd.read_csv('apple_valid_huji.csv')



############################################################
# 데이터 전처리
############################################################

# 훈련용, 검증용 사과데이터셋을 각각 하나의 데이터셋으로 통합
tr_apple = pd.concat([huji,sinano,arisu,hongro])
va_apple = pd.concat([huji1,sinano1,arisu1,hongro1])

# 필요없는 열 제거
tr_apple = tr_apple.drop(['Unnamed: 0','type','position','sugar_content','img_angle','licenses_id','licenses_name','img_file_name','img_height','img_width','area','segmentation','bbox','img_dist','sugar_content_nir','tod_attribute','img_attribute','img_time'], axis= 1)
va_apple = va_apple.drop(['Unnamed: 0','type','position','sugar_content','img_angle','licenses_id','licenses_name','img_file_name','img_height','img_width','area','segmentation','bbox','img_dist','sugar_content_nir','tod_attribute','img_attribute','img_time'], axis= 1)

# 중복행 제거
tr_apple = tr_apple.drop_duplicates()
va_apple = va_apple.drop_duplicates()

# 결측값이 존재하는 행 제거 (약 5%의 비율로 결측값이 존재하여 결측값을 대체가 아닌 제거를 선택)
tr_apple = tr_apple.dropna(axis=0)
va_apple = va_apple.dropna(axis=0)

# 인덱스 0부터 순차 재정렬
tr_apple = tr_apple.reset_index(drop = True)
va_apple = va_apple.reset_index(drop = True)

# 전처리 (원핫 인코딩을 위한 데이터 통합)
total = pd.concat([tr_apple,va_apple], axis =0, ignore_index = True)

# 원핫 인코딩 진행 (사과종류, 사진촬영일시, 일출시간, 일몰시간)
total = pd.get_dummies(data = total, columns = ['apple_kind'], prefix = 'apple_kind')
total = pd.get_dummies(data = total, columns = ['sunset_time'], prefix = 'sunset_time')
total = pd.get_dummies(data = total, columns = ['sunrise_time'], prefix = 'sunrise_time')

# 훈련용, 검증용 데이터셋 분리
tr_apple = total.iloc[:42348,:]
va_apple = total.iloc[42348:,:]

va_apple = va_apple.reset_index(drop = True)



############################################################
# train_X, train_y 분리
############################################################

train_y = tr_apple['sugar_grade']
train_X = tr_apple.drop(['sugar_grade'], axis = 1)

valid_y = va_apple['sugar_grade']
valid_X = va_apple.drop(['sugar_grade'], axis = 1)

# 데이터프레임 형태로 변환
train_y = pd.DataFrame(train_y)
valid_y = pd.DataFrame(valid_y)



############################################################
# 목표값 라벨 인코딩
############################################################

encoder = LabelEncoder()
encoder.fit(train_y)
train_y = encoder.transform(train_y)
valid_y = encoder.transform(valid_y)



############################################################
# 인공지능 모델 제작 및 학습
############################################################

from sklearn.ensemble import BaggingClassifier

model = BaggingClassifier(n_estimators = 120, random_state = 50)
model.fit(train_X, train_y)
model_pred = model.predict(valid_X)



############################################################
# 모델 성능 확인
############################################################

# 성적 지표
print(f'정확도: {accuracy_score(valid_y, model_pred): .4f}')
print('평균제곱오차: %.2f' % mean_squared_error(valid_y, model_pred))
print('결정계수: %.2f' % r2_score(valid_y, model_pred))

# 혼동행렬 히트맵
matrix = confusion_matrix(valid_y, model_pred)
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



############################################################
# 모델 저장하기
############################################################

import joblib
joblib.dump(model, "경로 설정하기/apple_model.pkl")




############################################################
# 모델 이용하기
############################################################

apple_m = joblib.load("모델 파일 경로")


############################################################
# test 샘플 준비
############################################################

# 실제 등급
answer = va_apple['sugar_grade']
answer = answer[0]

# 환경 조건
question = va_apple.drop(['sugar_grade'], axis = 1)
question = pd.DataFrame(question.iloc[0,:])
question = question.transpose()

# 등급 dictionary
grade_dict = {0:"A", 1:"B", 2:"C"}


############################################################
# 모델 적용
############################################################

apple_pred = apple_m.predict(question)[0]

# 결과 출력
print(f"사과 당도의 예상 등급은 <{grade_dict[apple_pred]}> 이고, 실제 등급은 <{answer}> 입니다.")