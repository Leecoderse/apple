'''
# 버전 다운그레이드가 필요합니다

tensorflow : 1.13.1
keras : 2.1.5
h5py : 2.10.0
'''

# 필요한 라이브러리 import
import warnings
warnings.filterwarnings(action='ignore')

import os
import sys
import itertools
import math
import logging
import json
import re
import random

from collections import OrderedDict

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon

import skimage.io
import tensorflow as tf

# Root directory 경로 설정
ROOT_DIR = os.path.abspath('경로 설정하기')

# model과 log를 저장할 경로 설정
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Import Mask RCNN
sys.path.append(ROOT_DIR)

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log

%matplotlib inline

# import 메인 코드
import apple as am



############################################################
# Configuration
############################################################

class InferenceConfig(am.AppleConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

DEVICE = "/cpu:0"
TEST_MODE = "inference"



############################################################
# Load Model and Weights
############################################################

# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# 가중치 파일 경로
weights_path = '경로 설정하기'

# 가중치 load
model.load_weights(weights_path, by_name=True)



############################################################
# 데이터 불러오기
############################################################

# 모델 적용할 이미지 폴더 경로
SAMPLE_IMG_PATH = '경로 설정하기'

# 이미지 데이터 리스트 생성 및 개수 확인
image_list = os.listdir(SAMPLE_IMG_PATH)
image_list_py = [file for file in image_list if file.endswith('.jpg')]
print("Image Count: {}".format(len(image_list_py)))

# 이미지에 담긴 class 종류 2가지 - BackGround, Apple
class_names = ['BG', 'apple']



############################################################
# 데이터 리스트 분할하기
############################################################

# 이미지 데이터 리스트 분할하는 함수
def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]

# 이미지 데이터 리스트 80개씩 분할
image_list_py_80 = list_chunk(image_list_py, 80)



############################################################
# 모델 적용 예시
############################################################

# 샘플이미지 임의로 1개 선택
image_name = random.choice(image_list_py)

# 본인 환경에 맞게 경로를 수정해야 함
image = skimage.io.imread('../경로 수정하기' + image_name)
print("Image Name : {}".format(image_name))

# 모델 적용
results = model.detect([image], verbose=1)
r = results[0]

# 모델 적용 결과 보여주기
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], title="Sample Image Predictions")



############################################################
# Mask 좌표 추출
############################################################

temp_ = {}
num = 0

for i in image_list_py_80[x]:       # 이미지 데이터 리스트 번호 지정

    # 본인 환경에 맞게 이미지 경로를 수정해야 함
    temp = skimage.io.imread('../경로 수정하기' + i)
    print("Image Name : {}".format(i))

    # 모델 적용
    results = model.detect([temp], verbose=1)
    r = results[0]

    # 모델의 Mask 좌표 0과 1로 수치화
    mask = r['masks']
    mask = mask.astype(int)

    # Mask 좌표 추출하기
    for i in range(mask.shape[2]):

        # 이미지와 Mask 크기 맞추기 (1024,1024, )
        temp, window, scale, padding, _ = utils.resize_image(temp, 
                                                            min_dim=config.IMAGE_MIN_DIM, 
                                                            max_dim=config.IMAGE_MAX_DIM, 
                                                            mode=config.IMAGE_RESIZE_MODE)
        mask, window, scale, padding, _ = utils.resize_image(mask, 
                                                            min_dim=config.IMAGE_MIN_DIM, 
                                                            max_dim=config.IMAGE_MAX_DIM, 
                                                            mode=config.IMAGE_RESIZE_MODE)
        # Mask 적용된 부분만 추출
        for j in range(temp.shape[2]):
            temp[:,:,j] = temp[:,:,j] * mask[:,:,i]
        
        # 추출된 Mask 좌표에서 0이 아닌 부분만 저장
        temp_[num] = (temp>0).nonzero() 

        print(f'\n{num}번째 사과 이미지 mask 좌표 추출 완료\n\n')        
        num += 1



############################################################
# Mask 좌표 전처리
############################################################

# 첫번째 Mask 좌표 전처리
temp_[0] = np.concatenate((temp_[0][0],temp_[0][1],temp_[0][2]), axis = 0)
temp_[0] = np.array2string(temp_[0])
temp_[0] = {'segmentation':temp_[0]}

# Mask 좌표 담는 데이터프레임 생성
df = pd.DataFrame(temp_[0], index = [0])

# 나머지 Mask 좌표 데이터프레임에 추가
for i in range(1,len(image_list_py_80[x])):     # 위에서 Mask 좌표 추출했던 이미지 데이터 리스트 번호
    temp_[i] = np.concatenate((temp_[i][0],temp_[i][1],temp_[i][2]), axis = 0)
    temp_[i] = np.array2string(temp_[i])
    temp_[i] = {'segmentation':temp_[i]}
    df.loc[i] = temp_[i]

# 데이터프레임에 '이미지 이름' 열 추가
df['img_file_name'] = image_list_py_80[5]

# csv형태로 저장
df.to_csv("파일 이름.csv")


