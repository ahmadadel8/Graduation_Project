from __future__ import division
import tensorflow as tf
import tensornets as nets
import numpy as np
import matplotlib.pyplot as plt
import math
from IPython.display import clear_output
import random
import cv2
from copy import copy, deepcopy
from pathlib import Path
import os
import time 
from datetime import timedelta
from tqdm import tqdm
#import zipfile
import tarfile
import shutil
import wget
import sys
import voc
from utils import model 
import time 
from datetime import timedelta


tf.reset_default_graph() # It's importat to resume training from latest checkpoint 


N_classes=20
is_training = tf.placeholder(tf.bool)
x = tf.placeholder(tf.float32, shape=(None, 416, 416, 3), name='input_x')

YOLF_V1=model(x,nets.MobileNet25,'voc')

YOLF_V2=model(x,nets.MobileNet100v2, 'voc')

YOLF_V2_50=model(x,nets.MobileNet50v2, 'voc')


TinyYOLOv2=nets.TinyYOLOv2VOC(x, is_training=False)
YOLOv2=nets.YOLOv2COCO(x, is_training=False)
YOLOv3=nets.YOLOv3VOC(x, is_training=False)

t_diff_YOLF_V1=[]
t_diff_YOLF_V2=[]
t_diff_YOLF_V2_50=[]


t_diff_TinyYOLOv2=[]
t_diff_YOLOv2=[]
t_diff_YOLOv3=[]

voc_dir = '/home/alex054u4/data/nutshell/newdata/VOCdevkit/VOC%d'

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())


    acc_data  = voc.load(voc_dir % 2007, 'test', total_num=1000)

    print("TESTING STARTING.")


    for (img,_) in acc_data:

        ts=time.time()
        acc_outs = sess.run(YOLF_V1, {x: YOLF_V1.preprocess(img),is_training: False})
        t_diff_YOLF_V1.append(time.time()-ts)

        ts=time.time()
        acc_outs = sess.run(YOLF_V2, {x: YOLF_V2.preprocess(img),is_training: False})
        t_diff_YOLF_V2.append(time.time()-ts)

        ts=time.time()
        acc_outs = sess.run(YOLF_V2_50, {x: YOLF_V2_50.preprocess(img),is_training: False})
        t_diff_YOLF_V2_50.append(time.time()-ts)


        ts=time.time()
        acc_outs = sess.run(TinyYOLOv2, {x: TinyYOLOv2.preprocess(img),is_training: False})
        t_diff_TinyYOLOv2.append(time.time()-ts)

        ts=time.time()
        acc_outs = sess.run(YOLOv2, {x: YOLOv2.preprocess(img),is_training: False})
        t_diff_YOLOv2.append(time.time()-ts)


        ts=time.time()
        acc_outs = sess.run(YOLOv3, {x: YOLOv3.preprocess(img),is_training: False})
        t_diff_YOLOv3.append(time.time()-ts)


    print("TESTING DONE.")

    print("=============================================")

    print("YOLF_V1 FPS:", 1.0/np.mean(t_diff_YOLF_V1))

    print("=============================================")
    
    print("YOLF_V2 FPS:", 1.0/np.mean(t_diff_YOLF_V2))

    print("=============================================")

    print("YOLF_V2_50 FPS:", 1.0/np.mean(t_diff_YOLF_V2_50))

    print("=============================================")

    print("TinyYOLOv2 FPS:", 1.0/np.mean(t_diff_TinyYOLOv2))


    print("=============================================")

    print("YOLOv2 FPS:", 1.0/np.mean(t_diff_YOLOv2))

    print("=============================================")

    print("YOLOv3 FPS:", 1.0/np.mean(t_diff_YOLOv3))

