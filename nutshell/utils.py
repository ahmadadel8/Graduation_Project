from __future__ import absolute_import
from __future__ import division

import os
import tensorflow as tf
import numpy as np
import tensornets as nets
#from stem import *
from tensorflow.keras.utils import plot_model
from tensornets.references.yolo_utils import get_v2_boxes, v2_loss, v2_inputs
from tensornets.preprocess import darknet_preprocess as preprocess
from tensornets.layers import darkconv
import os

def darkdepthsepconv(inputs, filters, kernel, scope, lmbda=5e-4, dropout_rate=0):
  with tf.name_scope(scope):
    x = tf.keras.layers.DepthwiseConv2D(kernel, depth_multiplier=1, padding='same', use_bias=False, name=scope+'/sconv', kernel_regularizer=tf.keras.regularizers.l2(lmbda),kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.53846))(inputs)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-5, center=False, scale=True, name=scope+'/bnd')(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)

    x = tf.keras.layers.Conv2D(filters, 1, padding='same', use_bias=False, name=scope+'/conv', kernel_regularizer=tf.keras.regularizers.l2(lmbda),kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1.53846))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-5, center=False, scale=True, name=scope+'/bns')(x)
    x = tf.nn.bias_add(x, tf.Variable(tf.random_normal([filters])), name= scope+'bias_add')
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    x = tf.keras.layers.Dropout(rate=dropout_rate)(x, training=True)
    return x
    


with open(os.path.join(os.path.dirname(__file__), 'coco.names'), 'r') as f:
  labels_coco = [line.rstrip() for line in f.readlines()]

with open(os.path.join(os.path.dirname(__file__), 'voc.names'), 'r') as f:
  labels_voc = [line.rstrip() for line in f.readlines()]

bases = dict()
bases['coco'] = {'anchors': [0.57273, 0.677385, 1.87446, 2.06253, 3.33843,
                             5.47434, 7.88282, 3.52778, 9.77052, 9.16828]}
bases['voc'] = {'anchors': [1.3221, 1.73145, 3.19275, 4.00944, 5.05587,
                                8.09892, 9.47112, 4.84053, 11.2364, 10.0071]}

def meta(dataset_name='voc'):


  opt = bases[dataset_name].copy()
  opt.update({'num': len(opt['anchors']) // 2})
  if dataset_name=='voc':
      opt.update({'classes': len(labels_voc), 'labels': labels_voc})
  elif dataset_name=='coco':
      opt.update({'classes': len(labels_coco), 'labels': labels_coco})
  else: raise Exception ('Dataset not supported')
  return opt
  

def model(inputs, stem_fn, dataset_name, yolo_head='sep', scope='stem' ,is_training=True): 
  metas=meta(dataset_name)
  N_classes=metas['classes']

  with tf.name_scope('stem'):
    x = stem =  stem_fn(inputs, is_training=True, stem=True,  scope=scope) #bulding the model



  p = x.p

  if (yolo_head=='sep'): 
  	conv=darkdepthsepconv
  elif (yolo_head=='dark'):
  	conv=darkconv

  x = conv(x, 1024, 3, scope='genYOLOv1/conv7')
  x = conv(x, 1024, 3, scope='genYOLOv1/conv8')
  p = conv(p, 64, 1, scope='genYOLOv1/conv5a')
  p = tf.reshape(p,[-1, 13,13,256], name='flat5a')
  x = tf.concat([p, x], axis=3, name='concat')

  x = conv(x, 1024, 3, scope='genYOLOv1/conv9')
  x = tf.keras.layers.Conv2D((N_classes+ 5) * 5, 1, kernel_regularizer=tf.keras.regularizers.l2(), padding='same', name='genYOLOv2/linear/conv')(x)
  x.aliases = []

  def get_boxes(*args, **kwargs):
  	return get_v2_boxes(metas, *args, **kwargs)
  x.get_boxes = get_boxes
  x.stem = stem
  x.inputs = [inputs]
  x.inputs += v2_inputs(x.shape[1:3], metas['num'], N_classes, x.dtype)
  if isinstance(is_training, tf.Tensor):
      x.inputs.append(is_training)
  x.loss = v2_loss(x, metas['anchors'], N_classes)
  def preprocess_(*args, **kwargs):
  	return preprocess(target_size=(416,416), *args, **kwargs)
  x.preprocess=preprocess_
  return x

