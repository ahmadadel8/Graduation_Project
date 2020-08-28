import cv2

from tensornets import MobileNet25
import numpy as np

from utils import model 
import time 
from datetime import timedelta
import os 
import tensorflow.compat.v1 as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
tf.disable_v2_behavior()


x = tf.placeholder(tf.float32, shape=(None, 416, 416, 3), name='input_x')

N_classes=20
is_training = tf.placeholder(tf.bool)


YOLF_25=model(x,MobileNet25,'voc','sep', scope='YOLF_25')
#YOLF_50=model(x,nets.MobileNet50, 'voc','sep', scope='YOLF_50')
#YOLO_MOB=nets.YOLOv2(x,nets.MobileNet50v2,is_training=False,  scope='YOLF_50_tiny')


#TinyYOLOv2=nets.TinyYOLOv2VOC(x, is_training=False)
#YOLOv2=YOLOv2COCO(x, is_training=False)
#YOLOv3=nets.YOLOv3VOC(x, is_training=False)

t_diff_YOLF_25=[]
#t_diff_YOLF_50=[]
#t_diff_YOLO_MOB=[]


#t_diff_TinyYOLOv2=[]
#t_diff_YOLOv2=[]
#t_diff_YOLOv3=[]

#voc_dir = '/home/alex054u4/data/nutshell/newdata/VOCdevkit/VOC%d'
config= tf.ConfigProto()
config.gpu_options.allow_growth= True
config.gpu_options.per_process_gpu_memory_fraction = 0.4

sess= tf.Session(config=config)
sess.run(tf.global_variables_initializer())


print("TESTING STARTING.")	
img=np.zeros((1, 416,416,3))
checkpoint_path   =  '/home/grad/Desktop/training_trial_YOLF'
checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")
saver = tf.train.import_meta_graph('/home/grad/Desktop/training_trial_YOLF/ckpt.meta')
# Then restore your training data from checkpoint files:
saver.restore(sess, checkpoint_prefix)
# Finally, freeze the graph:
frozen_graph = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        output_node_names=['genYOLOv2/linear/conv/BiasAdd:0'])
        
converter = trt.TrtGraphConverter(
	input_graph_def=frozen_graph,
	nodes_blacklist=['genYOLOv2/linear/conv/BiasAdd:0'])
frozen_graph = converter.convert()
output_node = tf.import_graph_def(
	trt_graph,
	return_elements=['genYOLOv2/linear/conv/BiasAdd:0'])

for i in range(100):        
	ts=time.time()
	acc_outs = sess.run(output_node, {x: YOLF_25.preprocess(img),is_training: False})
	t_diff_YOLF_25.append(time.time()-ts)




#     ts=time.time()
#    acc_outs = sess.run(YOLF_50, {x: YOLF_50.preprocess(img),is_training: False})
#   t_diff_YOLF_50.append(time.time()-ts)

#  ts=time.time()
# acc_outs = sess.run(YOLO_MOB, {x: YOLO_MOB.preprocess(img),is_training: False})
#t_diff_YOLO_MOB.append(time.time()-ts)


#   ts=time.time()
#  acc_outs = sess.run(TinyYOLOv2, {x: TinyYOLOv2.preprocess(img),is_training: False})
# t_diff_TinyYOLOv2.append(time.time()-ts)

#ts=time.time()
#acc_outs = sess.run(YOLOv2, {x: YOLOv2.preprocess(img),is_training: False})
#t_diff_YOLOv2.append(time.time()-ts)


#ts=time.time()
#acc_outs = sess.run(YOLOv3, {x: YOLOv3.preprocess(img),is_training: False})
#t_diff_YOLOv3.append(time.time()-ts)


print("TESTING DONE.")

print("=============================================")

print("YOLF_25 FPS:", 1.0/np.mean(t_diff_YOLF_25))

print("=============================================")

#print("YOLF_50 FPS:", 1.0/np.mean(t_diff_YOLF_50))

#print("=============================================")

#print("YOLO_MOB FPS:", 1.0/np.mean(t_diff_YOLO_MOB))

#print("=============================================")

#print("TinyYOLOv2 FPS:", 1.0/np.mean(t_diff_TinyYOLOv2))


#print("=============================================")

#print("YOLOv2 FPS:", 1.0/np.mean(t_diff_YOLOv2))

#print("=============================================")

#print("YOLOv3 FPS:", 1.0/np.mean(t_diff_YOLOv3))


