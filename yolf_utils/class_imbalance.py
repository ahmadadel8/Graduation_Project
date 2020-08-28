from yolf_data import *
import numpy as np
from data_aug.data_aug import *
from data_aug.bbox_util import *

data_dir = '/content/TRAINdevKit/train/train'
data_name='trainval'

with open(os.path.join(os.path.dirname(__file__), 'yolf.names'), 'r') as f:
    classnames = [line.rstrip() for line in f.readlines()]
classes_num = len(classnames)


data=load(data_dir, data_name)
files = get_files(data_dir, data_name)
annotations = get_annotations(data_dir, files)
_annotations={}
img_classes =[[] for _ in range(classes_num)]

for idx, (file_id, ann) in enumerate(annotations.items()):
	_annotations[file_id]=[]
	for c, obj in enumerate(ann):
		_annotations[file_id].append([obj['bbox'], c])
		if c not in img_classes[idx]:
			img_classes[idx].append(c)

def count_instances(data,annotations, print_flag=True):
	
	class_counter=[0]*classes_num
	for idx, (img,_) in enumrate(data):
		for box in _annotations[file_id=files[i]]:
			class_counter[box[4]]+=1
	if print_flag:
		print("==================================================")
		for i,count in enumerate(class_counter):
			print("Class ", i, " has appeared a total of ", count, "times in the data set. \n")
		print("The mean of the classes instances is: ", np.mean(class_counter), "with a standard deviation of: ", np.std(class_counter))

	return class_counter

def fix_imbalance(data,annotations, std_limit):

	classes 	  = [i for i in range(classes_num)]
	class_counter = count_instances(data,annotations, print_flag=False)
	max_class     = np.argmax(class_counter)
	classes.pop(max_class)
	init_std	  = np.std(class_counter)

	final_imgs=[]
	final_boxes=[]

	while(np.std(class_counter) > std_limit):
		for idx, (img,_) in enumrate(data):

			final_imgs.append(img)
			final_boxes.append(_annotations[files[idx]])

			if max_class not in img_classes[idx]:

				boxes=np.array(_annotations[files[idx]], dtype=np.float64)
				transforms = Sequence([RandomHSV(40, 40, 30), RandomHorizontalFlip(0.5),RandomTranslate(np.random.uniform(0,0.2), diff = True), RandomShear(np.random.uniform(-0.5,0.5))])
				_x, _boxes = transforms(x.copy(), boxes.copy())
				final_imgs.append(_x)
				final_boxes.append(_boxes)

				for _box in _boxes:
					class_counter[_box[4]]+=1

	print("\n ================================================================== \n")
	print("DONE")
	for i,count in enumerate(class_counter):
		print("Class ", i, " has appeared a total of ", count, "times in the data set. \n")
	print("The mean of the classes instances is: ", np.mean(class_counter), "with a standard deviation of: ", np.std(class_counter))

	return final_imgs, final_boxes