# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Wed Jul 12 15:53:44 2017

@author: Walid Benbihi
@mail : w.benbihi(at)gmail.com
@github : https://github.com/wbenbihi/hourglasstensorlfow/

Abstract:
        This python code creates a Stacked Hourglass Model
        (Credits : A.Newell et al.)
        (Paper : https://arxiv.org/abs/1603.06937)

        Code translated from 'anewell' github
        Torch7(LUA) --> TensorFlow(PYTHON)
        (Code : https://github.com/anewell/pose-hg-train)

        Modification are made and explained in the report
        Goal : Achieve Real Time detection (Webcam)
        ----- Modifications made to obtain faster results (trade off speed/accuracy)

        This work is free of use, please cite the author if you use it!

"""
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import time
from skimage import transform
import scipy.misc as scm
from imageio import imread

class DataGenerator():
	""" DataGenerator Class : To generate Train, Validatidation and Test sets
	for the Deep Human Pose Estimation Model
	Formalized DATA:
		Inputs:
			Inputs have a shape of (Number of Image) X (Height: 256) X (Width: 256) X (Channels: 3)
		Outputs:
			Outputs have a shape of (Number of Image) X (Number of Stacks) X (Heigth: 64) X (Width: 64) X (OutputDimendion: 16)
	Joints:
		We use the MPII convention on joints numbering
		List of joints:
			00 - Right Ankle
			01 - Right Knee
			02 - Right Hip
			03 - Left Hip
			04 - Left Knee
			05 - Left Ankle
			06 - Pelvis (Not present in other dataset ex : LSP)
			07 - Thorax (Not present in other dataset ex : LSP)
			08 - Neck
			09 - Top Head
			10 - Right Wrist
			11 - Right Elbow
			12 - Right Shoulder
			13 - Left Shoulder
			14 - Left Elbow
			15 - Left Wrist
	# TODO : Modify selection of joints for Training

	How to generate Dataset:
		Create a TEXT file with the following structure:
			image_name.jpg[LETTER] box_xmin box_ymin box_xmax b_ymax joints
			[LETTER]:
				One image can contain multiple person. To use the same image
				finish the image with a CAPITAL letter [A,B,C...] for
				first/second/third... person in the image
 			joints :
				Sequence of x_p y_p (p being the p-joint)
				/!\ In case of missing values use -1

	The Generator will read the TEXT file to create a dictionnary
	Then 2 options are available for training:
		Store image/heatmap arrays (numpy file stored in a folder: need disk space but faster reading)
		Generate image/heatmap arrays when needed (Generate arrays while training, increase training time - Need to compute arrays at every iteration)
	"""
	def __init__(self, img_dir=None, train_data_file = None, num_out = None, val_dir=None, val_data_file=None, test_dir = None, test_data_file=None, resolutions=False, headed = False, head_train = None, head_test = None, head_val = None):
		""" Initializer
		Args:
			joints_name			: List of joints condsidered
			img_dir				: Directory containing every images
			train_data_file		: Text file with training set data
			remove_joints		: Joints List to keep (See documentation)
		"""


		self.num_out = num_out
		self.img_dir = img_dir
		self.train_data_file = train_data_file
		self.val_data_file = val_data_file
		self.val_dir = val_dir
		self.val_images = os.listdir(val_dir)
		self.images = os.listdir(img_dir)
		self.test_dir = test_dir
		self.test_data_file = test_data_file
		self.resolutions = resolutions
		self.headed = headed
		if self.headed:
			self.head_train = head_train
			self.head_test = head_test
			self.head_val = head_val
			
		print("IS resolutions activated: {} and mixed maps {}?".format(self.resolutions,self.headed))

	# --------------------Generator Initialization Methods ---------------------

	def _create_test_table(self):
		""" Create Table of samples from TEXT file
		"""
		self.test_table = []
		self.data_dict_test = {}
		input_file = open(self.test_data_file, 'r')
		print('READING TRAIN DATA')
		for line in input_file:
			line = line.strip()
			line = line.split(' ')
			name = line[0]
			gt_name = line[1]

			self.data_dict_test[name] = {'gt_name' : gt_name}
			self.test_table.append(name)
                #lista de diccionarios con caja, coordin. de joints y weights de cada uno de los joints
		input_file.close()
		
		############################################test joints
		if self.headed:
			input_file = open(self.head_test, 'r')
			print('READING TRAIN DATA')
			for line in input_file:
				line = line.strip()
				line = line.split(' ')
				name = line[0]
				joints = list(map(int,line[1:]))
				joints = np.reshape(joints, (-1,2))
	
				self.data_dict_test[name]['joints'] = joints
	                #lista de diccionarios con caja, coordin. de joints y weights de cada uno de los joints
			input_file.close()
			
		########################################################33
		random.shuffle(self.test_table)

		self.test_set = self.test_table
		print('--Testing set :', len(self.test_set), ' samples.')

	def _create_train_table(self):
		""" Create Table of samples from TEXT file
		"""
		
		#########################################train segmentation
		self.train_table = []
		self.data_dict = {}
		input_file = open(self.train_data_file, 'r')
		print('READING TRAIN DATA')
		for line in input_file:
			line = line.strip()
			line = line.split(' ')
			name = line[0]
			gt_name = line[1]

			self.data_dict[name] = {'gt_name' : gt_name}
			self.train_table.append(name)
                #lista de diccionarios con caja, coordin. de joints y weights de cada uno de los joints
		input_file.close()
		
		############################################3train joints
		if self.headed:
			input_file = open(self.head_train, 'r')
			print('READING TRAIN DATA')
			for line in input_file:
				line = line.strip()
				line = line.split(' ')
				name = line[0]
				joints = list(map(int,line[1:]))
				joints = np.reshape(joints, (-1,2))
	
				self.data_dict[name]['joints'] = joints
	                #lista de diccionarios con caja, coordin. de joints y weights de cada uno de los joints
			input_file.close()

		###########################################validation segmentations
		self.val_table = []
		self.data_dict_val = {}
		input_file = open(self.val_data_file, 'r')
		print('READING TRAIN DATA')
		for line in input_file:
			line = line.strip()
			line = line.split(' ')
			name = line[0]
			gt_name = line[1]

			self.data_dict_val[name] = {'gt_name' : gt_name}
			self.val_table.append(name)
                #lista de diccionarios con caja, coordin. de joints y weights de cada uno de los joints
		input_file.close()
		
		############################################validation joints
		if self.headed:
			input_file = open(self.head_val, 'r')
			print('READING TRAIN DATA')
			for line in input_file:
				line = line.strip()
				line = line.split(' ')
				name = line[0]
				joints = list(map(int,line[1:]))
				joints = np.reshape(joints, (-1,2))
	
				self.data_dict_val[name]['joints'] = joints
	                #lista de diccionarios con caja, coordin. de joints y weights de cada uno de los joints
			input_file.close()


	def _randomize(self):
		""" Randomize the set
		"""
		random.shuffle(self.train_table)


	def _give_batch_name(self, batch_size = 16, set = 'train'):
		""" Returns a List of Samples
		Args:
			batch_size	: Number of sample wanted
			set				: Set to use (valid/train)
		"""
		list_file = []
		for i in range(batch_size):
			if set == 'train':
				list_file.append(random.choice(self.train_set))
			elif set == 'valid':
				list_file.append(random.choice(self.valid_set))
			else:
				print('Set must be : train/valid')
				break
		return list_file


	def _create_sets(self, validation_rate = 0.1):
		""" Select Elements to feed training and validation set
		Args:
			validation_rate		: Percentage of validation data (in ]0,1[, don't waste time use 0.1)
		"""
		self.train_set = self.train_table
		self.valid_set = self.val_table

		print('SET CREATED')
#		np.save('Dataset-Validation-Set', self.valid_set)
#		np.save('Dataset-Training-Set', self.train_set)
		print('--Training set :', len(self.train_set), ' samples.')
		print('--Validation set :', len(self.valid_set), ' samples.')

	def generateSet(self, rand = False):
		""" Generate the training and validation set
		Args:
			rand : (bool) True to shuffle the set
		"""
		self._create_train_table()
		if rand:
			self._randomize()
		self._create_sets()

	# ---------------------------- Generating Methods --------------------------

	def _augment(self,img, hm, max_rotation = 30):
		""" # TODO : IMPLEMENT DATA AUGMENTATION
		"""
		if random.choice([0,1]):
			r_angle = np.random.randint(-1*max_rotation, max_rotation)
			img = 	transform.rotate(img, r_angle, preserve_range = True)
			hm = transform.rotate(hm, r_angle)
		return img, hm

	# ----------------------- Batch Generator ----------------------------------

	def _generator(self, batch_size = 16, stacks = 4, set = 'train', stored = False, normalize = True, debug = False):
		""" Create Generator for Training
		Args:
			batch_size	: Number of images per batch
			stacks			: Number of stacks/module in the network
			set				: Training/Testing/Validation set # TODO: Not implemented yet
			stored			: Use stored Value # TODO: Not implemented yet
			normalize		: True to return Image Value between 0 and 1
			_debug			: Boolean to test the computation time (/!\ Keep False)
		# Done : Optimize Computation time
			16 Images --> 1.3 sec (on i7 6700hq)
		"""
		while True:
			if debug:
				t = time.time()
			train_img = np.zeros((batch_size, 256,256,3), dtype = np.float32)
			train_gtmap = np.zeros((batch_size, stacks, 64, 64, len(self.joints_list)), np.float32)
			files = self._give_batch_name(batch_size= batch_size, set = set)
			for i, name in enumerate(files):
				if name[:-1] in self.images:
					try :
						img = self.open_img(name)
						joints = self.data_dict[name]['joints']
						box = self.data_dict[name]['box']
						weight = self.data_dict[name]['weights']
						if debug:
							print(box)
						padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp = 0.2)
						if debug:
							print(cbox)
							print('maxl :', max(cbox[2], cbox[3]))
						new_j = self._relative_joints(cbox,padd, joints, to_size=64)
						hm = self._generate_hm(64, 64, new_j, 64, weight)
						img = self._crop_img(img, padd, cbox)
						img = img.astype(np.uint8)
						# On 16 image per batch
						# Avg Time -OpenCV : 1.0 s -skimage: 1.25 s -scipy.misc.imresize: 1.05s
						img = scm.imresize(img, (256,256))
						# Less efficient that OpenCV resize method
						#img = transform.resize(img, (256,256), preserve_range = True, mode = 'constant')
						# May Cause trouble, bug in OpenCV imgwrap.cpp:3229
						# error: (-215) ssize.area() > 0 in function cv::resize
						#img = cv2.resize(img, (256,256), interpolation = cv2.INTER_CUBIC)
						img, hm = self._augment(img, hm)
						hm = np.expand_dims(hm, axis = 0)
						hm = np.repeat(hm, stacks, axis = 0)
						if normalize:
							train_img[i] = img.astype(np.float32) / 255
						else :
							train_img[i] = img.astype(np.float32)
						train_gtmap[i] = hm
					except :
						i = i-1
				else:
					i = i - 1
			if debug:
				print('Batch : ',time.time() - t, ' sec.')
			yield train_img, train_gtmap


	def _transform_labels(self, image, classes, class_list,class_values):
		for i,j in zip(classes,class_list):
			   for h in i:
				      image = np.where(image ==h,j,image)
		for i in range(1,len(class_list)+1):
			   image = np.where(image == class_list[i-1],class_values[i-1], image)

		return image.astype(np.int64)

	def _give_resolutions(self, ground, image, max_rotation=30):

		gt_1 = np.where(ground != 0, 1, 0)

		classes_1 = [[16,13],
          [7,10,14,15],
		  [1,4],
          [17,18,19,20,21,22,23,24],
          [2,3,5,6,8,9,11,12]]
		class_list_1 = [25,26,27,28,29]
		class_values_1 = [1,2,3,4,5]

		gt_2 = self._transform_labels(ground, classes_1,class_list_1, class_values_1)

		classes_2 = [[16,13],
          [8,11],
          [9,12],
          [21,23],
          [22,24],
          [17,19],
          [18,20],
          [2,5],
          [3,6],
          [14,15,10,7],
          [1,4]]
		class_list_2 = [25,26,27,28,29,30,31,32,33,34,35]
		class_values_2 = [1,2,3,4,5,6,7,8,9,10,11]

		gt_3 = self._transform_labels(ground, classes_2,class_list_2, class_values_2)

#		ground = scm.imresize(ground,(64,64),'nearest', mode='F').astype(np.int64)
#		gt_1 = scm.imresize(gt_1,(64,64),'nearest', mode='F').astype(np.int64)
#		gt_2 = scm.imresize(gt_2,(64,64),'nearest', mode='F').astype(np.int64)
#		gt_3 = scm.imresize(gt_3,(64,64),'nearest', mode='F').astype(np.int64)

		
		r_angle = np.random.randint(-1*max_rotation, max_rotation)
		image = 	transform.rotate(image, r_angle, preserve_range = True)
		ground = transform.rotate(ground, r_angle, preserve_range = True).astype(np.int64)
		gt_1 = transform.rotate(gt_1, r_angle, preserve_range = True).astype(np.int64)
		gt_2 = transform.rotate(gt_2, r_angle, preserve_range = True).astype(np.int64)
		gt_3 = transform.rotate(gt_3, r_angle, preserve_range = True).astype(np.int64)

		f_gt = np.stack([gt_1,gt_2,gt_3,ground])

		return f_gt


	def _aux_generator(self, batch_size = 16, stacks = 4, normalize = True, sample_set = 'train', head_stacks = 2):
		""" Auxiliary Generator
		Args:
			See Args section in self._generator
		"""
		while True:
			train_img = np.zeros((batch_size, 320,320,3), dtype = np.float32)
			if self.headed:
				train_gtmap_t = np.zeros((batch_size, stacks, 320, 320), np.int32)
				train_gtmap_h = np.zeros((batch_size, head_stacks, 320, 320, 24), np.float32)
			else:
				train_gtmap = np.zeros((batch_size, stacks, 320, 320), np.int32)
			train_weights = np.zeros((batch_size, self.num_out), np.float32)
			i = 0
			while i < batch_size:
#				try:

					if sample_set == 'train':
						name = random.choice(self.train_set)
						gt = self.data_dict[name]['gt_name']
						if self.headed:
							coords = self.data_dict[name]['joints']  ### uncomment corresponding lines if online joints
						gt_name = os.path.join(self.img_dir, gt)
						gt = imread(gt_name)
						name = os.path.join(self.img_dir, name)
						
					elif sample_set == 'valid':
						name = random.choice(self.valid_set)
						gt = self.data_dict_val[name]['gt_name']
						if self.headed:
							coords = self.data_dict_val[name]['joints']
						gt_name = os.path.join(self.val_dir, gt)
						gt = imread(gt_name)
						name = os.path.join(self.val_dir, name)
						
					else:
						name = random.choice(self.test_set)
						gt = self.data_dict_test[name]['gt_name']
						if self.headed:
							coords = self.data_dict_test[name]['joints']
						gt_name = os.path.join(self.test_dir, gt)
						gt = imread(gt_name)
						name = os.path.join(self.test_dir, name)


					img = self.open_img(name)
					#img = scm.imresize(img, (256,256),'nearest')

					if self.resolutions:
						gt = self._give_resolutions(gt, img)

					elif self.headed:
						n_joints = 25
						#coords= self._joints(gt, n_joints) #### uncomment this line if you want to have inline joiint search
						
						hm = self._generate_hm(320,320, coords,320, np.ones(n_joints))
						
						
						r_angle = np.random.randint(-30, 30)
						img = 	transform.rotate(img, r_angle, preserve_range = True)
						gt = transform.rotate(gt, r_angle, preserve_range = True).astype(np.int64)
						hm = transform.rotate(hm, r_angle, preserve_range = True)

						gt = np.expand_dims(gt, axis = 0)
						gt_2 = np.repeat(gt, stacks, axis = 0)
						hm = np.expand_dims(hm, axis = 0)
						hm_2 = np.repeat(hm, head_stacks, axis = 0)
						
						
					else:
						#gt = scm.imresize(gt,(64,64),'nearest')
						img, gt = self._augment(img, gt)
						gt = np.expand_dims(gt, axis = 0)
						gt = np.repeat(gt, stacks, axis = 0)

					if normalize:
						train_img[i] = img.astype(np.float32) / 255
					else :
						train_img[i] = img.astype(np.float32)
					if self.headed:
						train_gtmap_t[i] = gt_2.astype(np.int32)
						train_gtmap_h[i] = hm_2.astype(np.float32)
					else:
						train_gtmap[i] = gt.astype(np.int32)
					i = i + 1
#				except :
#					print('error file: ', name)
			if self.headed:
				if sample_set == "test":
					yield train_img, train_gtmap_t, train_gtmap_h, train_weights, name
				else:
					yield train_img, train_gtmap_t, train_gtmap_h, train_weights
			else:
				if sample_set == "test":
					yield train_img, train_gtmap, train_weights, name
				else:
					yield train_img, train_gtmap, train_weights



	def _joints(self,image, n_joints):
		coords = np.zeros((n_joints,2))
		for i in range(1, n_joints+1):
			n_im = np.where(image==i,1,0)
			x,y = self._find_joints_box(n_im)
			coords[i-1] = x,y
		return coords

	def _find_joints_box(self,im,perc_w =0.1, perc_h=0.1):
		shapy = im.shape
		left = []
		right = []
		top = []
		bottom = []

		for i in range(shapy[0]):
			for j in range(shapy[1]):
				if im[i,j] != 0:
					left.append(j)
					break
			for j in range(shapy[1]-1,-1,-1):
				if im[i,j] != 0:
					right.append(j)
					break

		for j in range(shapy[1]):
			for i in range(shapy[0]):
				if im[i,j] != 0:
					top.append(i)
					break
			for i in range(shapy[0]-1,-1,-1):
				if im[i,j] != 0:
					bottom.append(i)
					break

		if len(bottom) != 0:
			f_left = np.min(left)
			f_right = np.max(right)
			f_top = np.min(top)
			f_bottom = np.max(bottom)

			width = np.abs(f_right -f_left)
			height = np.abs(f_bottom -f_top)

			x = int(np.floor(f_left + width/2))
			y = int(np.floor(f_top +height/2))

		else:
			x = -1
			y = -1
			width,height = 0,0

		return x,y

	def _makeGaussian(self, height, width, sigma = 3, center=None):
		""" Make a square gaussian kernel.
		size is the length of a side of the square
		sigma is full-width-half-maximum, which
		can be thought of as an effective radius.
		"""
		x = np.arange(0, width, 1, float)
		y = np.arange(0, height, 1, float)[:, np.newaxis]
		if center is None:
			x0 =  width // 2
			y0 = height // 2
		else:
			x0 = center[0]
			y0 = center[1]
		return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

	def _generate_hm(self, height, width ,joints, maxlenght, weight):
		""" Generate a full Heap Map for every joints in an array
		Args:
			height			: Wanted Height for the Heat Map
			width			: Wanted Width for the Heat Map
			joints			: Array of Joints
			maxlenght		: Lenght of the Bounding Box
		"""
		num_joints = joints.shape[0]
		hm = np.zeros((height, width, num_joints), dtype = np.float32)
		for i in range(num_joints):
			if not(np.array_equal(joints[i], [-1,-1])) and weight[i] == 1:
				s = int(np.sqrt(maxlenght) * maxlenght * 10 / 4096) + 2
				hm[:,:,i] = self._makeGaussian(height, width, sigma= s, center= (joints[i,0], joints[i,1]))
			else:
				hm[:,:,i] = np.zeros((height,width))
		return hm

	def generator(self, batchSize = 16, stacks = 4, norm = True, sample = 'train'):
		""" Create a Sample Generator
		Args:
			batchSize 	: Number of image per batch
			stacks 	 	: Stacks in HG model
			norm 	 	 	: (bool) True to normalize the batch
			sample 	 	: 'train'/'valid' Default: 'train'
		"""
		return self._aux_generator(batch_size=batchSize, stacks=stacks, normalize=norm, sample_set=sample)

	# ---------------------------- Image Reader --------------------------------
	def open_img(self, name, color = 'RGB'):
		""" Open an image
		Args:
			name	: Name of the sample
			color	: Color Mode (RGB/BGR/GRAY)
		"""
		img = cv2.imread(name)
		if color == 'RGB':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			return img
		elif color == 'BGR':
			return img
		elif color == 'GRAY':
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		else:
			print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

	def plot_img(self, name, plot = 'cv2'):
		""" Plot an image
		Args:
			name	: Name of the Sample
			plot	: Library to use (cv2: OpenCV, plt: matplotlib)
		"""
		if plot == 'cv2':
			img = self.open_img(name, color = 'BGR')
			cv2.imshow('Image', img)
		elif plot == 'plt':
			img = self.open_img(name, color = 'RGB')
			plt.imshow(img)
			plt.show()

	def test(self, toWait = 0.2):
		""" TESTING METHOD
		You can run it to see if the preprocessing is well done.
		Wait few seconds for loading, then diaporama appears with image and highlighted joints
		/!\ Use Esc to quit
		Args:
			toWait : In sec, time between pictures
		"""
		self._create_train_table()
		self._create_sets()
		for i in range(len(self.train_set)):
			img = self.open_img(self.train_set[i])
			w = self.data_dict[self.train_set[i]]['weights']
			padd, box = self._crop_data(img.shape[0], img.shape[1], self.data_dict[self.train_set[i]]['box'], self.data_dict[self.train_set[i]]['joints'], boxp= 0.0)
			new_j = self._relative_joints(box,padd, self.data_dict[self.train_set[i]]['joints'], to_size=256)
			rhm = self._generate_hm(256, 256, new_j,256, w)
			rimg = self._crop_img(img, padd, box)
			# See Error in self._generator
			#rimg = cv2.resize(rimg, (256,256))
			rimg = scm.imresize(rimg, (256,256))
			#rhm = np.zeros((256,256,16))
			#for i in range(16):
			#	rhm[:,:,i] = cv2.resize(rHM[:,:,i], (256,256))
			grimg = cv2.cvtColor(rimg, cv2.COLOR_RGB2GRAY)
			cv2.imshow('image', grimg / 255 + np.sum(rhm,axis = 2))
			# Wait
			time.sleep(toWait)
			if cv2.waitKey(1) == 27:
				print('Ended')
				cv2.destroyAllWindows()
				break



	# ------------------------------- PCK METHODS-------------------------------
	def pck_ready(self, idlh = 3, idrs = 12, testSet = None):
		""" Creates a list with all PCK ready samples
		(PCK: Percentage of Correct Keypoints)
		"""
		id_lhip = idlh
		id_rsho = idrs
		self.total_joints = 0
		self.pck_samples = []
		for s in self.data_dict.keys():
			if testSet == None:
				if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][id_rsho] == 1:
					self.pck_samples.append(s)
					wIntel = np.unique(self.data_dict[s]['weights'], return_counts = True)
					self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
			else:
				if self.data_dict[s]['weights'][id_lhip] == 1 and self.data_dict[s]['weights'][id_rsho] == 1 and s in testSet:
					self.pck_samples.append(s)
					wIntel = np.unique(self.data_dict[s]['weights'], return_counts = True)
					self.total_joints += dict(zip(wIntel[0], wIntel[1]))[1]
		print('PCK PREPROCESS DONE: \n --Samples:', len(self.pck_samples), '\n --Num.Joints', self.total_joints)

	def getSample(self, sample = None):
		""" Returns information of a sample
		Args:
			sample : (str) Name of the sample
		Returns:
			img: RGB Image
			new_j: Resized Joints
			w: Weights of Joints
			joint_full: Raw Joints
			max_l: Maximum Size of Input Image
		"""
		if sample != None:
			try:
				joints = self.data_dict[sample]['joints']
				box = self.data_dict[sample]['box']
				w = self.data_dict[sample]['weights']
				img = self.open_img(sample)
				padd, cbox = self._crop_data(img.shape[0], img.shape[1], box, joints, boxp = 0.2)
				new_j = self._relative_joints(cbox,padd, joints, to_size=256)
				joint_full = np.copy(joints)
				max_l = max(cbox[2], cbox[3])
				joint_full = joint_full + [padd[1][0], padd[0][0]]
				joint_full = joint_full - [cbox[0] - max_l //2,cbox[1] - max_l //2]
				img = self._crop_img(img, padd, cbox)
				img = img.astype(np.uint8)
				img = scm.imresize(img, (256,256))
				return img, new_j, w, joint_full, max_l
			except:
				return False
		else:
			print('Specify a sample name')


