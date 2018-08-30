# -*- coding: utf-8 -*-
"""
Deep Human Pose Estimation

Project by Walid Benbihi
MSc Individual Project
Imperial College
Created on Mon Jul 10 19:13:56 2017

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
import time
import tensorflow as tf
import numpy as np
import sys
import datetime
import os
from scipy import misc

class HourglassModel():
	""" HourglassModel class: (to be renamed)
	Generate TensorFlow model to train and predict Human Pose from images (soon videos)
	Please check README.txt for further information on model management.
	"""
	def __init__(self, nFeat = 512, nStack = 4, nModules = 1, nLow = 4, outputDim = 16, batch_size = 16, drop_rate = 0.2, lear_rate = 2.5e-4, decay = 0.96, decay_step = 2000, dataset = None, training = True, w_summary = True, logdir_train = None, logdir_test = None,tiny = True, attention = False,modif = True,w_loss = False, name = 'tiny_hourglass', save_photos=False,where_save=None, number_save= None, headed = False, resolutions = False, head_stacks = None):
		""" Initializer
		Args:
			nStack				: number of stacks (stage/Hourglass modules)
			nFeat				: number of feature channels on conv layers
			nLow				: number of downsampling (pooling) per module
			outputDim			: number of output Dimension (16 for MPII)
			batch_size			: size of training/testing Batch
			dro_rate			: Rate of neurons disabling for Dropout Layers
			lear_rate			: Learning Rate starting value
			decay				: Learning Rate Exponential Decay (decay in ]0,1], 1 for constant learning rate)
			decay_step			: Step to apply decay
			dataset			: Dataset (class DataGenerator)
			training			: (bool) True for training / False for prediction
			w_summary			: (bool) True/False for summary of weight (to visualize in Tensorboard)
			tiny				: (bool) Activate Tiny Hourglass
			attention			: (bool) Activate Multi Context Attention Mechanism (MCAM)
			modif				: (bool) Boolean to test some network modification # DO NOT USE IT ! USED TO TEST THE NETWORK
			name				: name of the model
		"""
		self.nStack = nStack
		self.nFeat = nFeat
		self.nModules = nModules
		self.outDim = outputDim
		self.batchSize = batch_size
		self.training = training
		self.w_summary = w_summary
		self.tiny = tiny
		self.dropout_rate = drop_rate
		self.learning_rate = lear_rate
		self.decay = decay
		self.name = name
		self.attention = attention
		self.decay_step = decay_step
		self.nLow = nLow
		self.modif = modif
		self.dataset = dataset
		self.cpu = '/cpu:0'
		self.gpu = '/gpu:0'
		self.logdir_train = logdir_train
		self.logdir_test = logdir_test
		self.w_loss = w_loss
		self.number_save = number_save
		self.where_save = where_save
		self.save_photos = save_photos
		self.headed = headed
		self.resolutions = resolutions
		self.head_stacks = head_stacks
		
		if self.resolutions:
			self.outDim = [2,6,12,25]

		if self.save_photos:
			self.photos_list = ['ung_126_09_c0008_85.jpg',
				'ung_137_31_c0008_77.jpg',
				'19_12_c0006_29.jpg',
				'ung_144_34_c0008_39.jpg']

		if not self.where_save == None:
			if not os.path.exists(self.where_save):
				os.mkdir(self.where_save)

	# ACCESSOR

	def get_input(self):
		""" Returns Input (Placeholder) Tensor
		Image Input :
			Shape: (None,256,256,3)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
		return self.img
	def get_output(self):
		""" Returns Output Tensor
		Output Tensor :
			Shape: (None, nbStacks, 64, 64, outputDim)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
		return self.output
	def get_label(self):
		""" Returns Label (Placeholder) Tensor
		Image Input :
			Shape: (None, nbStacks, 64, 64, outputDim)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
		return self.gtMaps
	def get_loss(self):
		""" Returns Loss Tensor
		Image Input :
			Shape: (1,)
			Type : tf.float32
		Warning:
			Be sure to build the model first
		"""
		return self.loss
	def get_saver(self):
		""" Returns Saver
		/!\ USE ONLY IF YOU KNOW WHAT YOU ARE DOING
		Warning:
			Be sure to build the model first
		"""
		return self.saver

	def generate_test_model(self):
		""" Create the complete graph
		"""
		startTime = time.time()
		print('CREATE MODEL:')
		with tf.device(self.gpu):
			with tf.name_scope('inputs'):
				# Shape Input Image - batchSize: None, height: 256, width: 256, channel: 3 (RGB)
				self.img = tf.placeholder(dtype= tf.float32, shape= (None, 320, 320, 3), name = 'input_img')
				if self.w_loss:
					self.weights = tf.placeholder(dtype = tf.float32, shape = (None, self.outDim))
				# Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
				if self.headed:
					self.gtMaps = tf.placeholder(dtype = tf.int32, shape = (None, self.nStack , 320, 320))
					self.gtMaps_h = tf.placeholder(dtype = tf.float32, shape = (None, self.head_stacks, 320, 320, 24))
				else:
					self.gtMaps = tf.placeholder(dtype = tf.int32, shape = (None, self.nStack, 320, 320))
				# TODO : Implement weighted loss function
				# NOT USABLE AT THE MOMENT
				#weights = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 1, 1, self.outDim))
			inputTime = time.time()
			print('---Inputs : Done (' + str(int(abs(inputTime-startTime))) + ' sec.)')
			if self.attention:
				self.output = self._graph_mcam(self.img)
			else :
				self.output = self._graph_hourglass(self.img)
			graphTime = time.time()
			print('---Graph : Done (' + str(int(abs(graphTime-inputTime))) + ' sec.)')
#			with tf.name_scope('loss'):
#				if self.w_loss:
#					self.loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_loss')
#				else:
#					with tf.device(self.cpu):
#						self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps), name= 'cross_entropy_loss')
#			lossTime = time.time()
#			print('---Loss : Done (' + str(int(abs(graphTime-lossTime))) + ' sec.)')
		with tf.device(self.cpu):
			if self.save_photos:
					if len(os.listdir(self.where_save))<self.number_save:
						if self.resolutions:
							outy = self.output[-1]
						elif self.headed:
							outy = self.output[0][:,self.nStack-1,:,:,:]
						else:
							outy = self.output[:,self.nStack-1,:,:,:]
						outy = tf.argmax(outy, axis=3)
						self.pred = self.image_creation(outy)
			with tf.name_scope('accuracy'):
				self._miou_computation()
			accurTime = time.time()
			print('---Acc : Done (' + str(int(abs(accurTime-graphTime))) + ' sec.)')
#			with tf.name_scope('steps'):
#				self.train_step = tf.Variable(0, name = 'global_step', trainable= False)
#			with tf.name_scope('lr'):
#				self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay, staircase= True, name= 'learning_rate')
#			lrTime = time.time()
#			print('---LR : Done (' + str(int(abs(accurTime-lrTime))) + ' sec.)')
		with tf.device(self.gpu):
#			with tf.name_scope('rmsprop'):
#				self.rmsprop = tf.train.RMSPropOptimizer(learning_rate= self.lr)
#			optimTime = time.time()
#			print('---Optim : Done (' + str(int(abs(optimTime-lrTime))) + ' sec.)')
			with tf.name_scope('minimizer'):
				self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				self.running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")
#				with tf.control_dependencies(self.update_ops):
#					self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
			minimTime = time.time()
			print('---Minimizer : Done (' + str(int(abs(accurTime-minimTime))) + ' sec.)')
		self.init = tf.global_variables_initializer()
		self.running_vars_initializer = tf.variables_initializer(var_list=self.running_vars)
		initTime = time.time()
		print('---Init : Done (' + str(int(abs(initTime-minimTime))) + ' sec.)')
#		with tf.device(self.cpu):
#			with tf.name_scope('training'):
#				tf.summary.scalar('loss', self.loss, collections = ['train'])
#				tf.summary.scalar('learning_rate', self.lr, collections = ['train'])
#			with tf.name_scope('summary'):
#				tf.summary.scalar("All", self.joint_accur, collections = ['train', 'test'])
#		self.train_op = tf.summary.merge_all('train')
#		self.test_op = tf.summary.merge_all('test')
#		self.weight_op = tf.summary.merge_all('weight')
		endTime = time.time()
		print('Model created (' + str(int(abs(endTime-startTime))) + ' sec.)')
		del endTime, startTime, initTime, minimTime,  accurTime, graphTime, inputTime






	def generate_model(self):
		""" Create the complete graph
		"""
		startTime = time.time()
		print('CREATE MODEL:')
		with tf.device(self.gpu):
			with tf.name_scope('inputs'):
				# Shape Input Image - batchSize: None, height: 256, width: 256, channel: 3 (RGB)
				self.img = tf.placeholder(dtype= tf.float32, shape= (None, 320, 320, 3), name = 'input_img')
				if self.w_loss:
					self.weights = tf.placeholder(dtype = tf.float32, shape = (None, self.outDim))
				# Shape Ground Truth Map: batchSize x nStack x 64 x 64 x outDim
				if self.headed:
					self.gtMaps = tf.placeholder(dtype = tf.int32, shape = (None, self.nStack , 320, 320))
					self.gtMaps_h = tf.placeholder(dtype = tf.float32, shape = (None, self.head_stacks, 320, 320, 24))
				else:
					self.gtMaps = tf.placeholder(dtype = tf.int32, shape = (None, self.nStack, 320, 320))
				# TODO : Implement weighted loss function
				# NOT USABLE AT THE MOMENT
				#weights = tf.placeholder(dtype = tf.float32, shape = (None, self.nStack, 1, 1, self.outDim))
			inputTime = time.time()
			print('---Inputs : Done (' + str(int(abs(inputTime-startTime))) + ' sec.)')
			if self.attention:
				self.output = self._graph_mcam(self.img)
			else :
				self.output = self._graph_hourglass(self.img)
			graphTime = time.time()
			print('---Graph : Done (' + str(int(abs(graphTime-inputTime))) + ' sec.)')
			with tf.name_scope('loss'):
				if self.w_loss:
					self.loss = tf.reduce_mean(self.weighted_bce_loss(), name='reduced_loss')
				else:
					with tf.device(self.cpu):
						if not self.headed:
							if self.resolutions:
								lossy = []
								for ll in range(self.nStack):
									lossy.append(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output[ll], labels= self.gtMaps[:, ll, :, :])))
								self.loss = tf.add_n(lossy, name= 'cross_entropy_loss')
							else:
								self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps), name= 'cross_entropy_loss')
						else:

							out = tf.cast(self.output[1],tf.float32)
							loss_1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output[0], labels= tf.cast(self.gtMaps, tf.int32)), name= 'cross_entropy_loss_1')
							loss_2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=out, labels= tf.cast(self.gtMaps_h, tf.float32)), name= 'cross_entropy_loss_1')
							self.loss = loss_1 +loss_2
			lossTime = time.time()
			print('---Loss : Done (' + str(int(abs(graphTime-lossTime))) + ' sec.)')
		with tf.device(self.cpu):
			with tf.name_scope('accuracy'):
				self._miou_computation()
			accurTime = time.time()
			print('---Acc : Done (' + str(int(abs(accurTime-lossTime))) + ' sec.)')
			with tf.name_scope('steps'):
				self.train_step = tf.Variable(0, name = 'global_step', trainable= False)
			with tf.name_scope('lr'):
				self.lr = tf.train.exponential_decay(self.learning_rate, self.train_step, self.decay_step, self.decay, staircase= True, name= 'learning_rate')
			lrTime = time.time()
			print('---LR : Done (' + str(int(abs(accurTime-lrTime))) + ' sec.)')
		with tf.device(self.gpu):
			with tf.name_scope('rmsprop'):
				self.rmsprop = tf.train.RMSPropOptimizer(learning_rate= self.lr)
			optimTime = time.time()
			print('---Optim : Done (' + str(int(abs(optimTime-lrTime))) + ' sec.)')
			with tf.name_scope('minimizer'):
				self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
				self.running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="accuracy")
				with tf.control_dependencies(self.update_ops):
					self.train_rmsprop = self.rmsprop.minimize(self.loss, self.train_step)
			minimTime = time.time()
			print('---Minimizer : Done (' + str(int(abs(optimTime-minimTime))) + ' sec.)')
		self.init = tf.global_variables_initializer()
		self.running_vars_initializer = tf.variables_initializer(var_list=self.running_vars)
		initTime = time.time()
		print('---Init : Done (' + str(int(abs(initTime-minimTime))) + ' sec.)')
		with tf.device(self.cpu):
			with tf.name_scope('training'):
				tf.summary.scalar('loss', self.loss, collections = ['train'])
				tf.summary.scalar('learning_rate', self.lr, collections = ['train'])
			with tf.name_scope('summary'):
				tf.summary.scalar("All", self.joint_accur, collections = ['train', 'test'])
		self.train_op = tf.summary.merge_all('train')
		self.test_op = tf.summary.merge_all('test')
		self.weight_op = tf.summary.merge_all('weight')
		endTime = time.time()
		print('Model created (' + str(int(abs(endTime-startTime))) + ' sec.)')
		del endTime, startTime, initTime, optimTime, minimTime, lrTime, accurTime, lossTime, graphTime, inputTime


	def restore(self, load = None):
		""" Restore a pretrained model
		Args:
			load	: Model to load (None if training from scratch) (see README for further information)
		"""
		with tf.name_scope('Session'):
			with tf.device(self.gpu):
				self._init_session()
				self._define_saver_summary(summary = False)
				if load is not None:
					print('Loading Trained Model')
					t = time.time()
					self.saver.restore(self.Session, load)
					print('Model Loaded (', time.time() - t,' sec.)')
				else:
					print('Please give a Model in args (see README for further information)')

	def _train(self, nEpochs = 10, epochSize = 1000, saveStep = 500, validIter = 10):
		"""
		"""
		with tf.name_scope('Train'):

			###for file
			if os.path.exists("./logs/train/loss.npy"):
				print("loss file reloaded")
				ll = np.load("./logs/train/loss.npy")
				listy = list(ll)
			else:
				listy = []
			#####
			best_loss = np.infty
			self.generator = self.dataset._aux_generator(self.batchSize, self.nStack, normalize = True, sample_set = 'train')
			self.valid_gen = self.dataset._aux_generator(self.batchSize, self.nStack, normalize = True, sample_set = 'valid')
			startTime = time.time()
			self.resume = {}
			self.resume['accur'] = []
			self.resume['loss'] = []
			self.resume['err'] = []
			
			total_parameters = 0
			for variable in tf.trainable_variables():

				shape = variable.get_shape()

				variable_parameters = 1
				for dim in shape:
					variable_parameters *= dim.value
					total_parameters += variable_parameters
			print("Total parameters", total_parameters)
			
			for epoch in range(nEpochs):
				epochstartTime = time.time()
				avg_cost = 0.
				cost = 0.
				print('Epoch :' + str(epoch) + '/' + str(nEpochs) + '\n')
				# Training Set
				for i in range(epochSize):
					# DISPLAY PROGRESS BAR
					# TODO : Customize Progress Bar
					percent = ((i+1)/epochSize) * 100
					num = np.int(20*percent/100)
					tToEpoch = int((time.time() - epochstartTime) * (100 - percent)/(percent))
					sys.stdout.write('\r Train: {0}>'.format("="*num) + "{0}>".format(" "*(20-num)) + '||' + str(percent)[:4] + '%' + ' -cost: ' + str(cost)[:6] + ' -avg_loss: ' + str(avg_cost)[:5] + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
					sys.stdout.flush()
					if self.headed:
						img_train, gt_train, gt_train_h, weight_train = next(self.generator)
					else:
						img_train, gt_train, weight_train = next(self.generator)
					if i % saveStep == 0:
						if self.w_loss:
							_, c, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
						else:
							if self.headed:
								_, c, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.gtMaps_h:gt_train_h})
								_ = self.Session.run([self.update_acc,self.update_prec,self.update_rec,self.update_opsy], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.gtMaps_h:gt_train_h})
							else:
								_, c, summary = self.Session.run([self.train_rmsprop, self.loss, self.train_op], feed_dict = {self.img : img_train, self.gtMaps: gt_train})
								_ = self.Session.run([self.update_acc,self.update_prec,self.update_rec,self.update_opsy], feed_dict = {self.img : img_train, self.gtMaps: gt_train})
						# Save summary (Loss + Accuracy)
						#self.train_summary.add_summary(summary, epoch*epochSize + i)
						#self.train_summary.flush()
					else:
						if self.w_loss:
							_, c, = self.Session.run([self.train_rmsprop, self.loss], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
						else:
							if self.headed:
								_, c, = self.Session.run([self.train_rmsprop, self.loss], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.gtMaps: gt_train, self.gtMaps_h:gt_train_h})
								_ = self.Session.run([self.update_acc,self.update_prec,self.update_rec,self.update_opsy], feed_dict = {self.img : img_train, self.gtMaps: gt_train, self.gtMaps: gt_train, self.gtMaps_h:gt_train_h})
							else:
								_, c, = self.Session.run([self.train_rmsprop, self.loss], feed_dict = {self.img : img_train, self.gtMaps: gt_train})
								_ = self.Session.run([self.update_acc,self.update_prec,self.update_rec,self.update_opsy], feed_dict = {self.img : img_train, self.gtMaps: gt_train})
					cost += c
					avg_cost += c/epochSize

				#### for file
				resum = np.array([avg_cost])
				#####

				epochfinishTime = time.time()
				#Save Weight (axis = epoch)
#				if self.w_loss:
#					weight_summary = self.Session.run(self.weight_op, {self.img : img_train, self.gtMaps: gt_train, self.weights: weight_train})
#				else :
#					weight_summary = self.Session.run(self.weight_op, {self.img : img_train, self.gtMaps: gt_train})
				#self.train_summary.add_summary(weight_summary, epoch)
				#self.train_summary.flush()
				#self.weight_summary.add_summary(weight_summary, epoch)
				#self.weight_summary.flush()
				print('Epoch ' + str(epoch) + '/' + str(nEpochs) + ' done in ' + str(int(epochfinishTime-epochstartTime)) + ' sec.' + ' -avg_time/batch: ' + str(((epochfinishTime-epochstartTime)/epochSize))[:4] + ' sec.')
				with tf.name_scope('save'):
					if avg_cost< best_loss:
						self.saver.save(self.Session, os.path.join("./logs/test/",str(self.name + '_' + str(epoch + 1))))
						best_loss = avg_cost
				self.resume['loss'].append(cost)
				# Validation Setgt_valid_h
				accuracy_arr = np.zeros((28))
				for i in range(validIter):
					if self.headed:
						img_valid, gt_valid,gt_valid_h, w_valid = next(self.valid_gen)
					else:
						img_valid, gt_valid, w_valid = next(self.valid_gen)
						
					miou, accura, preci, reci, acc_class = self.Session.run([self.joint_accur,self.accura, self.prec, self.rec, self.acc_per_class], feed_dict = {self.img : img_valid, self.gtMaps: gt_valid})
					f1 = 2*preci*reci/(preci+reci)
					arr = np.array([miou, accura, f1])
					arr = np.append(arr, acc_class)
					accuracy_arr += arr / validIter
				print('--Avg. Accuracy =', str(accuracy_arr*100), '%' )
				resum = np.append(resum, accuracy_arr)
				listy.append(resum)
				np.save("./logs/train/loss.npy", np.array(listy))
				self.resume['accur'].append(miou)
				self.resume['err'].append(np.sum(acc_class) / len(acc_class))
#				valid_summary = self.Session.run(self.test_op, feed_dict={self.img : img_valid, self.gtMaps: gt_valid})
				#self.test_summary.add_summary(valid_summary, epoch)
				#self.test_summary.flush()
			print('Training Done')
			print('Resume:' + '\n' + '  Epochs: ' + str(nEpochs) + '\n' + '  n. Images: ' + str(nEpochs * epochSize * self.batchSize) )
			print('  Final Loss: ' + str(cost) + '\n' + '  Relative Loss: ' + str(100*self.resume['loss'][-1]/(self.resume['loss'][0] + 0.1)) + '%' )
			print('  Relative Improvement: ' + str((self.resume['err'][-1] - self.resume['err'][0]) * 100) +'%')
			print('  Training Time: ' + str( datetime.timedelta(seconds=time.time() - startTime)))


	def _test(self, steps=15000):
		"""
		"""
		with tf.name_scope('Test'):

			###for file
			#####
			epochstartTime = time.time()
			self.generator = self.dataset._aux_generator(batch_size = 1, stacks = self.nStack, normalize = True, sample_set = 'test')
			accuracy_arr = np.zeros((28))
			for step in range(steps):


				percent = ((step+1)/steps) * 100
				num = np.int(20*percent/100)
				tToEpoch = int((time.time() - epochstartTime) * (100 - percent)/(percent))
				sys.stdout.write('\r Test: {0}>'.format("="*num) + "{0}>".format(" "*(20-num)) + '||' + str(percent)[:4] + '%' + ' -timeToEnd: ' + str(tToEpoch) + ' sec.')
				sys.stdout.flush()
				
				if self.headed:
					img_test, gt_test, gt_test_h, weight_test, name = next(self.generator)
				else:
					img_test, gt_test, weight_test, name = next(self.generator)

				if self.headed:
					_ = self.Session.run([self.update_acc,self.update_prec,self.update_rec,self.update_opsy], feed_dict = {self.img : img_test, self.gtMaps: gt_test, self.gtMaps_h:gt_test_h})
				else:
					_ = self.Session.run([self.update_acc,self.update_prec,self.update_rec,self.update_opsy], feed_dict = {self.img : img_test, self.gtMaps: gt_test})
					
				miou, accura, preci, reci, acc_class = self.Session.run([self.joint_accur,self.accura, self.prec, self.rec, self.acc_per_class], feed_dict = {self.img : img_test, self.gtMaps: gt_test})

				f1 = 2*preci*reci/(preci+reci)
				arr = np.array([miou, accura, f1])
				arr = np.append(arr, acc_class)
				arr = np.nan_to_num(arr)
				accuracy_arr += arr / steps

				if self.save_photos:
					if np.random.rand(1)<0.02:
						if len(os.listdir(self.where_save))<self.number_save:
							out = self.Session.run(self.pred, feed_dict = {self.img : img_test, self.gtMaps: gt_test})
							misc.imsave(os.path.join(self.where_save, name.split("/")[-1]), out[0])
					if name.split("/")[-1] in self.photos_list:
						out = self.Session.run(self.pred, feed_dict = {self.img : img_test, self.gtMaps: gt_test})
						misc.imsave(os.path.join(self.where_save, name.split("/")[-1]), out[0])

			np.save("./logs/testloss.npy", accuracy_arr)

	def image_creation(self, mask):
		label_colours = [[0, 0, 0], [153, 76, 0], [153, 153, 0]
                # 0 = road, 1 = sidewalk, 2 = building
                ,[76, 153, 0], [0, 153, 0], [0, 153, 76]
                # 3 = wall, 4 = fence, 5 = pole
                ,[0,153, 153], [0, 76, 153], [0, 0, 153]
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,[76, 0, 153], [153, 0, 153], [153, 0, 76]
                # 9 = terrain, 10 = sky, 11 = person
                ,[255, 51, 51], [255, 153, 51], [255, 255, 51]
                # 12 = rider, 13 = car, 14 = truck
                ,[153, 255, 51], [51, 255, 51], [51, 255, 153]
                # 15 = bus, 16 = train, 17 = motocycle
                ,[51, 255, 255], [51,153,255],[51,51,255],[153,51,255],[255,52,255],[255,52,153], [192,192,192]]

		color_table = label_colours
		if self.resolutions:
			dim = self.outDim[-1]
		else:
			dim = self.outDim
			
		color_mat = tf.constant(color_table, dtype=tf.float32)
		onehot_output = tf.one_hot(mask, depth=dim)
		onehot_output = tf.reshape(onehot_output, (-1, dim))
		pred = tf.matmul(onehot_output, color_mat)
		print("Third shape",mask.shape)
		pred = tf.reshape(pred, (1, 320, 320, 3))
		return pred




	def record_training(self, record):
		""" Record Training Data and Export them in CSV file
		Args:
			record		: record dictionnary
		"""
		out_file = open(self.name + '_train_record.csv', 'w')
		for line in range(len(record['accur'])):
			out_string = ''
			labels = [record['loss'][line]] + [record['err'][line]] + record['accur'][line]
			for label in labels:
				out_string += str(label) + ', '
			out_string += '\n'
			out_file.write(out_string)
		out_file.close()
		print('Training Record Saved')

	def training_init(self, nEpochs = 10, epochSize = 1000, saveStep = 500, dataset = None, load = None):
		""" Initialize the training
		Args:
			nEpochs		: Number of Epochs to train
			epochSize		: Size of one Epoch
			saveStep		: Step to save 'train' summary (has to be lower than epochSize)
			dataset		: Data Generator (see generator.py)
			load			: Model to load (None if training from scratch) (see README for further information)
		"""
		with tf.name_scope('Session'):
			with tf.device(self.gpu):
				self._init_weight()
				self._define_saver_summary(summary=False)
				if load is not None:
					self.saver.restore(self.Session, load)
					#try:
						#	self.saver.restore(self.Session, load)
					#except Exception:
						#	print('Loading Failed! (Check README file for further information)')
				self._train(nEpochs, epochSize, saveStep, validIter=10)

	def test_init(self, dataset = None, load = None):
		""" Initialize the training
		Args:
			nEpochs		: Number of Epochs to train
			epochSize		: Size of one Epoch
			saveStep		: Step to save 'train' summary (has to be lower than epochSize)
			dataset		: Data Generator (see generator.py)
			load			: Model to load (None if training from scratch) (see README for further information)
		"""
		with tf.name_scope('Session'):
			with tf.device(self.gpu):
				self._init_weight()
				self._define_saver_summary(summary=False)
				if load is not None:
					self.saver.restore(self.Session, load)
					#try:
						#	self.saver.restore(self.Session, load)
					#except Exception:
						#	print('Loading Failed! (Check README file for further information)')
				self._test(len(dataset.test_set))

	def weighted_bce_loss(self):
		""" Create Weighted Loss Function
		WORK IN PROGRESS
		"""
		self.bceloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output, labels= self.gtMaps), name= 'cross_entropy_loss')
		e1 = tf.expand_dims(self.weights,axis = 1, name = 'expdim01')
		e2 = tf.expand_dims(e1,axis = 1, name = 'expdim02')
		e3 = tf.expand_dims(e2,axis = 1, name = 'expdim03')
		return tf.multiply(e3,self.bceloss, name = 'lossW')

	def _miou_computation(self):

		if self.resolutions:
			out = tf.argmax(self.output[-1],axis=3)
			dim = self.outDim[-1]
		elif self.headed:
			print(self.output)
			out = tf.argmax(self.output[0],axis=4)
			print(out)
			out = out[:, self.nStack - 1, :, :]
			print(out)
			print(self.gtMaps)
			dim = self.outDim
		else:
			self.output = tf.argmax(self.output, axis=4)
			out = self.output[:, self.nStack - 1, :, :]
			dim = self.outDim
		self.joint_accur, self.update_opsy = tf.metrics.mean_iou(self.gtMaps[:, self.nStack - 1, :, :],out, dim)
		self.joint_accur = tf.cast(self.joint_accur, tf.float32)
		self.accura, self.update_acc = tf.metrics.accuracy(self.gtMaps[:, self.nStack - 1, :, :],out)
		self.prec, self.update_prec = tf.metrics.precision(self.gtMaps[:, self.nStack - 1, :, :],out)
		self.rec, self.update_rec = tf.metrics.recall(self.gtMaps[:, self.nStack - 1, :, :], out)
		maps = tf.reshape(self.gtMaps[:, self.nStack - 1, :, :], [-1])
		outs = tf.reshape(out, [-1])
		self.conf = tf.confusion_matrix(maps,outs, dim)
		self.acc_per_class = tf.diag_part(self.conf)/tf.reduce_sum(self.conf,1)
		self.acc_per_class = tf.where(tf.is_nan(self.acc_per_class), tf.zeros_like(self.acc_per_class), self.acc_per_class)

	def _define_saver_summary(self, summary = False):
		""" Create Summary and Saver
		Args:
			logdir_train		: Path to train summary directory
			logdir_test		: Path to test summary directory
		"""
		if (self.logdir_train == None) or (self.logdir_test == None):
			raise ValueError('Train/Test directory not assigned')
		else:
			with tf.device(self.cpu):
				self.saver = tf.train.Saver(max_to_keep=1)
			if summary:
				with tf.device(self.gpu):

					self.train_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())
					self.test_summary = tf.summary.FileWriter(self.logdir_test)
					self.weight_summary = tf.summary.FileWriter(self.logdir_train, tf.get_default_graph())

	def _init_weight(self):
		""" Initialize weights
		"""
		print('Session initialization')
		self.Session = tf.Session()
		t_start = time.time()
		self.Session.run(self.init)
		self.Session.run(self.running_vars_initializer)

		print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

	def _init_session(self):
		""" Initialize Session
		"""
		print('Session initialization')
		t_start = time.time()
		self.Session = tf.Session()
		print('Sess initialized in ' + str(int(time.time() - t_start)) + ' sec.')

	def _graph_hourglass(self, inputs):
		"""Create the Network
		Args:
			inputs : TF Tensor (placeholder) of shape (None, 256, 256, 3) #TODO : Create a parameter for customize size
		"""
		with tf.name_scope('model'):
			with tf.name_scope('preprocessing'):
				# Input Dim : nbImages x 256 x 256 x 3
#				pad1 = tf.pad(inputs, [[0,0],[3,3],[3,3],[0,0]], name='pad_1')
#				# Dim pad1 : nbImages x 260 x 260 x 3
				conv1 = self._conv_bn_relu(inputs, filters= 64, kernel_size = 1, strides = 1, name = 'conv_256_to_128')
#				# Dim conv1 : nbImages x 128 x 128 x 64
				r1 = self._residual(conv1, numOut = self.nFeat, name = 'r1')
#				# Dim pad1 : nbImages x 128 x 128 x 128
#				pool1 = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], padding='VALID')
				# Dim pool1 : nbImages x 64 x 64 x 128
				if self.tiny:
					r3 = self._residual(r1, numOut=self.nFeat, name='r3')
				else:
					r2 = self._residual(r1, numOut= int(self.nFeat/2), name = 'r2')
					r3 = self._residual(r2, numOut= self.nFeat, name = 'r3')
			# Storage Table
			hg = [None] * self.nStack
			ll = [None] * self.nStack
			ll_ = [None] * self.nStack
			drop = [None] * self.nStack
			out = [None] * self.nStack
			out_ = [None] * self.nStack
			sum_ = [None] * self.nStack
			########################## for headed version
			hgh = [None] * self.head_stacks
			llh = [None] * self.head_stacks
			ll_h = [None] * self.head_stacks
			droph = [None] * self.head_stacks
			outh = [None] * self.head_stacks
			out_h = [None] * self.head_stacks
			sum_h = [None] * self.head_stacks
			##################################
			
			if self.tiny:
				with tf.name_scope('stacks'):
					with tf.name_scope('stage_0'):
						hg[0] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
						drop[0] = tf.layers.dropout(hg[0], rate = self.dropout_rate, training = self.training, name = 'dropout')
						ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1, 1, name = 'll')
						if self.modif:
							# TEST OF BATCH RELU
							out[0] = self._conv_bn_relu(ll[0], self.outDim, 1, 1, 'VALID', 'out')
						else:
							if self.resolutions:
								out[0] = self._conv(ll[0], self.outDim[0], 1, 1, 'VALID', 'out')
							else:
								out[0] = self._conv(ll[0], self.outDim, 1, 1, 'VALID', 'out')
						out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
						sum_[0] = tf.add_n([out_[0], ll[0], r3], name = 'merge')
					for i in range(1, self.nStack - 1):
						with tf.name_scope('stage_' + str(i)):
							hg[i] = self._hourglass(sum_[i-1], self.nLow, self.nFeat, 'hourglass')
							drop[i] = tf.layers.dropout(hg[i], rate = self.dropout_rate, training = self.training, name = 'dropout')
							ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, name= 'll')
							if self.modif:
								# TEST OF BATCH RELU
								out[i] = self._conv_bn_relu(ll[i], self.outDim, 1, 1, 'VALID', 'out')
							else:
								if self.resolutions:
									out[i] = self._conv(ll[i], self.outDim[i], 1, 1, 'VALID', 'out')
								else:
									out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')
							out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
							sum_[i] = tf.add_n([out_[i], ll[i], sum_[i-1]], name= 'merge')
					with tf.name_scope('stage_' + str(self.nStack - 1)):
						hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
						drop[self.nStack-1] = tf.layers.dropout(hg[self.nStack-1], rate = self.dropout_rate, training = self.training, name = 'dropout')
						ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack-1], self.nFeat,1,1, 'VALID', 'conv')
						if self.modif:
							out[self.nStack - 1] = self._conv_bn_relu(ll[self.nStack - 1], self.outDim, 1,1, 'VALID', 'out')
						else:
							if self.resolutions:
								out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim[self.nStack - 1], 1,1, 'VALID', 'out')
							else:
								out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim, 1,1, 'VALID', 'out')
							
	########################################### pose head
				if self.headed: 
					with tf.name_scope('head_stacks'):
						with tf.name_scope('stage_0'):
							hgh[0] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
							droph[0] = tf.layers.dropout(hgh[0], rate = self.dropout_rate, training = self.training, name = 'dropout')
							llh[0] = self._conv_bn_relu(droph[0], self.nFeat, 1, 1, name = 'll')

							outh[0] = self._conv(llh[0], self.outDim-1, 1, 1, 'VALID', 'out')
							out_h[0] = self._conv(outh[0], self.nFeat, 1, 1, 'VALID', 'out_')
							sum_h[0] = tf.add_n([out_h[0], llh[0], sum_[self.nStack - 2]], name = 'merge')
						for i in range(1, self.head_stacks - 1):
							with tf.name_scope('stage_' + str(i)):
								hgh[i] = self._hourglass(sum_h[i-1], self.nLow, self.nFeat, 'hourglass')
								droph[i] = tf.layers.dropout(hgh[i], rate = self.dropout_rate, training = self.training, name = 'dropout')
								llh[i] = self._conv_bn_relu(droph[i], self.nFeat, 1, 1, name= 'll')

								outh[i] = self._conv(llh[i], self.outDim-1, 1, 1, 'VALID', 'out')
								out_h[i] = self._conv(outh[i], self.nFeat, 1, 1, 'VALID', 'out_')
								sum_h[i] = tf.add_n([out_h[i], llh[i], sum_h[i-1]], name= 'merge')
						with tf.name_scope('stage_' + str(self.head_stacks - 1)):
							hgh[self.head_stacks - 1] = self._hourglass(sum_h[self.head_stacks - 2], self.nLow, self.nFeat, 'hourglass')
							droph[self.head_stacks-1] = tf.layers.dropout(hgh[self.head_stacks-1], rate = self.dropout_rate, training = self.training, name = 'dropout')
							llh[self.head_stacks - 1] = self._conv_bn_relu(droph[self.head_stacks-1], self.nFeat,1,1, 'VALID', 'conv')
							outh[self.head_stacks - 1] = self._conv(llh[self.head_stacks - 1], self.outDim-1, 1,1, 'VALID', 'out')
	########################################################################################
				if self.modif:
					return tf.nn.sigmoid(tf.stack(out, axis= 1 , name= 'stack_output'),name = 'final_output')
				else:
					if self.resolutions:
						returns = []
						for i in range(self.nStack):
							returns.append(out[i])
						return returns
					elif self.headed:
						return [tf.stack(out, axis= 1 , name = 'final_output'),tf.stack(outh, axis= 1 , name = 'final_output')]
					else:
						return tf.stack(out, axis= 1 , name = 'final_output')
			else:
				with tf.name_scope('stacks'):
					with tf.name_scope('stage_0'):
						hg[0] = self._hourglass(r3, self.nLow, self.nFeat, 'hourglass')
						drop[0] = tf.layers.dropout(hg[0], rate = self.dropout_rate, training = self.training, name = 'dropout')
						ll[0] = self._conv_bn_relu(drop[0], self.nFeat, 1,1, 'VALID', name = 'conv')
						ll_[0] =  self._conv(ll[0], self.nFeat, 1, 1, 'VALID', 'll')
						if self.modif:
							# TEST OF BATCH RELU
							out[0] = self._conv_bn_relu(ll[0], self.outDim, 1, 1, 'VALID', 'out')
						else:
							if self.resolutions:
								out[0] = self._conv(ll[0], self.outDim[0], 1, 1, 'VALID', 'out')
							else:
								out[0] = self._conv(ll[0], self.outDim, 1, 1, 'VALID', 'out')
						out_[0] = self._conv(out[0], self.nFeat, 1, 1, 'VALID', 'out_')
						sum_[0] = tf.add_n([out_[0], r3, ll_[0]], name='merge')
					for i in range(1, self.nStack -1):
						with tf.name_scope('stage_' + str(i)):
							hg[i] = self._hourglass(sum_[i-1], self.nLow, self.nFeat, 'hourglass')
							drop[i] = tf.layers.dropout(hg[i], rate = self.dropout_rate, training = self.training, name = 'dropout')
							ll[i] = self._conv_bn_relu(drop[i], self.nFeat, 1, 1, 'VALID', name= 'conv')
							ll_[i] = self._conv(ll[i], self.nFeat, 1, 1, 'VALID', 'll')
							if self.modif:
								out[i] = self._conv_bn_relu(ll[i], self.outDim, 1, 1, 'VALID', 'out')
							else:
								if self.resolutions:
									out[i] = self._conv(ll[i], self.outDim[i], 1, 1, 'VALID', 'out')
								else:
									out[i] = self._conv(ll[i], self.outDim, 1, 1, 'VALID', 'out')
							out_[i] = self._conv(out[i], self.nFeat, 1, 1, 'VALID', 'out_')
							sum_[i] = tf.add_n([out_[i], sum_[i-1], ll_[0]], name= 'merge')
					with tf.name_scope('stage_' + str(self.nStack -1)):
						hg[self.nStack - 1] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
						drop[self.nStack-1] = tf.layers.dropout(hg[self.nStack-1], rate = self.dropout_rate, training = self.training, name = 'dropout')
						ll[self.nStack - 1] = self._conv_bn_relu(drop[self.nStack-1], self.nFeat, 1, 1, 'VALID', 'conv')
						if self.modif:
							out[self.nStack - 1] = self._conv_bn_relu(ll[self.nStack - 1], self.outDim, 1,1, 'VALID', 'out')
						else:
							if self.resolutions:
								out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim[self.nStack - 1], 1,1, 'VALID', 'out')
							else:
								out[self.nStack - 1] = self._conv(ll[self.nStack - 1], self.outDim, 1,1, 'VALID', 'out')
								
								
			########################################### pose head
				if self.headed: 
					with tf.name_scope('head_stacks'):
						with tf.name_scope('stage_0'):
							hgh[0] = self._hourglass(sum_[self.nStack - 2], self.nLow, self.nFeat, 'hourglass')
							droph[0] = tf.layers.dropout(hgh[0], rate = self.dropout_rate, training = self.training, name = 'dropout')
							llh[0] = self._conv_bn_relu(droph[0], self.nFeat, 1, 1, name = 'll')

							outh[0] = self._conv(llh[0], self.outDim-1, 1, 1, 'VALID', 'out')
							out_h[0] = self._conv(outh[0], self.nFeat, 1, 1, 'VALID', 'out_')
							sum_h[0] = tf.add_n([out_h[0], llh[0], sum_[self.nStack - 2]], name = 'merge')
						for i in range(1, self.head_stacks - 1):
							with tf.name_scope('stage_' + str(i)):
								hgh[i] = self._hourglass(sum_h[i-1], self.nLow, self.nFeat, 'hourglass')
								droph[i] = tf.layers.dropout(hgh[i], rate = self.dropout_rate, training = self.training, name = 'dropout')
								llh[i] = self._conv_bn_relu(droph[i], self.nFeat, 1, 1, name= 'll')

								outh[i] = self._conv(llh[i], self.outDim-1, 1, 1, 'VALID', 'out')
								out_h[i] = self._conv(outh[i], self.nFeat, 1, 1, 'VALID', 'out_')
								sum_h[i] = tf.add_n([out_h[i], llh[i], sum_h[i-1]], name= 'merge')
						with tf.name_scope('stage_' + str(self.head_stacks - 1)):
							hgh[self.head_stacks - 1] = self._hourglass(sum_h[self.head_stacks - 2], self.nLow, self.nFeat, 'hourglass')
							droph[self.head_stacks-1] = tf.layers.dropout(hgh[self.head_stacks-1], rate = self.dropout_rate, training = self.training, name = 'dropout')
							llh[self.head_stacks - 1] = self._conv_bn_relu(droph[self.head_stacks-1], self.nFeat,1,1, 'VALID', 'conv')
							outh[self.head_stacks - 1] = self._conv(llh[self.head_stacks - 1], self.outDim-1, 1,1, 'VALID', 'out')
	########################################################################################
				if self.modif:
					return tf.nn.sigmoid(tf.stack(out, axis= 1 , name= 'stack_output'),name = 'final_output')
				else:
					if self.resolutions:
						returns = []
						for i in range(self.nStack):
							returns.append(out[i])
						return returns
					elif self.headed:
						return [tf.stack(out, axis= 1 , name = 'final_output'),tf.stack(outh, axis= 1 , name = 'final_output')]
					else:
						return tf.stack(out, axis= 1 , name = 'final_output')


	def _conv(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv'):
		""" Spatial Convolution (CONV2D)
		Args:
			inputs			: Input Tensor (Data Type : NHWC)
			filters		: Number of filters (channels)
			kernel_size	: Size of kernel
			strides		: Stride
			pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
			name			: Name of the block
		Returns:
			conv			: Output Tensor (Convolved Input)
		"""
		with tf.name_scope(name):
			# Kernel for convolution, Xavier Initialisation
			kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
			conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding=pad, data_format='NHWC')
			if self.w_summary:
				with tf.device('/cpu:0'):
					tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
			return conv

	def _conv_bn_relu(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'VALID', name = 'conv_bn_relu'):
		""" Spatial Convolution (CONV2D) + BatchNormalization + ReLU Activation
		Args:
			inputs			: Input Tensor (Data Type : NHWC)
			filters		: Number of filters (channels)
			kernel_size	: Size of kernel
			strides		: Stride
			pad				: Padding Type (VALID/SAME) # DO NOT USE 'SAME' NETWORK BUILT FOR VALID
			name			: Name of the block
		Returns:
			norm			: Output Tensor
		"""
		with tf.name_scope(name):
			kernel = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
			conv = tf.nn.conv2d(inputs, kernel, [1,strides,strides,1], padding='VALID', data_format='NHWC')
			norm = tf.contrib.layers.batch_norm(conv, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
			if self.w_summary:
				with tf.device('/cpu:0'):
					tf.summary.histogram('weights_summary', kernel, collections = ['weight'])
			return norm

	def _conv_block(self, inputs, numOut, name = 'conv_block'):
		""" Convolutional Block
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the block
		Returns:
			conv_3	: Output Tensor
		"""
		if self.tiny:
			with tf.name_scope(name):
				norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
				pad = tf.pad(norm, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
				conv = self._conv(pad, int(numOut), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
				return conv
		else:
			with tf.name_scope(name):
				with tf.name_scope('norm_1'):
					norm_1 = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
					conv_1 = self._conv(norm_1, int(numOut/2), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
				with tf.name_scope('norm_2'):
					norm_2 = tf.contrib.layers.batch_norm(conv_1, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
					pad = tf.pad(norm_2, np.array([[0,0],[1,1],[1,1],[0,0]]), name= 'pad')
					conv_2 = self._conv(pad, int(numOut/2), kernel_size=3, strides=1, pad = 'VALID', name= 'conv')
				with tf.name_scope('norm_3'):
					norm_3 = tf.contrib.layers.batch_norm(conv_2, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
					conv_3 = self._conv(norm_3, int(numOut), kernel_size=1, strides=1, pad = 'VALID', name= 'conv')
				return conv_3

	def _skip_layer(self, inputs, numOut, name = 'skip_layer'):
		""" Skip Layer
		Args:
			inputs	: Input Tensor
			numOut	: Desired output number of channel
			name	: Name of the bloc
		Returns:
			Tensor of shape (None, inputs.height, inputs.width, numOut)
		"""
		with tf.name_scope(name):
			if inputs.get_shape().as_list()[3] == numOut:
				return inputs
			else:
				conv = self._conv(inputs, numOut, kernel_size=1, strides = 1, name = 'conv')
				return conv

	def _residual(self, inputs, numOut, name = 'residual_block'):
		""" Residual Unit
		Args:
			inputs	: Input Tensor
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
		with tf.name_scope(name):
			convb = self._conv_block(inputs, numOut)
			skipl = self._skip_layer(inputs, numOut)
			if self.modif:
				return tf.nn.relu(tf.add_n([convb, skipl], name = 'res_block'))
			else:
				return tf.add_n([convb, skipl], name = 'res_block')

	def _hourglass(self, inputs, n, numOut, name = 'hourglass'):
		""" Hourglass Module
		Args:
			inputs	: Input Tensor
			n		: Number of downsampling step
			numOut	: Number of Output Features (channels)
			name	: Name of the block
		"""
		with tf.name_scope(name):
			# Upper Branch
			up_1 = self._residual(inputs, numOut, name = 'up_1')
			# Lower Branch
			low_ = tf.contrib.layers.max_pool2d(inputs, [2,2], [2,2], padding='VALID')
			low_1= self._residual(low_, numOut, name = 'low_1')

			if n > 0:
				low_2 = self._hourglass(low_1, n-1, numOut, name = 'low_2')
			else:
				low_2 = self._residual(low_1, numOut, name = 'low_2')

			low_3 = self._residual(low_2, numOut, name = 'low_3')
			up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name = 'upsampling')
			if self.modif:
				# Use of RELU
				return tf.nn.relu(tf.add_n([up_2,up_1]), name='out_hg')
			else:
				return tf.add_n([up_2,up_1], name='out_hg')

	def _argmax(self, tensor):
		""" ArgMax
		Args:
			tensor	: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			arg		: Tuple of max position
		"""
		resh = tf.reshape(tensor, [-1])
		argmax = tf.arg_max(resh, 0)
		return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

	def _compute_err(self, u, v):
		""" Given 2 tensors compute the euclidean distance (L2) between maxima locations
		Args:
			u		: 2D - Tensor (Height x Width : 64x64 )
			v		: 2D - Tensor (Height x Width : 64x64 )
		Returns:
			(float) : Distance (in [0,1])
		"""
		u_x,u_y = self._argmax(u)
		v_x,v_y = self._argmax(v)
		return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))), tf.to_float(91))

	def _accur(self, pred, gtMap, num_image):
		""" Given a Prediction batch (pred) and a Ground Truth batch (gtMaps),
		returns one minus the mean distance.
		Args:
			pred		: Prediction Batch (shape = num_image x 64 x 64)
			gtMaps		: Ground Truth Batch (shape = num_image x 64 x 64)
			num_image 	: (int) Number of images in batch
		Returns:
			(float)
		"""
		err = tf.to_float(0)
		for i in range(num_image):
			err = tf.add(err, self._compute_err(pred[i], gtMap[i]))
		return tf.subtract(tf.to_float(1), err/num_image)

	# MULTI CONTEXT ATTENTION MECHANISM
	# WORK IN PROGRESS DO NOT USE THESE METHODS
	# BASED ON:
	# Multi-Context Attention for Human Pose Estimation
	# Authors: Xiao Chu, Wei Yang, Wanli Ouyang, Cheng Ma, Alan L. Yuille, Xiaogang Wang
	# Paper: https://arxiv.org/abs/1702.07432
	# GitHub Torch7 Code: https://github.com/bearpaw/pose-attention

	def _bn_relu(self, inputs):
		norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, activation_fn = tf.nn.relu, is_training = self.training)
		return norm

	def _pool_layer(self, inputs, numOut, name = 'pool_layer'):
		with tf.name_scope(name):
			bnr_1 = self._bn_relu(inputs)
			pool = tf.contrib.layers.max_pool2d(bnr_1,[2,2],[2,2],padding='VALID')
			pad_1 = tf.pad(pool, np.array([[0,0],[1,1],[1,1],[0,0]]))
			conv_1 = self._conv(pad_1, numOut, kernel_size=3, strides=1, name='conv')
			bnr_2 = self._bn_relu(conv_1)
			pad_2 = tf.pad(bnr_2, np.array([[0,0],[1,1],[1,1],[0,0]]))
			conv_2 = self._conv(pad_2, numOut, kernel_size=3, strides=1, name='conv')
			upsample = tf.image.resize_nearest_neighbor(conv_2, tf.shape(conv_2)[1:3]*2, name = 'upsampling')
		return upsample

	def _attention_iter(self, inputs, lrnSize, itersize, name = 'attention_iter'):
		with tf.name_scope(name):
			numIn = inputs.get_shape().as_list()[3]
			padding = np.floor(lrnSize/2)
			pad = tf.pad(inputs, np.array([[0,0],[1,1],[1,1],[0,0]]))
			U = self._conv(pad, filters=1, kernel_size=3, strides=1)
			pad_2 = tf.pad(U, np.array([[0,0],[padding,padding],[padding,padding],[0,0]]))
			sharedK = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([lrnSize,lrnSize, 1, 1]), name= 'shared_weights')
			Q = []
			C = []
			for i in range(itersize):
				if i ==0:
					conv = tf.nn.conv2d(pad_2, sharedK, [1,1,1,1], padding='VALID', data_format='NHWC')
				else:
					conv = tf.nn.conv2d(Q[i-1], sharedK, [1,1,1,1], padding='SAME', data_format='NHWC')
				C.append(conv)
				Q_tmp = tf.nn.sigmoid(tf.add_n([C[i], U]))
				Q.append(Q_tmp)
			stacks = []
			for i in range(numIn):
				stacks.append(Q[-1])
			pfeat = tf.multiply(inputs,tf.concat(stacks, axis = 3) )
		return pfeat

	def _attention_part_crf(self, inputs, lrnSize, itersize, usepart, name = 'attention_part'):
		with tf.name_scope(name):
			if usepart == 0:
				return self._attention_iter(inputs, lrnSize, itersize)
			else:
				partnum = self.outDim
				pre = []
				for i in range(partnum):
					att = self._attention_iter(inputs, lrnSize, itersize)
					pad = tf.pad(att, np.array([[0,0],[0,0],[0,0],[0,0]]))
					s = self._conv(pad, filters=1, kernel_size=1, strides=1)
					pre.append(s)
				return tf.concat(pre, axis = 3)

	def _residual_pool(self, inputs, numOut, name = 'residual_pool'):
		with tf.name_scope(name):
			return tf.add_n([self._conv_block(inputs, numOut), self._skip_layer(inputs, numOut), self._pool_layer(inputs, numOut)])

	def _rep_residual(self, inputs, numOut, nRep, name = 'rep_residual'):
		with tf.name_scope(name):
			out = [None]*nRep
			for i in range(nRep):
				if i == 0:
					tmpout = self._residual(inputs,numOut)
				else:
					tmpout = self._residual_pool(out[i-1],numOut)
				out[i] = tmpout
			return out[nRep-1]

	def _hg_mcam(self, inputs, n, numOut, imSize, nModual, name = 'mcam_hg'):
		with tf.name_scope(name):
			#------------Upper Branch
			pool = tf.contrib.layers.max_pool2d(inputs,[2,2],[2,2],padding='VALID')
			up = []
			low = []
			for i in range(nModual):
				if i == 0:
					if n>1:
						tmpup = self._rep_residual(inputs, numOut, n -1)
					else:
						tmpup = self._residual(inputs, numOut)
					tmplow = self._residual(pool, numOut)
				else:
					if n>1:
						tmpup = self._rep_residual(up[i-1], numOut, n-1)
					else:
						tmpup = self._residual_pool(up[i-1], numOut)
					tmplow = self._residual(low[i-1], numOut)
				up.append(tmpup)
				low.append(tmplow)
				#up[i] = tmpup
				#low[i] = tmplow
			#----------------Lower Branch
			if n>1:
				low2 = self._hg_mcam(low[-1], n-1, numOut, int(imSize/2), nModual)
			else:
				low2 = self._residual(low[-1], numOut)
			low3 = self._residual(low2, numOut)
			up_2 = tf.image.resize_nearest_neighbor(low3, tf.shape(low3)[1:3]*2, name = 'upsampling')
			return tf.add_n([up[-1], up_2], name = 'out_hg')

	def _lin(self, inputs, numOut, name = 'lin'):
		l = self._conv(inputs, filters = numOut, kernel_size = 1, strides = 1)
		return self._bn_relu(l)

	def _graph_mcam(self, inputs):
		with tf.name_scope('preprocessing'):
			pad1 = tf.pad(inputs, np.array([[0,0],[3,3],[3,3],[0,0]]))
			cnv1_ = self._conv(pad1, filters = 64, kernel_size = 7, strides = 1)
			cnv1 = self._bn_relu(cnv1_)
			r1 = self._residual(cnv1, 64)
			pool1 = tf.contrib.layers.max_pool2d(r1,[2,2],[2,2],padding='VALID')
			r2 = self._residual(pool1, 64)
			r3 = self._residual(r2, 128)
			pool2 = tf.contrib.layers.max_pool2d(r3,[2,2],[2,2],padding='VALID')
			r4 = self._residual(pool2,128)
			r5 = self._residual(r4, 128)
			r6 = self._residual(r5, 256)
		out = []
		inter = []
		inter.append(r6)
		if self.nLow == 3:
			nModual = int(16/self.nStack)
		else:
			nModual = int(8/self.nStack)
		with tf.name_scope('stacks'):
			for i in range(self.nStack):
				with tf.name_scope('houglass_' + str(i+1)):
					hg = self._hg_mcam(inter[i], self.nLow, self.nFeat, 64, nModual)

				if i == self.nStack - 1:
					ll1 = self._lin(hg, self.nFeat*2)
					ll2 = self._lin(ll1, self.nFeat*2)
					drop = tf.layers.dropout(ll2, rate=0.1, training = self.training)
					att =  self._attention_part_crf(drop, 1, 3, 0)
					tmpOut = self._attention_part_crf(att, 1, 3, 1)
				else:
					ll1 = self._lin(hg, self.nFeat)
					ll2 = self._lin(ll1, self.nFeat)
					drop = tf.layers.dropout(ll2, rate=0.1, training = self.training)
					if i > self.nStack // 2:
						att = self._attention_part_crf(drop, 1, 3, 0)
						tmpOut = self._attention_part_crf( att, 1, 3, 1)
					else:
						att = self._attention_part_crf(ll2, 1, 3, 0)
						tmpOut = self._conv(att, filters = self.outDim, kernel_size = 1, strides = 1)
				out.append(tmpOut)
				if i < self.nStack - 1:
					outmap = self._conv(tmpOut, filters = self.nFeat, kernel_size = 1, strides = 1)
					ll3 = self._lin(outmap, self.nFeat)
					tmointer = tf.add_n([inter[i], outmap, ll3])
					inter.append(tmointer)
		return tf.stack(out, axis= 1 , name = 'final_output')










