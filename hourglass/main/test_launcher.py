#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 16:15:07 2018

@author: blue
"""

import configparser
from hourglass_tiny import HourglassModel
from datagen import DataGenerator
from os import environ

def process_config(conf_file):
	"""
	"""
	params = {}
	config = configparser.ConfigParser()
	config.read(conf_file)
	for section in config.sections():
		if section == 'DataSetHG':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Network':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Train':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Validation':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
		if section == 'Saver':
			for option in config.options(section):
				params[option] = eval(config.get(section, option))
	return params


if __name__ == '__main__':

	environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

	environ["CUDA_VISIBLE_DEVICES"]="7"

	print('--Parsing Config File')
	params = process_config('config.cfg')

	print('--Creating Dataset')
	dataset = DataGenerator(params['img_directory'], params['training_txt_file'], params['num_joints'], params['val_directory'], params['val_txt_file'], params['test_directory'], params['test_txt_file'],params['resolutions'], params['headed'], head_train = params['head_train'], head_test = params['head_test'], head_val = params['head_val'])
	dataset._create_test_table() #creates the lists with dicts of the coord. of boxes, joints and the corresp. weights

	model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'], attention = params['mcam'],training=False, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, name=params['name'], logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], tiny= params['tiny'], w_loss=params['weighted_loss'] ,modif=False, save_photos = params['save_photos'], number_save=params['number_save'],where_save = params['where_save'], headed = params['headed'], resolutions = params['resolutions'], head_stacks = params['head_stacks'])
	model.generate_test_model()
	model.test_init(dataset = dataset, load="./logs/test/model_hour_150")
