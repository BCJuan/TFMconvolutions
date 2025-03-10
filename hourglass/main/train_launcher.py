"""
TRAIN LAUNCHER

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
	dataset = DataGenerator(params['img_directory'], params['training_txt_file'], params['num_joints'], params['val_directory'], params['val_txt_file'], resolutions = params['resolutions'], headed = params['headed'], head_train = params['head_train'], head_test = params['head_test'], head_val = params['head_val'])
	dataset._create_train_table() #creates the lists with dicts of the coord. of boxes, joints and the corresp. weights
	dataset._randomize() # shuffles the previous lists
	dataset._create_sets() # validation and training lists

	model = HourglassModel(nFeat=params['nfeats'], nStack=params['nstacks'], nModules=params['nmodules'], nLow=params['nlow'], outputDim=params['num_joints'], batch_size=params['batch_size'], attention = params['mcam'],training=True, drop_rate= params['dropout_rate'], lear_rate=params['learning_rate'], decay=params['learning_rate_decay'], decay_step=params['decay_step'], dataset=dataset, name=params['name'], logdir_train=params['log_dir_train'], logdir_test=params['log_dir_test'], tiny= params['tiny'], w_loss=params['weighted_loss'] ,modif=False, headed = params['headed'], resolutions=params['resolutions'], head_stacks = params['head_stacks'])
	model.generate_model()
	model.training_init(nEpochs=params['nepochs'], epochSize=params['epoch_size'], saveStep=params['saver_step'], dataset = None)

