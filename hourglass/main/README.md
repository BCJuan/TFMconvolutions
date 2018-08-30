# Adaptation of Stacked Hourglass Model to Semantic Segmentation

Hourglass code from [wbenbihi](wbenbihi/hourglasstensorlfow) adapted to semantic segmentation task.

For general doubts about the code refer to the general repository where the documentation can be found.

Also the possibility to make two kind of experiments has been enabled: different resolution ground truths and auxiliary task head (joint detection). 

## Use changes

### Config file

New files in the configuration file:

	training_txt_file: file where the path of training images and ground truths is specified 
	head_train: path of file with joint info
	head_test: path of file with joint info
	head_val: path of file with joint info
	val_txt_file: file where the path of validation images and ground truths is specified
	test_txt_file: file where the path of testing images and ground truths is specified
	img_directory: training image directory
	val_directory: validation image directory
	test_directory: test iamge directory
	
	headed: (BOOL) to enable auxiliary head or not
	resolutions: (BOOL) to enable different resolution ground truths or not
	head_stacks: number of hourglass modules for the auxiliary head	
	
	save_photos: wether to save some inference instances when testing or not
	number_save: how many inference photos want to save
	where_save: directory where you will save the photos
	
### Datasets

In the case of the semantic segmentation the format of the files regarding the path of image and ground (`training_txt_file.txt`, for example) truth should be like:

	path_image.jpg path_ground_truth.png
	path_iamge_1.jpg path_ground_truth_1.jpg
	...
	
and the one regarding the joint information (`head_train.txt`) should be :

	path_image.jpg x1 y1 x2 y2 x3 y3 ...
	path_image_1.jpg x1 y1 x2 y2 ...
	
where `x1 y1` are the coordinates of the joint number 1 (as in the original repo)

### Training/testing

The training is the same as in the original repo:

* Configure the `config.cfg`
* Run `train_launcher.py`

But now the possibility of testing the model has been added: just run `test_launcher.py`

## Experiments

The possibility to make two kind experiments or modifications have been added apart from the semantic segmentation:

* Different resolution ground truths
* Joint detection head modules

	



 


