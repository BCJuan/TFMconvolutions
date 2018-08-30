# Fully Convolutional Networks for Human Body Part Segmentation

Performance evaluation of three networks, [ICNet](https://arxiv.org/abs/1704.08545), [SegNet](https://arxiv.org/abs/1505.07293)
and [Stacked Hourglass](https://arxiv.org/abs/1603.06937), in the [SURREAL](https://github.com/gulvarol/surreal) dataset.

The whole code, report and presentation are the result of the Master's Degree in Fundamentals of Data Science by University of Barcelona.

The repository consists in mainly two parts: folders for networks and notebooks for data processing.

## Notebooks

* [Data exploration](https://github.com/BCJuan/TFMconvolutions/blob/master/data_exploration.ipynb): Explores the SURREAL dataset and its contents.
* [From films to frames](https://github.com/BCJuan/TFMconvolutions/blob/master/from_films_to_frames.ipynb): transforms mp4 videos to .jpg images
* [Image mean](https://github.com/BCJuan/TFMconvolutions/blob/master/color_mean.npy): reports RGB mean of dataset and several transforms.
* [Crop and top](https://github.com/BCJuan/TFMconvolutions/blob/master/crop_n_top_dataset.ipynb): creates cropped images.
* [Make clusters](https://github.com/BCJuan/TFMconvolutions/blob/master/make_clusters.ipynb): makes clusters for final training, validation and test sets.
* [Resolutions](https://github.com/BCJuan/TFMconvolutions/blob/master/resize_n_diff_resolutions.ipynb): studies resiing functions and creates different ground truth resolutions.
* [Joint creation](https://github.com/BCJuan/TFMconvolutions/blob/master/joint_creation.ipynb) Recreates body joints information.
* [Weights](https://github.com/BCJuan/TFMconvolutions/blob/master/weight_analysis.ipynb): different weighting schemes for body parts.

## Network codes

All three code folders, [ICNet](https://github.com/hellochick/ICNet-tensorflow), [SegNet](https://github.com/tkuanlun350/Tensorflow-SegNet) and [Stacked Hourglass](https://github.com/wbenbihi/hourglasstensorlfow) are modifications of the respective original repos. 
There also can be found folders to recreate the results.

## Example of results
On the left ground truth and on the right prediction by ICNet.

![Ground Truth](https://github.com/BCJuan/TFMconvolutions/blob/master/19_12_c0006_segm_29_gt.png?raw=true)
![Prediction ICNet](https://github.com/BCJuan/TFMconvolutions/blob/master/19_12_c0006_29.jpg?raw=true)

## Disclaimer

The code used in this repo belongs to other repositories as indicated. Its use and modifications obey educational purposes only.
All the credit to the original creators. 
