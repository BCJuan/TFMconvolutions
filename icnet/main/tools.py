import scipy.io as sio
import numpy as np
from PIL import Image
import tensorflow as tf

#MODIFIED 
#color range to label colours 2


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
                # 18 = bicycle

matfn = './utils/color150.mat'
def read_labelcolours(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    shape = color_table.shape
    color_list = [tuple(color_table[i]) for i in range(shape[0])]

    return color_list

def decode_labels(mask, img_shape, num_classes):
    if num_classes == 150:
        color_table = read_labelcolours(matfn)
    else:
        color_table = label_colours

    color_mat = tf.constant(color_table, dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, img_shape[0], img_shape[1], 3))
    
    return pred

def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
            
    return input_batch
