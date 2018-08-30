from __future__ import print_function
import argparse
import os
import time

import tensorflow as tf
import numpy as np
from tqdm import trange

from model import ICNet, ICNet_BN
from image_reader import read_labeled_image_list
from tools import prepare_label

IMG_MEAN = np.array(( 106.96, 120.77, 136.31), dtype=np.float32)

#I have modified the following
#have created parameters for SURREAL
	#in it important changes are directories: datadir which is /data and the list surreal_val_list.txt
#also I have changed the dataset selection in the argument parsing section adding the option surreal
# the selection of the parameters depending on the dataset at the beginning of the main section
#also have change the preprocess puttin the entry for surreal, but I do not know how to preprocess i
	# I think that the part of cityscapes does nothing since there adds no padding, it only expands the dimensions
#also have added accuracy and what i think is mean average precision where miou was calculated
	#changed also the final two runs adding the array of updates for each metric and the final print
#where the model is restored from; now is snapshots previusly was models
#add metrics

# define setting & model configuration
ADE20k_param = {'name': 'ade20k',
                'input_size': [480, 480],
                'num_classes': 150, # predict: [0~149] corresponding to label [1~150], ignore class 0 (background) 
                'ignore_label': 0,
                'num_steps': 2000,
                'data_dir': '../../ADEChallengeData2016/', 
                'data_list': './list/ade20k_val_list.txt'}
                
cityscapes_param = {'name': 'cityscapes',
                    'input_size': [1025, 2049],
                    'num_classes': 19,
                    'ignore_label': 255,
                    'num_steps': 500,
                    'data_dir': '/data/cityscapes_dataset/cityscape', 
                    'data_list': './list/cityscapes_val_list.txt'}

surreal_param = {'name': 'surreal',
		 'input_size': [320, 320],
                 'num_classes': 25,
                 'ignore_label': 100,
                 'num_steps': 16965,
                 'data_dir': './cluster_test/', 
                 'data_list': './list/test_cluster_list.txt'}

model_paths = {'train': './model/icnet_cityscapes_train_30k.npy', 
              'trainval': './model/icnet_cityscapes_trainval_90k.npy',
              'train_bn': './model/icnet_cityscapes_train_30k_bnnomerge.npy',
              'trainval_bn': './model/icnet_cityscapes_trainval_90k_bnnomerge.npy',
              'others': './snapshots/'}

# mapping different model
model_config = {'train': ICNet, 'trainval': ICNet, 'train_bn': ICNet_BN, 'trainval_bn': ICNet_BN, 'others': ICNet_BN}


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")

    parser.add_argument("--measure-time", action="store_true",
                        help="whether to measure inference time")
    parser.add_argument("--model", type=str, default='',
                        help="Model to use.",
                        choices=['train', 'trainval', 'train_bn', 'trainval_bn', 'others'],
                        required=True)
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes', 'surreal'],
                        required=True)
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])

    return parser.parse_args()

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

time_list = []
def calculate_time(sess, net, pred, feed_dict):
    start = time.time()
    sess.run(net.layers['data'], feed_dict=feed_dict)
    data_time = time.time() - start

    start = time.time()
    sess.run(pred, feed_dict=feed_dict)
    total_time = time.time() - start

    inference_time = total_time - data_time

    time_list.append(inference_time)
    print('average inference time: {}'.format(np.mean(time_list)))

def preprocess(img, param):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    shape = param['input_size']

#important!!! I do not know which one I have to choose
    if param['name'] == 'cityscapes' or param['name'] == 'surreal':
        img = tf.image.pad_to_bounding_box(img, 0, 0, shape[0], shape[1])
        img.set_shape([shape[0], shape[1], 3])
        img = tf.expand_dims(img, axis=0)
    elif param['name'] == 'ade20k':
        img = tf.expand_dims(img, axis=0)
        img = tf.image.resize_bilinear(img, shape, align_corners=True)
        
    return img

def get_mask(gt, num_classes, ignore_label):
    less_equal_class = tf.less_equal(gt, num_classes - 1)  ##here had num_classes -1
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)
    
    return indices

def create_loss(output, label, num_classes, ignore_label):
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1,])

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss

def main():
    args = get_arguments()
    
    if args.dataset == 'ade20k':
        param = ADE20k_param
    elif args.dataset == 'cityscapes':
        param = cityscapes_param
    else:
        param = surreal_param

    # Set placeholder
    image_filename = tf.placeholder(dtype=tf.string)
    anno_filename = tf.placeholder(dtype=tf.string)

    # Read & Decode image
    img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
    anno = tf.image.decode_image(tf.read_file(anno_filename), channels=1)
    img.set_shape([None, None, 3])
    anno.set_shape([None, None, 1])

    ori_shape = tf.shape(img)
    img = preprocess(img, param)

    model = model_config[args.model]
    net = model({'data': img}, num_classes=param['num_classes'], 
                    filter_scale=args.filter_scale, evaluation=True)

    # Predictions.
    raw_output = net.layers['conv6_cls']

    raw_output_up = tf.image.resize_bilinear(raw_output, size=ori_shape[:2], align_corners=True)
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    raw_pred = tf.expand_dims(raw_output_up, dim=3)

    # mIoU
    pred_flatten = tf.reshape(raw_pred, [-1,])
    raw_gt = tf.reshape(anno, [-1,])

    mask = tf.not_equal(raw_gt, param['ignore_label'])
    indices = tf.squeeze(tf.where(mask), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    
#I do not know which one I have to choose
    if args.dataset == 'ade20k':
        pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes']+1)
    elif args.dataset == 'cityscapes':
        mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=param['num_classes'])
    elif args.dataset == 'surreal':
        less_equal_class = tf.less_equal(raw_gt, param['num_classes']-1)
        not_equal_ignore = tf.not_equal(raw_gt, param['ignore_label'])
        mask = tf.logical_and(less_equal_class, not_equal_ignore)
        indices = tf.squeeze(tf.where(mask), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int64)
        pred = tf.cast(tf.gather(pred_flatten, indices),tf.int64)
        gt_n = tf.reshape(gt, [-1])
        pred_n = tf.reshape(pred, [-1])
        with tf.name_scope('metrics'):
            mIoU, update_op = tf.metrics.mean_iou(gt_n,pred_n, num_classes=param['num_classes'])
            accu, update_acc = tf.metrics.accuracy(gt_n,pred_n)
            reca, update_rec = tf.metrics.recall(gt_n,pred_n)
            prec, update_pre = tf.metrics.precision(gt_n,pred_n)
            mean, update_mean = tf.metrics.mean_per_class_accuracy(gt_n,pred_n, num_classes =param['num_classes'])
            conf_matrix = tf.confusion_matrix(gt_n,pred_n, num_classes=param['num_classes'])
            acc_per_class = tf.diag_part(conf_matrix)/tf.reduce_sum(conf_matrix,1)
            acc_per_class_good = tf.where(tf.is_nan(acc_per_class), tf.zeros_like(acc_per_class), acc_per_class)
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
        
    # Set up tf session and initialize variables.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
    
    sess.run(init)
    sess.run(running_vars_initializer)


    listy = []
#tiene pinta que lo que tenia que estar en snapshots en train ahora va a model
    model_path = model_paths[args.model]
    if args.model == 'others':
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            loader = tf.train.Saver(var_list=tf.global_variables())
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
    else:
        #net.load(model_path, sess)
        print('Restore from {}'.format(model_path))

    img_files, anno_files = read_labeled_image_list(param['data_dir'], param['data_list'])
    for i in trange(param['num_steps'], desc='evaluation', leave=True):
        feed_dict = {image_filename: img_files[i], anno_filename: anno_files[i]}
        _ = sess.run([update_op,update_acc,update_rec,update_pre, update_mean], feed_dict=feed_dict)
        m,a,r,p,ma,apc = sess.run([mIoU,accu,reca,prec, mean,acc_per_class_good], feed_dict=feed_dict)
        f = 2*p*r/(p+r)
        metris = np.array([m,a,f,r,p])
        metris = np.append(metris, ma)
        metris = np.append(metris, apc)
        listy.append(metris) 
        
        
        if i > 0 and args.measure_time:
            calculate_time(sess, net, raw_pred, feed_dict)
    
    ll = np.mean(np.array(listy), axis = 0)
    np.save("./loss_data/loss_metrics.npy",ll)
    print('MIOU: {}'.format(m))


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main()
