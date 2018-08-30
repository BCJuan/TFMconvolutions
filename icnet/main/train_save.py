"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet 
"""
from __future__ import print_function

import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np

from model import ICNet_BN
from tools import decode_labels, prepare_label
from image_reader import ImageReader

IMG_MEAN = np.array(( 106.96, 120.77, 136.31), dtype=np.float32)

#have changed: ignore_label to 0 fro backgroun
# inputsize
#data_dir
#data_list_path
#num_classes to 25 including background
#uncommented line 191 where it loads a pretrained model (.netload )
#NUM_STEPS to a 100 it was at 60001
#LAMBDAS to what is stated in github
#image mean
#metrics
#learning rate
#max to keep

# If you want to apply to other datasets, change following four lines
DATA_DIR = './cluster_train/'
DATA_LIST_PATH = './list/train_cluster_list.txt' 
IGNORE_LABEL = 100 # The class number of background
INPUT_SIZE = '320, 320' # Input size for training

BATCH_SIZE = 64
LEARNING_RATE = 1e-2
LEARNING_RATE_2 = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 25
NUM_STEPS = 3001
NUM_STEPS_2 = 3001
POWER = 0.9
RANDOM_SEED = 1234
WEIGHT_DECAY = 0.0001
PRETRAINED_MODEL = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'
SNAPSHOT_DIR = './snapshots/'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 100

# Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
LAMBDA1 = 0.4
LAMBDA2 = 0.6
LAMBDA3 = 1.0



def get_arguments():
    parser = argparse.ArgumentParser(description="ICNet")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--update-mean-var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true",
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

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

def for_metrics(output, label, num_classes, ignore_label):
    
    ori_shape = tf.shape(label)
    raw_output_up = tf.image.resize_bilinear(output, size=ori_shape[1:3], align_corners=True)
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    raw_pred = tf.expand_dims(raw_output_up, dim=3)
    
    pred_flatten = tf.reshape(raw_pred, [-1,])
    raw_gt = tf.reshape(label, [-1,])
    
    less_equal_class = tf.less_equal(raw_gt, num_classes - 1)  ##here had num_classes -1
    not_equal_ignore = tf.not_equal(raw_gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)
    
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int64)
    pred = tf.cast(tf.gather(pred_flatten, indices),tf.int64) 
    
    return gt, pred

def main():
    """Create the model and start the training."""
    
    args = get_arguments()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    coord = tf.train.Coordinator()
    
    if args.update_mean_var == False and args.train_beta_gamma == False:
        DATA_LIST_PATH = './list/train_cluster_list_1.txt' 
    else:
        DATA_LIST_PATH = './list/train_cluster_list_2.txt' 
        
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            DATA_DIR,
            DATA_LIST_PATH,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
    
    net = ICNet_BN({'data': image_batch}, is_training=True, num_classes=args.num_classes, filter_scale=args.filter_scale)
    
    sub4_out = net.layers['sub4_out']
    sub24_out = net.layers['sub24_out']
    sub124_out = net.layers['conv6_cls']

    if args.update_mean_var == False and args.train_beta_gamma == False:
        restore_var = [v for v in tf.global_variables() if 'conv6_cls' not in v.name]
    else:
        restore_var = tf.global_variables() #
        
    all_trainable = [v for v in tf.trainable_variables() if ('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma]
   
    loss_sub4 = create_loss(sub4_out, label_batch, args.num_classes, args.ignore_label)
    loss_sub24 = create_loss(sub24_out, label_batch, args.num_classes, args.ignore_label)
    loss_sub124 = create_loss(sub124_out, label_batch, args.num_classes, args.ignore_label)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    
    reduced_loss = LAMBDA1 * loss_sub4 +  LAMBDA2 * loss_sub24 + LAMBDA3 * loss_sub124 + tf.add_n(l2_losses)

    #metrics by me
    good_label, good_pred = for_metrics(sub124_out,label_batch,args.num_classes, args.ignore_label)
    good_label_re = tf.reshape(good_label,[args.batch_size,h*w])
    good_pred_re = tf.reshape(good_pred,[args.batch_size,h*w])
    
    with tf.name_scope('metrics'):
        mIoU, update_op_m = tf.metrics.mean_iou( good_label_re, good_pred_re, num_classes=args.num_classes)
        accu, update_acc = tf.metrics.accuracy(good_label_re, good_pred_re)
        reca, update_rec = tf.metrics.recall(good_label_re, good_pred_re)
        prec, update_pre = tf.metrics.precision(good_label_re, good_pred_re)
        mean, update_mean = tf.metrics.mean_per_class_accuracy(good_label_re, good_pred_re, num_classes = args.num_classes)
    
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")

    if args.update_mean_var == False and args.train_beta_gamma == False:
    # Using Poly learning rate policy 
        base_lr = tf.constant(args.learning_rate)
    else:
        base_lr = tf.constant(LEARNING_RATE_2)
        
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    
    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if args.update_mean_var == False:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
        grads = tf.gradients(reduced_loss, all_trainable)
        train_op = opt_conv.apply_gradients(zip(grads, all_trainable))
        
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
    
    sess.run(init)
    sess.run(running_vars_initializer)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)

    ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('Restore from pre-trained model...')
        net.load(args.restore_from, sess)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    if os.path.exists("./loss_data/loss_metrics.npy"):
        ll = np.load("./loss_data/loss_metrics.npy")
        listy = list(ll)
    else:
        listy = []
        
        
    if args.update_mean_var == False and args.train_beta_gamma == False:
        RANGE = NUM_STEPS
    else:
        RANGE = NUM_STEPS_2
    # Iterate over training steps.
    for step in range(RANGE):
        start_time = time.time()
        
        feed_dict = {step_ph: step}
        try:
            if step % args.save_pred_every == 0:
                loss_value, loss1, loss2, loss3, _ = sess.run([reduced_loss, loss_sub4, loss_sub24, loss_sub124, train_op], feed_dict=feed_dict)
                _ = sess.run([update_op_m, update_acc, update_pre, update_rec, update_mean], feed_dict=feed_dict)
                m,a,r,p,m_a = sess.run([mIoU,accu,reca,prec,mean], feed_dict=feed_dict)
                
                f = 2*p*r/(p+r)
                save(saver, sess, args.snapshot_dir, step)
                metris = np.array([step, loss_value,loss1, loss2, loss3, m,a,f])
                metris = np.append(metris, m_a)
                listy.append(metris) 
                np.save("./loss_data/loss_metrics.npy",np.array(listy))
                
            else:
                loss_value, loss1, loss2, loss3, _ = sess.run([reduced_loss, loss_sub4, loss_sub24, loss_sub124, train_op], feed_dict=feed_dict)

                _ = sess.run([update_op_m, update_acc, update_pre, update_rec, update_mean], feed_dict=feed_dict)
                m,a,r,p,m_a = sess.run([mIoU,accu,reca,prec,mean], feed_dict=feed_dict)
                f = 2*p*r/(p+r)
                
                metris = np.array([step, loss_value,loss1, loss2, loss3, m,a,f])
                metris = np.append(metris, m_a)
                listy.append(metris) 

                
        except tf.errors.OutOfRangeError:
            break
        duration = time.time() - start_time
            
        print('step {:d} \t total loss = {:.3f}, sub4 = {:.3f}, sub24 = {:.3f}, sub124 = {:.3f} ({:.3f} sec/step)'.format(step, loss_value, loss1, loss2, loss3, duration))
        print('MIOU: {} Acc: {} F1: {}'.format(m,a,f))
        
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6"
    main()