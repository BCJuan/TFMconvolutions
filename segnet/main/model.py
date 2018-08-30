import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from sklearn.metrics import precision_score, recall_score

import os, sys
import numpy as np
import math
from datetime import datetime
import time
from PIL import Image
from math import ceil
from tensorflow.python.ops import gen_nn_ops
# modules
from Utils import _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage
from Inputs import *

#changed 
#number of classes
# image heigjht and width
#batch size
#line 291: the shape of labels were 360,480 changed them for 320,320


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0      # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

INITIAL_LEARNING_RATE = 0.001      # Initial learning rate.
EVAL_BATCH_SIZE = 8
BATCH_SIZE = 32
# for CamVid
IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320
IMAGE_DEPTH = 3

NUM_CLASSES = 25
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 92961
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 15233
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1
TEST_ITER = NUM_EXAMPLES_PER_EPOCH_FOR_TEST / BATCH_SIZE

def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl**2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def orthogonal_initializer(scale = 1.1):
    ''' From Lasagne and Keras. Reference: Saxe et al., http://arxiv.org/abs/1312.6120
    '''
    def _initializer(shape, dtype=tf.float32, partition_info=None):
      flat_shape = (shape[0], np.prod(shape[1:]))
      a = np.random.normal(0.0, 1.0, flat_shape)
      u, _, v = np.linalg.svd(a, full_matrices=False)
      # pick the one with the correct shape
      q = u if u.shape == flat_shape else v
      q = q.reshape(shape) #this needs to be corrected to float32
      return tf.constant(scale * q[:shape[0], :shape[1]], dtype=tf.float32)
    return _initializer

def loss_ww(logits, labels):
  """
      loss func without re-weighting
  """
  # Calculate the average cross entropy loss across the batch.
  logits = tf.reshape(logits, [-1,NUM_CLASSES])
  labels = tf.reshape(labels, [-1])

  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)

  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def loss_ww_manual(logits, labels):

    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, NUM_CLASSES))
        
        epsilon = tf.constant(value=1e-10)

        logits = logits + epsilon

        # consturct one-hot label array
        label_flat = tf.reshape(labels, (-1, 1))

        # should be [batch ,num_classes]
        labels = tf.reshape(tf.one_hot(label_flat, depth=NUM_CLASSES), (-1, NUM_CLASSES))

        softmax = tf.nn.softmax(logits)

        cross_entropy = -tf.reduce_sum(labels * tf.log(softmax + epsilon), axis=[1])
            
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
        
    return loss


def weighted_loss(logits, labels, num_classes, w_type, head=None):
    """ median-frequency re-weighting """
    with tf.name_scope('loss'):

        
        if w_type == "normal":
            logits = tf.reshape(logits, (-1, num_classes))
            print("logits",logits.shape , logits)
            
            epsilon = tf.constant(value=1e-10)
    
            logits = logits + epsilon
    
            # consturct one-hot label array
            label_flat = tf.reshape(labels, (-1, 1))
    
            # should be [batch ,num_classes]
            labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))
    
            softmax = tf.nn.softmax(logits)
    
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])

        else:
            logits = tf.reshape(logits, (-1, num_classes))
        
            epsilon = tf.constant(value=1e-10)
    
            logits = logits + epsilon
    
            # should be [batch ,num_classes]
            labels = tf.reshape(tf.one_hot(labels, depth=num_classes), (-1, num_classes))
    
            softmax = tf.nn.softmax(tf.multiply(logits,head))
    
            cross_entropy = -tf.reduce_sum(labels * tf.log(softmax + epsilon), axis=[1])

            
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return loss


def cal_loss(logits, labels,w_type):
    loss_weight = np.array([0.1265395 , 0.97901945, 0.98779824, 0.98798795, 0.99297994,
       0.99004227, 0.9902172 , 0.99399384, 0.99761395, 0.99752619,
       0.99085405, 0.99956869, 0.99918055, 0.99838453, 0.99770222,
       0.99789736, 0.99134871, 0.99592386, 0.99592018, 0.99679445,
       0.99659237, 0.99884241, 0.99871236, 0.99932548, 0.99923426]) # class 0~11
#    
#    loss_weight = tf.constant([0.00134934769594975,	
#           0.0548351842469636,	
#           0.0923963767207384,	
#           0.0952650592913482,	
#           0.156380870740497,	
#           0.113052936739173,	
#           0.116187279167639,	
#           0.181073044725212,	
#           0.405285806552916,	
#           0.400809143317001,	
#           0.122243562996140,	
#           1,	
#           0.941120945509562,	
#           0.560479631931825,	
#           0.415442280425244,	
#           0.464607162224048,	
#           0.130283282851676,	
#           0.255949048107327,	
#           0.263686918927640,	
#           0.316056007969892,	
#           0.309816548557251,	
#           0.684273567653671,	
#           0.651440097071119,	
#           0.904838636036523,	
#           0.865534010270693])

    labels = tf.cast(labels, tf.int32)
    # return loss(logits, labels)
    return weighted_loss(logits, labels, NUM_CLASSES,w_type, head=loss_weight)

def conv_layer_with_bn(inputT, shape, train_phase, activation=True, name=None):
    in_channel = shape[2]
    out_channel = shape[3]
    k_size = shape[0]
    with tf.variable_scope(name) as scope:
      kernel = _variable_with_weight_decay('ort_weights', shape=shape, initializer=orthogonal_initializer(), wd=None)
      conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [out_channel], tf.constant_initializer(0.0))
      bias = tf.nn.bias_add(conv, biases)
      if activation is True:
        conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
      else:
        conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def get_deconv_filter(f_shape):
  """
    reference: https://github.com/MarvinTeichmann/tensorflow-fcn
  """
  width = f_shape[0]
  heigh = f_shape[0]
  f = ceil(width/2.0)
  c = (2 * f - 1 - f % 2) / (2.0 * f)
  bilinear = np.zeros([f_shape[0], f_shape[1]])
  for x in range(width):
      for y in range(heigh):
          value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
          bilinear[x, y] = value
  weights = np.zeros(f_shape)
  for i in range(f_shape[2]):
      weights[:, :, i, i] = bilinear

  init = tf.constant_initializer(value=weights,
                                 dtype=tf.float32)
  return tf.get_variable(name="up_filter", initializer=init,
                         shape=weights.shape)

def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
  # output_shape = [b, w, h, c]
  # sess_temp = tf.InteractiveSession()
  sess_temp = tf.global_variables_initializer()
  strides = [1, stride, stride, 1]
  with tf.variable_scope(name):
    weights = get_deconv_filter(f_shape)
    deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,
                                        strides=strides, padding='SAME')
  return deconv

def batch_norm_layer(inputT, is_training, scope):
  return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))


def inference(images, labels, batch_size, phase_train,weighted, w_type,dl = 1):
    # norm1
    norm1 = tf.nn.lrn(images, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75,
                name='norm1')
    # conv1
    conv1 = conv_layer_with_bn(norm1, [7, 7, images.get_shape().as_list()[3], dl*64], phase_train, name="conv1")
    # pool1
    pool1, pool1_indices = tf.nn.max_pool_with_argmax(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # conv2
    conv2 = conv_layer_with_bn(pool1, [7, 7, dl*64, dl*64], phase_train, name="conv2")

    # pool2
    pool2, pool2_indices = tf.nn.max_pool_with_argmax(conv2, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    # conv3
    conv3 = conv_layer_with_bn(pool2, [7, 7, dl*64, dl*64], phase_train, name="conv3")

    # pool3
    pool3, pool3_indices = tf.nn.max_pool_with_argmax(conv3, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    # conv4
    conv4 = conv_layer_with_bn(pool3, [7, 7, dl*64, dl*64], phase_train, name="conv4")

    # pool4
    pool4, pool4_indices = tf.nn.max_pool_with_argmax(conv4, ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    """ End of encoder """
    """ start upsample """
    # upsample4
    # Need to change when using different dataset out_w, out_h
    # upsample4 = upsample_with_pool_indices(pool4, pool4_indices, pool4.get_shape(), out_w=45, out_h=60, scale=2, name='upsample4')
    upsample4 = deconv_layer(pool4, [2, 2, dl*64, dl*64], [batch_size, 40, 40, dl*64], 2, "up4")
    # decode 4
    conv_decode4 = conv_layer_with_bn(upsample4, [7, 7, dl*64, dl*64], phase_train, False, name="conv_decode4")

    # upsample 3
    # upsample3 = upsample_with_pool_indices(conv_decode4, pool3_indices, conv_decode4.get_shape(), scale=2, name='upsample3')
    upsample3= deconv_layer(conv_decode4, [2, 2, dl*64, dl*64], [batch_size, 80, 80, dl*64], 2, "up3")
    # decode 3
    conv_decode3 = conv_layer_with_bn(upsample3, [7, 7, dl*64, dl*64], phase_train, False, name="conv_decode3")

    # upsample2
    # upsample2 = upsample_with_pool_indices(conv_decode3, pool2_indices, conv_decode3.get_shape(), scale=2, name='upsample2')
    upsample2= deconv_layer(conv_decode3, [2, 2, dl*64, dl*64], [batch_size, 160, 160, dl*64], 2, "up2")
    # decode 2
    conv_decode2 = conv_layer_with_bn(upsample2, [7, 7, dl*64, dl*64], phase_train, False, name="conv_decode2")

    # upsample1
    # upsample1 = upsample_with_pool_indices(conv_decode2, pool1_indices, conv_decode2.get_shape(), scale=2, name='upsample1')
    upsample1= deconv_layer(conv_decode2, [2, 2, dl*64, dl*64], [batch_size, 320, 320, dl*64], 2, "up1")
    # decode4
    conv_decode1 = conv_layer_with_bn(upsample1, [7, 7, dl*64, dl*64], phase_train, False, name="conv_decode1")
    """ end of Decode """
    """ Start Classify """
    # output predicted class number (6)
    with tf.variable_scope('conv_classifier') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[1, 1, dl*64, NUM_CLASSES],
                                           initializer=msra_initializer(1, dl*64),
                                           wd=0.0005)
      conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))
      conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    logit = conv_classifier
    print("this is weights",weighted)
    if weighted:
        loss = cal_loss(conv_classifier, labels, w_type)
    else:
        loss = loss_ww_manual(logit, labels)
    return loss, logit

def train(total_loss, global_step):
    total_sample = 92961
    num_batches_per_epoch = 92961/1
    """ fix lr """
    lr = INITIAL_LEARNING_RATE
    loss_averages_op = _add_loss_summaries(total_loss)
    print("totalloss", total_loss)
    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
      opt = tf.train.AdamOptimizer(lr)
      grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
      train_op = tf.no_op(name='train')

    return train_op

def test(FLAGS):
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  train_dir = FLAGS.log_dir # /tmp3/first350/TensorFlow/Logs
  test_dir = FLAGS.test_dir # /tmp3/first350/SegNet-Tutorial/CamVid/train.txt
  test_ckpt = FLAGS.testing
  image_w = FLAGS.image_w
  image_h = FLAGS.image_h
  image_c = FLAGS.image_c
  # testing should set BATCH_SIZE = 1
  batch_size = 1
  weighted = FLAGS.weighted_loss
  w_t = FLAGS.w_type
  dbf = FLAGS.doubled_filters
  
  photos_list = ['ung_104_36_c0011_7.jpg',
                 '36_10_c0019_10.jpg',
                 '40_02_c0011_19.jpg',
                 '104_52_c0002_64.jpg']
  
  if dbf == True: db = 2
  else: db = 1
  
  result_folder = "./out/"
  result_folder_2 = "./pred/"

  print(test_ckpt)
  image_filenames, label_filenames = get_filename_list(test_dir)

  test_data_node = tf.placeholder(
        tf.float32,
        shape=[batch_size, image_h, image_w, image_c])

  test_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])

  phase_train = tf.placeholder(tf.bool, name='phase_train')

  loss, logits = inference(test_data_node, test_labels_node, batch_size, phase_train,weighted,w_t, db)

  pred = tf.argmax(logits, axis=3)
  # get moving avg
  variable_averages = tf.train.ExponentialMovingAverage(
                      MOVING_AVERAGE_DECAY)
  variables_to_restore = variable_averages.variables_to_restore()

  saver = tf.train.Saver(variables_to_restore, max_to_keep=1)

  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)
  f1_ll = []
  with tf.Session() as sess:
    # Load checkpoint
    saver.restore(sess, test_ckpt )

    images, labels = get_all_test_data(image_filenames, label_filenames)

    threads = tf.train.start_queue_runners(sess=sess)
    hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
    count = 0
    bb = 0
    for image_batch, label_batch  in zip(images, labels):

      feed_dict = {
        test_data_node: image_batch,
        test_labels_node: label_batch,
        phase_train: False
      }

      dense_prediction, im = sess.run([logits, pred], feed_dict=feed_dict)
      
      
      #########################################################################
#      if len(os.listdir(result_folder_2)) <30:
#          if np.random.rand(1)<0.05:
#              np.save(result_folder_2+os.path.basename(label_filenames[count]+".npy"),dense_prediction)
#              print(bb)
#              bb +=1
      
      ############################################33333    
      
      pred_use_val = dense_prediction.argmax(3).astype(int)
      pred_use_val = np.reshape(pred_use_val,-1)
      labels_val = np.reshape(label_batch,-1)
      prec = precision_score(labels_val.astype(int), pred_use_val, average="macro")
      rec = recall_score(labels_val, pred_use_val, average="macro")
      f1 = 2*prec*rec/(prec+rec)
      f1_ll.append(f1)
      # output_image to verify
      if (FLAGS.save_image):
          if len(os.listdir(result_folder)) <25:
              if np.random.rand(1) <0.25: 
                  #writeImage(im[0], './out/testing_image.png')
                  #writeImage(im[0], './out/'+str(image_filenames[count]).split('/')[-1])
                  img = Image.fromarray(np.uint8(im[0]))	#add mahmuds
                  img.save(result_folder+os.path.basename(label_filenames[count]))
                  print(result_folder+os.path.basename(label_filenames[count]))
          if image_filenames[count].split("/")[-1] in photos_list:
              print("image saved")
              img = Image.fromarray(np.uint8(im[0]))	#add mahmuds
              img.save(result_folder+os.path.basename(label_filenames[count]))
      count += 1
      if count % 200 == 0: print(count)
      
      hist += get_hist(dense_prediction, label_batch)
      # count+=1
    acc_total = np.diag(hist).sum() / hist.sum()
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print("acc: ", acc_total)
    print("mean IU: ", np.nanmean(iu))
    perf_arr = np.array([np.nanmean(iu),acc_total, np.nanmean(f1_ll)])
    
    ####acc_class
    num_class = dense_prediction.shape[3]
    acc_class = []
    for ii in range(num_class):
      if float(hist.sum(1)[ii]) == 0:
          acc = 0.0
          acc_class.append(acc)
      else:
          acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
          acc_class.append(acc) 
  
    ###
    
    perf_arr = np.append(perf_arr, acc_class)
    np.save("./loss_data/loss_test.npy",perf_arr)
    
    
    
    
def training(FLAGS, is_finetune=False):
  max_steps = FLAGS.max_steps
  batch_size = FLAGS.batch_size
  train_dir = FLAGS.log_dir # /tmp3/first350/TensorFlow/Logs
  image_dir = FLAGS.image_dir # /tmp3/first350/SegNet-Tutorial/CamVid/train.txt
  val_dir = FLAGS.val_dir # /tmp3/first350/SegNet-Tutorial/CamVid/val.txt
  finetune_ckpt = FLAGS.finetune
  image_w = FLAGS.image_w
  image_h = FLAGS.image_h
  image_c = FLAGS.image_c
  weighted = FLAGS.weighted_loss
  w_t = FLAGS.w_type
  mirror = FLAGS.random_mirror
  scale = FLAGS.random_scale
  dbf = FLAGS.doubled_filters
  
  if dbf == True: db = 2
  else: db = 1
  
  if os.path.exists("./loss_data/loss_metrics.npy"):
    ll = np.load("./loss_data/loss_metrics.npy")
    lv = np.load("./loss_data/loss_metrics_2.npy")
    list_train = list(ll)
    list_val = list(lv)
  else:
    list_train = []
    list_val = []
        
  # should be changed if your model stored by different convention
  startstep = 0 if not is_finetune else int(FLAGS.finetune.split('-')[-1])

  image_filenames, label_filenames = get_filename_list(image_dir)
  val_image_filenames, val_label_filenames = get_filename_list(val_dir)

  with tf.Graph().as_default():

    train_data_node = tf.placeholder( tf.float32, shape=[batch_size, image_h, image_w, image_c])

    train_labels_node = tf.placeholder(tf.int64, shape=[batch_size, image_h, image_w, 1])

    phase_train = tf.placeholder(tf.bool, name='phase_train')

    global_step = tf.Variable(0, trainable=False)

    # For CamVid
    images, labels = CamVidInputs(image_filenames, label_filenames, batch_size, mirror, scale)

    val_images, val_labels = CamVidInputs(val_image_filenames, val_label_filenames, batch_size, mirror, scale)

    # Build a Graph that computes the logits predictions from the inference model.
    loss, eval_prediction = inference(train_data_node, train_labels_node, batch_size, phase_train,weighted,w_t, db)

    # Build a Graph that trains the model with one batch of examples and updates the model parameters.
    train_op = train(loss, global_step)

    saver = tf.train.Saver(tf.global_variables())

    summary_op = tf.summary.merge_all()
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.allocator_type = 'BFC'
    config.allow_soft_placement = True
    run_opts = tf.RunOptions()
    run_opts.report_tensor_allocations_upon_oom = True

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.0001)

    with tf.Session(config=config) as sess:
      # Build an initialization operation to run below.
      if (is_finetune == True):
          saver.restore(sess, finetune_ckpt )
      else:
          init = tf.global_variables_initializer()
          sess.run(init)

      # Start the queue runners.
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)

      # Summery placeholders
      summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
      average_pl = tf.placeholder(tf.float32)
      acc_pl = tf.placeholder(tf.float32)
      iu_pl = tf.placeholder(tf.float32)
      average_summary = tf.summary.scalar("test_average_loss", average_pl)
      acc_summary = tf.summary.scalar("test_accuracy", acc_pl)
      iu_summary = tf.summary.scalar("Mean_IU", iu_pl)
      
      #######################################3
      total_parameters = 0
      for variable in tf.trainable_variables():
          # shape is an array of tf.Dimension
          shape = variable.get_shape()

          variable_parameters = 1
          for dim in shape:
              variable_parameters *= dim.value
          
          total_parameters += variable_parameters
      print("Total parameters", total_parameters)

      for step in range(startstep, startstep + max_steps):
        
        image_batch ,label_batch = sess.run([images, labels])
        # since we still use mini-batches in validation, still set bn-layer phase_train = True
        feed_dict = {
          train_data_node: image_batch,
          train_labels_node: label_batch,
          phase_train: True
        }
        start_time = time.time()

        
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict, options = run_opts)
        duration = time.time() - start_time

        #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
          num_examples_per_step = batch_size
          examples_per_sec = num_examples_per_step / duration
          sec_per_batch = float(duration)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
          print(format_str % (datetime.now(), step, loss_value,
                               examples_per_sec, sec_per_batch))

          # eval current training batch pre-class accuracy
          pred = sess.run(eval_prediction, feed_dict=feed_dict, options = run_opts)
          aacy, miou,p,r,f,acc_class = per_class_acc(pred, label_batch)
          aar = np.array([step,loss_value,aacy,miou,p,r,f])
          aar = np.append(aar, acc_class)
         
          list_train.append(aar)
          np.save("./loss_data/loss_metrics.npy",np.array(list_train))
          
        if step % 250 == 0:
          print("start validating.....")
          print(TEST_ITER)
          total_val_loss = 0.0
          hist = np.zeros((NUM_CLASSES, NUM_CLASSES))
          ###################33
          f1_l = []
          ######################
          for test_step in range(int(TEST_ITER)):
            val_images_batch, val_labels_batch = sess.run([val_images, val_labels])

            _val_loss, _val_pred = sess.run([loss, eval_prediction], feed_dict={
              train_data_node: val_images_batch,
              train_labels_node: val_labels_batch,
              phase_train: True
            })
            total_val_loss += _val_loss
            hist += get_hist(_val_pred, val_labels_batch)
            
            ###############################metrics val
            pred_use_val = _val_pred.argmax(3).astype(int)
            pred_use_val = np.reshape(pred_use_val,-1)
            labels_val = np.reshape(val_labels_batch,-1)
            prec = precision_score(labels_val.astype(int), pred_use_val, average="macro")
            rec = recall_score(labels_val, pred_use_val, average="macro")
            f1 = 2*prec*rec/(prec+rec)
            f1_l.append(f1)
            
            ###################################
            
          f1_f = np.mean(f1_l)
          print("val loss: ", total_val_loss / TEST_ITER)
          acc_total = np.diag(hist).sum() / hist.sum()
          iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
          test_summary_str = sess.run(average_summary, feed_dict={average_pl: total_val_loss / TEST_ITER})
          acc_summary_str = sess.run(acc_summary, feed_dict={acc_pl: acc_total})
          iu_summary_str = sess.run(iu_summary, feed_dict={iu_pl: np.nanmean(iu)})
          print_hist_summery(hist)
          
          ####acc_class
          num_class = _val_pred.shape[3]
          acc_class = []
          for ii in range(num_class):
            if float(hist.sum(1)[ii]) == 0:
                acc = 0.0
                acc_class.append(acc)
            else:
                acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
                acc_class.append(acc) 
          
          ###
          
          lusy = total_val_loss / TEST_ITER
          lisy = np.array([step,lusy, np.nanmean(iu), acc_total, f1_f])
          lisy = np.append(lisy, acc_class)
          list_val.append(lisy)
          np.save("./loss_data/loss_metrics_2.npy",np.array(list_val))
          print(" end validating.... ")

          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, step)
          summary_writer.add_summary(test_summary_str, step)
          summary_writer.add_summary(acc_summary_str, step)
          summary_writer.add_summary(iu_summary_str, step)
        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == max_steps:
          checkpoint_path = os.path.join(train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)

      coord.request_stop()
      coord.join(threads)
