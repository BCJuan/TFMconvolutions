import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import os, sys
import numpy as np
import math
import skimage
import skimage.io

IMAGE_HEIGHT = 320
IMAGE_WIDTH = 320
IMAGE_DEPTH = 3

NUM_CLASSES = 25
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 92961
NUM_EXAMPLES_PER_EPOCH_FOR_TEST = 15233
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 1

def image_mirroring(img, label):
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    
    return img, label

def image_scaling(img, label):
    scale = tf.random_uniform([1], minval=0.5, maxval=2.0, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])

    return img, label


def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, ignore_label=255):
    label = tf.cast(label, dtype=tf.uint8)
    image = tf.cast(image, dtype=tf.uint8)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4])
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)

    # Set static shape so that tensorflow knows shape at compile time.
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h,crop_w, 1))
    return img_crop, label_crop

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 3-D Tensor of [height, width, 1] type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 3D tensor of [batch_size, height, width ,1] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 1
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  # tf.image_summary('images', images)

  return images, label_batch

def CamVid_reader_seq(filename_queue, seq_length):
  image_seq_filenames = tf.split(axis=0, num_or_size_splits=seq_length, value=filename_queue[0])
  label_seq_filenames = tf.split(axis=0, num_or_size_splits=seq_length, value=filename_queue[1])

  image_seq = []
  label_seq = []
  for im ,la in zip(image_seq_filenames, label_seq_filenames):
    imageValue = tf.read_file(tf.squeeze(im))
    labelValue = tf.read_file(tf.squeeze(la))
    image_bytes = tf.image.decode_png(imageValue)
    label_bytes = tf.image.decode_png(labelValue)
    image = tf.cast(tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH)), tf.float32)
    label = tf.cast(tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1)), tf.int64)
    image_seq.append(image)
    label_seq.append(label)
  return image_seq, label_seq

def CamVid_reader(filename_queue, mirror, scale):

  image_filename = filename_queue[0]
  label_filename = filename_queue[1]

  imageValue = tf.read_file(image_filename)
  labelValue = tf.read_file(label_filename)

  image_bytes = tf.image.decode_png(imageValue)
  label_bytes = tf.image.decode_png(labelValue)

  image = tf.reshape(image_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
  label = tf.reshape(label_bytes, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
  
  if scale:
     image, label = image_scaling(image, label)
    
  if mirror:
     image, label = image_mirroring(image, label)
    
  image, label = random_crop_and_pad_image_and_labels(image, label, IMAGE_HEIGHT, IMAGE_WIDTH)
      
  return image, label

def get_filename_list(path):
  fd = open(path)
  image_filenames = []
  label_filenames = []
  filenames = []
  for i in fd:
    i = i.strip().split(" ")
    image_filenames.append(i[0])
    label_filenames.append(i[1])
  return image_filenames, label_filenames

def CamVidInputs(image_filenames, label_filenames, batch_size, mirror, scale):

  images = ops.convert_to_tensor(image_filenames, dtype=dtypes.string)
  labels = ops.convert_to_tensor(label_filenames, dtype=dtypes.string)

  filename_queue = tf.train.slice_input_producer([images, labels], shuffle=True)

  image, label = CamVid_reader(filename_queue, mirror, scale)
  reshaped_image = tf.cast(image, tf.float32)

  min_fraction_of_examples_in_queue = 0.2
  min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CamVid images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(reshaped_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)
def get_all_test_data(im_list, la_list):
  images = []
  labels = []
  index = 0
  for im_filename, la_filename in zip(im_list, la_list):
    im = np.array(skimage.io.imread(im_filename), np.float32)
    im = im[np.newaxis]
    la = skimage.io.imread(la_filename)
    la = la[np.newaxis]
    la = la[...,np.newaxis]
    images.append(im)
    labels.append(la)
  return images, labels
