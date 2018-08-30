import tensorflow as tf
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.multiclass import type_of_target


def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])

  # Attach a scalar summary to all individual losses and the total loss; do the
  # same for the averaged version of the losses.
  for l in losses + [total_loss]:
    # Name each loss as '(raw)' and name the moving average version of the loss
    # as the original loss name.
    tf.summary.scalar(l.op.name +' (raw)', l)
    tf.summary.scalar(l.op.name, loss_averages.average(l))

  return loss_averages_op

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  with tf.device('/gpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var

def _variable_with_weight_decay(name, shape, initializer, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(
      name,
      shape,
      initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def writeImage(image, filename):
    """ store label data to colored image """
    
    Road = [0, 0, 0]
    Sidewalk = [153, 76, 0]
    Building = [153, 153, 0]
    Wall = [76, 153, 0] 
    Fence = [0, 153, 0] 
    Pole = [0, 153, 76]
    TrafficLight = [0,153, 153]
    TrafficSign = [0, 76, 153]
    Vegetation = [0, 0, 153]
    Terrain = [76, 0, 153]
    Sky = [153, 0, 153]
    Person = [153, 0, 76]
    Rider = [255, 51, 51]
    Car =[255, 153, 51]
    Truck = [255, 255, 51]
    Bus = [153, 255, 51]
    Train = [51, 255, 51]
    Motorcycle = [51, 255, 153]
    Bicycle = [51, 255, 255]
    Other = [51,153,255]
    LicencePlate = [51,51,255]
    Other_2 = [153,51,255]
    Other_3 = [255,52,255]
    Other_4 = [255,52,153]
    Other_5 = [192,192,192]

    r = image.copy()
    g = image.copy()
    b = image.copy()
    label_colours = np.array([Road, Sidewalk, Building, Wall, Fence, Pole, TrafficLight, TrafficSign, Vegetation, Terrain, 
                              Sky, Person, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle, Other, LicencePlate, Other_2,
                              Other_3, Other_4, Other_5])	#change
    for l in range(0,25):
        r[image==l] = label_colours[l,0]
        g[image==l] = label_colours[l,1]
        b[image==l] = label_colours[l,2]
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    rgb[:,:,0] = r/1.0
    rgb[:,:,1] = g/1.0
    rgb[:,:,2] = b/1.0
    im = Image.fromarray(np.uint8(rgb))
    im.save(filename)

def storeImageQueue(data, labels, step):
  """ data and labels are all numpy arrays """
  for i in range(BATCH_SIZE):
    index = 0
    im = data[i]
    la = labels[i]
    im = Image.fromarray(np.uint8(im))
    im.save("batch_im_s%d_%d.png"%(step,i))
    writeImage(np.reshape(la,(320,320)), "batch_la_s%d_%d.png"%(step,i))

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def get_hist(predictions, labels):
  num_class = predictions.shape[3]
  batch_size = predictions.shape[0]
  hist = np.zeros((num_class, num_class))
  for i in range(batch_size):
    hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
  return hist

def print_hist_summery(hist):
  acc_total = np.diag(hist).sum() / hist.sum()
  print ('accuracy = %f'%np.nanmean(acc_total))
  iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
  print ('mean IU  = %f'%np.nanmean(iu))
  for ii in range(hist.shape[0]):
      if float(hist.sum(1)[ii]) == 0:
        acc = 0.0
      else:
        acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
      print("    class # %d accuracy = %f "%(ii, acc))

def per_class_acc(predictions, label_tensor):
    labels = label_tensor
    size = predictions.shape[0]
    num_class = predictions.shape[3]
    hist = np.zeros((num_class, num_class))
    for i in range(size):
      hist += fast_hist(labels[i].flatten(), predictions[i].argmax(2).flatten(), num_class)
    acc_total = np.diag(hist).sum() / hist.sum()
    print ('accuracy = %f'%np.nanmean(acc_total))
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    print ('mean IU  = %f'%np.nanmean(iu))
    acc_class = []
    for ii in range(num_class):
        if float(hist.sum(1)[ii]) == 0:
          acc = 0.0
          acc_class.append(acc)
        else:
          acc = np.diag(hist)[ii] / float(hist.sum(1)[ii])
          acc_class.append(acc)
            
    pred_use = predictions.argmax(3).astype(int)
    pred_use = np.reshape(pred_use,-1)
    labels = np.reshape(labels,-1)
    prec = precision_score(labels.astype(int), pred_use, average="macro")
    rec = recall_score(labels, pred_use, average="macro")
    f1 = 2*prec*rec/(prec+rec)
    print("P, R, F1",prec,";",rec,";",f1)
    return acc_total, np.nanmean(iu), prec, rec,f1,acc_class