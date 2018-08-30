import tensorflow as tf
import model
import os



FLAGS = tf.app.flags.FLAGS

#changed
#image width. height
#number of classes
#image folders (structure can be seen at segnet caffe tutorial)

tf.app.flags.DEFINE_string('testing', '', """ checkpoint file """)
tf.app.flags.DEFINE_string('finetune', '', """ finetune checkpoint file """)
tf.app.flags.DEFINE_integer('batch_size', "8", """ batch_size """)
tf.app.flags.DEFINE_float('learning_rate', "1e-3", """ initial lr """)
tf.app.flags.DEFINE_string('log_dir', "./log/", """ dir to store ckpt """)
tf.app.flags.DEFINE_string('image_dir', "./train/train.txt", """ path to CamVid image """)
tf.app.flags.DEFINE_string('test_dir', "./test/test.txt", """ path to CamVid test image """)
tf.app.flags.DEFINE_string('val_dir', "./val/val.txt", """ path to CamVid val image """)
tf.app.flags.DEFINE_integer('max_steps', "45001", """ max_steps """)
tf.app.flags.DEFINE_integer('image_h', "320", """ image height """)
tf.app.flags.DEFINE_integer('image_w', "320", """ image width """)
tf.app.flags.DEFINE_integer('image_c', "3", """ image channel (RGB) """)
tf.app.flags.DEFINE_integer('num_class', "25", """ total class number """)
tf.app.flags.DEFINE_boolean('save_image', True, """ whether to save predicted image """)
tf.app.flags.DEFINE_boolean('weighted_loss', False, """ whether to weight loss or not """)
tf.app.flags.DEFINE_string('w_type', "normal", """ type of weighting:direct or normal""")
tf.app.flags.DEFINE_boolean('random_mirror', False, """ mirror images """)
tf.app.flags.DEFINE_boolean('random_scale', False, """ scale images """)
tf.app.flags.DEFINE_boolean('doubled_filters', True, """ double filters """)


def checkArgs():
    if FLAGS.testing != '':
        print('The model is set to Testing')
        print("check point file: %s"%FLAGS.testing)
        print("CamVid testing dir: %s"%FLAGS.test_dir)
    elif FLAGS.finetune != '':
        print('The model is set to Finetune from ckpt')
        print("check point file: %s"%FLAGS.finetune)
        print("CamVid Image dir: %s"%FLAGS.image_dir)
        print("CamVid Val dir: %s"%FLAGS.val_dir)
    else:
        print('The model is set to Training')
        print("Max training Iteration: %d"%FLAGS.max_steps)
        print("Initial lr: %f"%FLAGS.learning_rate)
        print("CamVid Image dir: %s"%FLAGS.image_dir)
        print("CamVid Val dir: %s"%FLAGS.val_dir)

    print("Batch Size: %d"%FLAGS.batch_size)
    print("Log dir: %s"%FLAGS.log_dir)
    print("Loss weight: %s "%FLAGS.weighted_loss)
    print("tyoe of weighting: %s "%FLAGS.w_type)

def main(args):
    checkArgs()
    if FLAGS.testing:
        model.test(FLAGS)
    elif FLAGS.finetune:
        model.training(FLAGS, is_finetune=True)
    else:
        model.training(FLAGS, is_finetune=False)

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    tf.app.run()
