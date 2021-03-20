import os
import scipy.misc
import numpy as np
from model import DCGAN
from utils import image_viz, image_inverse
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 55, "Number of epochs = 55 ")
flags.DEFINE_float("learning_rate", 1e-5, "Learning Rate = 1e-5")
flags.DEFINE_float("beta1", 0.9, "Momentum for ADAM optimizer = 0.9")
flags.DEFINE_integer("train_size", 10000, "Size of Training Images = 10000")
flags.DEFINE_integer("batch_size", 64, "Batch size = 64")
flags.DEFINE_integer("input_height", 64, "The input image height, center cropped")
flags.DEFINE_integer("input_width", None, "The input image width, center cropped")
flags.DEFINE_integer("output_height", 64, "The output image height")
flags.DEFINE_integer("output_width", None, "The output image width")
flags.DEFINE_string("dataset", "meta_AmazonFashion.json", "The name of dataset")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Pattern for input images")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Folder to save the checkpoints")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples")
flags.DEFINE_boolean("train", False, "Parameter should be set to 'True' for training and 'False' for testing")
flags.DEFINE_boolean("crop", False, "Parameter should be set to 'True' for training and 'False' for testing")
flags.DEFINE_boolean("visualize", False, "Parameter should be set to 'True' for visualizing and 'False' for testing")
FLAGS = flags.FLAGS

def main(_):
  print(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    dcgan = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        y_dim=6,
        dataset_name=FLAGS.dataset,
        input_fname_pattern=FLAGS.input_fname_pattern,
        crop=FLAGS.crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir)

    image_inverse()

    if FLAGS.train:
      dcgan.train(FLAGS)
    else:
      if not dcgan.load(FLAGS.checkpoint_dir)[0]:
        raise Exception("[!] Train a model first, then run test mode")
      
    # Below is codes for visualization
    OPTION = 1
    image_viz(sess, dcgan, FLAGS, OPTION)

if __name__ == '__main__':
  tf.app.run()
