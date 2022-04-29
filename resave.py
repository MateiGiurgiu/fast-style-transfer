import transform, numpy as np, vgg, pdb, os
import scipy.misc
import tensorflow as tf
from utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import time
import json
import subprocess
import numpy

DEVICE = '/gpu:0'

def process(checkpoint_dir, image_in, out_path, device_t='/gpu:0'):

  img_shape = get_img(image_in[0]).shape

  g = tf.Graph()
  curr_num = 0
  soft_config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
  soft_config.gpu_options.allow_growth = True
  with g.as_default(), g.device(device_t), tf.compat.v1.Session(config=soft_config) as sess:
    batch_shape = (1,) + img_shape
    img_placeholder = tf.compat.v1.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')

    preds = transform.net(img_placeholder)
    saver = tf.compat.v1.train.Saver()
    if os.path.isdir(checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("No checkpoint found...")
    else:
        saver.restore(sess, checkpoint_dir)

    tf.compat.v1.saved_model.simple_save(sess, out_path, inputs={"X_content": img_placeholder}, outputs={"Y_output": preds})
      

def build_parser():
  parser = ArgumentParser()
  parser.add_argument('--checkpoint', type=str,
                      dest='checkpoint_dir',
                      help='dir or .ckpt file to load checkpoint from',
                      metavar='CHECKPOINT', required=True)

   parser.add_argument('--image-in', type=str,
                      dest='image_in',help='image to get the size from',
                      metavar='IMAGE_IN', required=True)

  parser.add_argument('--out-path', type=str,
                      dest='out_path', help='output directory', metavar='OUT_PATH',
                      required=True)

  return parser

def check_opts(opts):
  exists(opts.checkpoint_dir, 'Checkpoint not found!')
  exists(opts.out_path, 'Out dir not found!')

  
def main():
  parser = build_parser()
  opts = parser.parse_args()
  check_opts(opts)
  
  process(opts.checkpoint_dir, opts.image_in, opts.out_path, DEVICE)

  
if __name__ == '__main__':
  main()
