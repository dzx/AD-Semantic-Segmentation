#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 19:44:42 2018

@author: dzx
"""

import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_trained(sess, trained_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, logits)
    """
    trained_tag = 'VGG_Trained'
    trained_input_tensor_name = 'image_input:0'
    trained_keep_prob_tensor_name = 'keep_prob:0'
    trained_logits_tensor_name = 'logits:0'
    tf.saved_model.loader.load(sess, [trained_tag], trained_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(trained_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(trained_keep_prob_tensor_name)
    logits = graph.get_tensor_by_name(trained_logits_tensor_name)
    
    return image_input, keep_prob, logits

def run():
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    with tf.Session() as sess:
        trained_path = os.path.join(data_dir, 'trained')
        image_input, keep_prob, logits = load_trained(sess, trained_path)
        print('Trained model loaded. Running inference on test data.')
#        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)
        helper.process_video(data_dir, sess, image_shape, logits, keep_prob,
                             image_input)

if __name__ == '__main__':
    run()
