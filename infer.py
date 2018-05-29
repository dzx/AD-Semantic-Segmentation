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
import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO


# Check TensorFlow Version
#assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
#print('TensorFlow Version: {}'.format(tf.__version__))
#
## Check for a GPU
#if not tf.test.gpu_device_name():
#    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
#else:
#    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

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

# Define encoder function
def encode(array):
    #print(array.shape)
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def annotate_video(file_name, sess, logits, keep_prob, image_pl, image_shape, crop):
    video = skvideo.io.vread(file_name)

    answer_key = {}
    padding_t, padding_b, bottom = helper.create_paddings(image_shape, crop)
    # Frame numbering starts at 1
    frame = 1
    for rgb_frame in video:
        #print("frame shape {}".format(rgb_frame.shape))
        segments = helper.segment_image(sess, logits, keep_prob, image_pl, 
                                        rgb_frame[crop[0]:bottom], 2)
        for i in range(len(segments)):
            segments[i] = helper.pad_segment(segments[i], padding_t, padding_b)
            
        answer_key[frame] = [encode(segments[1]), encode(segments[0])] # cars, road
        # Increment frame
        frame+=1
    return answer_key

def run():
    image_shape = (600, 800)
    crop = (206, 74)
    data_dir = './'
    runs_dir = './runs'
    input_video = sys.argv[-1]

    with tf.Session() as sess:
        trained_path = os.path.join(data_dir, 'trained')
        image_input, keep_prob, logits = load_trained(sess, trained_path)
        #print('Trained model loaded. Running inference on test data.')
        annotations = annotate_video(input_video, sess, logits, keep_prob, image_input,
                                     image_shape, crop)
        print(json.dumps(annotations))
#        images, _, [_, val_idxs] = helper.gen_train_val_folds(os.path.join(data_dir, 'Train'), .1, 42)
#        test_paths = [images[i] for i in val_idxs]
#        helper.save_inference_samples(runs_dir, test_paths, sess, image_shape, logits, 
#                                      keep_prob, image_input, crop)
#        helper.process_video(data_dir, sess, image_shape, logits, keep_prob,
#                             image_input)

if __name__ == '__main__':
    run()
