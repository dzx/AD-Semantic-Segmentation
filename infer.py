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
import time
import cv2

#from tensorflow.python.client import timeline

# Check TensorFlow Version
#assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
#print('TensorFlow Version: {}'.format(tf.__version__))
#
## Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
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
    logits = tf.argmax(logits, -1)
    
    return image_input, keep_prob, logits

# Define encoder function
def encode(array):
    #print(array.shape)
    pil_img = Image.fromarray(array)
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")

def gen_frame_function(file_name):
    video = skvideo.io.vread(file_name)
    def get_frames_fn(batch_size):
        for batch_i in range(0, len(video), batch_size):
            frames = video[batch_i:batch_i+batch_size]
            yield frames
    return get_frames_fn

def annotate_video(file_name, sess, logits, keep_prob, image_pl, image_shape, crop):
    batch_size = 18
#    video = skvideo.io.vread(file_name)
    frame_gen = gen_frame_function(file_name)

    answer_key = {}
    padding_t, padding_b, bottom = helper.create_paddings(image_shape, crop, stacks=batch_size)
#    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#    run_metadata = tf.RunMetadata()
    # Frame numbering starts at 1
    inference_size = (640,256)
    cropped_size = (image_shape[1], image_shape[0]-sum((crop)))
    frame = 1
    for frames in frame_gen(batch_size):
        #print("frame shape {}".format(rgb_frame.shape))
        frames = [cv2.resize(frame[crop[0]:bottom], dsize=inference_size, interpolation=cv2.INTER_AREA) 
                    for frame in frames]
#        print(len(frames), frames[0].shape)
        segments = helper.segment_images(sess, logits, keep_prob, image_pl, 
                                        np.asarray(frames), 2)
#        print("dsize", cropped_size)
        for i in range(len(segments)):
#            print(segments[i].shape)
            blown_up = [cv2.resize(frame, dsize=cropped_size, interpolation=cv2.INTER_LINEAR)
                    for frame in segments[i]]
            segments[i] = helper.pad_segment(np.asarray(blown_up, dtype='uint8'), padding_t, padding_b)
        for i in range(len(frames)):
            answer_key[frame+i] = [encode(segments[1][i]), encode(segments[0][i])] # cars, road
#        fetched_timeline = timeline.Timeline(run_metadata.step_stats)
#        chrome_trace = fetched_timeline.generate_chrome_trace_format()
#        with open('timeline_op_step_%d.json' % frame, 'w') as f:
#            f.write(chrome_trace)
        # Increment frame
        frame+=len(frames)
    return answer_key

def run():
    image_shape = (600, 800)
    crop = (206, 74)
    data_dir = './'
    runs_dir = './runs'
    log_dir = './infer_log'
    input_video = sys.argv[-1]
    with tf.gfile.GFile('901_opt.pb', 'rb') as f:
        graph_def_optimized = tf.GraphDef()
        graph_def_optimized.ParseFromString(f.read())
    G = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    start_time = time.perf_counter()
    with tf.Session(graph=G) as sess: #config=config, graph=G
#        trained_path = os.path.join(data_dir, 'trained')
#         image_input, keep_prob, logits = load_trained(sess, trained_path,
#                                                     return_elements=['image_input:0', 'keep_prob:0', 'logits:0'])
        image_input, keep_prob, logits = tf.import_graph_def(graph_def_optimized, 
                                                             return_elements=['image_input:0', 'keep_prob:0', 'logits:0'])
        predictions = tf.argmax(logits, axis=-1, name='preds')

#        writter = tf.summary.FileWriter(log_dir, sess.graph)

        print('Trained model loaded at {}.  Running inference on test data.'.format(time.perf_counter() - start_time), file=sys.stderr)
        annotations = annotate_video(input_video, sess, predictions, keep_prob, image_input,
                                     image_shape, crop)
#        writter.close()
        print(json.dumps(annotations))
        print('Done at {}'.format(time.perf_counter() - start_time), file=sys.stderr)
#        images, _, [_, val_idxs] = helper.gen_train_val_folds(os.path.join(data_dir, 'Train'), .1, 42)
#        test_paths = [images[i] for i in val_idxs]
#        helper.save_inference_samples(runs_dir, test_paths, sess, image_shape, logits, 
#                                      keep_prob, image_input, crop)
#        helper.process_video(data_dir, sess, image_shape, logits, keep_prob,
#                             image_input)

if __name__ == '__main__':
    run()
