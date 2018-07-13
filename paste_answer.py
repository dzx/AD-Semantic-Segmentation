#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 02:10:41 2018

@author: dzx
"""

import helper
import warnings
import cv2
import sys, skvideo.io, json, base64
import matplotlib.pyplot as plt
from scipy import misc

def cv_frame_function(file_name):
    video = cv2.VideoCapture(file_name)
#    print("Opened Video")
    def get_cframes_fn(batch_size):
        frames = []
        loaded = 0
        loading = True
#        print(video.isOpened())
        while video.isOpened() and loading:
            (loading, frame) = video.read()
            if loading:
                loaded += 1
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if loaded == batch_size:
                    yield frames
                    frames = []
                    loaded = 0
            else:
                video.release()
        if len(frames):
            yield frames
    return get_cframes_fn

def decode(packet):
	img = base64.b64decode(packet)
	filename = './image.png'
	with open(filename, 'wb') as f:
			f.write(img)
	result = misc.imread(filename)
	return result

with open('./ep27_ds.json') as json_data:
	ans_data = json.loads(json_data.read())
	json_data.close()

input_video = sys.argv[-1]
frame_gen = cv_frame_function(input_video)
frame_count = 0
mask_colors = [[0, 255, 0, 127],[255, 0, 0, 127]] 
for frames in frame_gen(5):
    for frame in frames:
        frame_count += 1
        print(frame_count)
        masks = [decode(ans_data[str(frame_count)][1]), decode(ans_data[str(frame_count)][0])]
        img = misc.toimage(frame)
        fig, (left, right) = plt.subplots(ncols=2, figsize=(12, 9))
        left.imshow(img)
        annotated = helper.mask_image(mask_colors, img, masks)
        right.imshow(annotated)
        fig.tight_layout()
        plt.show()

        
        
        