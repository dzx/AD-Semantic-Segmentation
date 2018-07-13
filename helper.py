import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow as tf
from glob import glob
import sys
#from urllib.request import urlretrieve
#from tqdm import tqdm
#from moviepy.editor import VideoFileClip
from sklearn.model_selection import train_test_split
import jpeg4py as jpeg
import cv2
from io import BytesIO
from PIL import Image


#class DLProgress(tqdm):
#    last_block = 0
#
#    def hook(self, block_num=1, block_size=1, total_size=None):
#        self.total = total_size
#        self.update((block_num - self.last_block) * block_size)
#        self.last_block = block_num
#
#
#def maybe_download_pretrained_vgg(data_dir):
#    """
#    Download and extract pretrained vgg model if it doesn't exist
#    :param data_dir: Directory to download the model to
#    """
#    vgg_filename = 'vgg.zip'
#    vgg_path = os.path.join(data_dir, 'vgg')
#    vgg_files = [
#        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
#        os.path.join(vgg_path, 'variables/variables.index'),
#        os.path.join(vgg_path, 'saved_model.pb')]
#
#    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
#    if missing_vgg_files:
#        # Clean vgg dir
#        if os.path.exists(vgg_path):
#            shutil.rmtree(vgg_path)
#        os.makedirs(vgg_path)
#
#        # Download vgg
#        print('Downloading pre-trained vgg model...')
#        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
#            urlretrieve(
#                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
#                os.path.join(vgg_path, vgg_filename),
#                pbar.hook)
#
#        # Extract vgg
#        print('Extracting model...')
#        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
#        zip_ref.extractall(data_dir)
#        zip_ref.close()
#
#        # Remove zip file to save space
#        os.remove(os.path.join(vgg_path, vgg_filename))

def gen_train_val_folds(data_folder, tst_size=None, seed=None):
    """
    Generate filenames for training and test(if specified) images and labels
    :param data_folder: Path to folder that contains all the datasets
    :param tst_size: fraction of dataset to be used for training validation
    :param seed: random seed for set splitting
    :return: image_paths, label_paths, indices or (train, validation) indices if splitting
    """
    img_files = os.listdir(os.path.join(data_folder, 'CameraRGB'))
    image_paths = [os.path.join(data_folder, 'CameraRGB', i) for i in img_files]
    label_paths = [os.path.join(data_folder, 'CameraSeg', i) for i in img_files]
    #image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
    #label_paths = glob(os.path.join(data_folder, 'CameraSeg', '*.png'))
    indices = [i for i in range(len(image_paths))]
    if tst_size != None:
        indices = train_test_split(indices, test_size=tst_size, random_state=seed)
    return image_paths, label_paths, indices

#MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2', 'gamma1.8', 'gamma2.3', 'bicubic0.5', 'bicubic0.8', 'bicubic1.5', 'bicubic2.0']
MANIPULATIONS = ['jpg70', 'jpg90', 'gamma0.8', 'gamma1.2']


def random_manipulation(img, manipulation=None):

    if manipulation == None:
        manipulation = random.choice(MANIPULATIONS)

    if manipulation.startswith('jpg'):
        quality = int(manipulation[3:])
        out = BytesIO()
        im = Image.fromarray(img)
        im.save(out, format='jpeg', quality=quality)
        im_decoded = jpeg.JPEG(np.frombuffer(out.getvalue(), dtype=np.uint8)).decode()
        del out
        del im
    elif manipulation.startswith('gamma'):
        gamma = float(manipulation[5:])
        # alternatively use skimage.exposure.adjust_gamma
        # img = skimage.exposure.adjust_gamma(img, gamma)
        im_decoded = np.uint8(cv2.pow(img / 255., gamma)*255.)
    elif manipulation.startswith('bicubic'):
        scale = float(manipulation[7:])
        im_decoded = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        cv2.resize()
    else:
        assert False
    return im_decoded

def gen_batch_function(image_paths, label_paths, indexes=None, crops=None, downsample=None):
    """
    Generate function to create batches of training data
    :param image_paths: Paths of all images
    :param label_paths: Paths of all labels
    :param indexes: indices of images/labels to include in batches or None for all
    :param crops: number of pixels to vertical crop as tuple(top, bottom)
    :return: get_batches_fn(batch_size) function
    """
#    print('Indexes',len(indexes))
    if indexes is None:
        indexes = [i for i in range(len(image_paths))]
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: image_batch[:batch_size], label_batch[:batch_size]
        """
#        background_color = np.array([255, 0, 0])
        random.shuffle(indexes)
        print("get_batches set size:", len(indexes))
        crop_t, crop_b = None, None
        if crops and len(crops):
            crop_t = crops[0]
            if len(crops) > 1: crop_b = - crops[1]
        for batch_i in range(0, len(indexes), batch_size):
            images = []
            gt_images = []
            for i in indexes[batch_i:batch_i+batch_size]:
                image_file = image_paths[i]
                gt_image_file = label_paths[i]

                image = scipy.misc.imread(image_file) #, image_shape
#                if (np.random.rand() < 0.5): 
#                    image = random_manipulation(image)
                #image = random_manipulation(image,'jpg70')
                gt_image = scipy.misc.imread(gt_image_file) #, image_shape)
                gt_image = gt_image[:,:,0]

                gt_road = np.any(np.stack((gt_image == 6, gt_image==7), axis=2), axis=2)
                gt_vehicles = gt_image == 10
                #print(np.sum(gt_vehicles), np.sum(gt_image))
                #tt
                gt_vehicles[496:] = False
                gt_objects = np.stack((gt_road, gt_vehicles), axis=2)
                gt_other = np.logical_not(np.any(gt_objects, axis=2))
                gt = np.stack((gt_road, gt_vehicles, gt_other), axis=2)
#                gt_bg = np.all(gt_image == background_color, axis=2)
#                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
#                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
                image = image[crop_t:crop_b]                
                gt = gt[crop_t:crop_b]
                
                                                 
                
                #scale = 0.0; dims = (240,600)
                if downsample != None:
                    dims = (0,0) #scale 0.75 (240x600) puca
                
                    image = cv2.resize(image,dims, fx=downsample, fy=downsample, interpolation = cv2.INTER_AREA)
                              
                    gt_road = cv2.resize(gt[:,:,0].astype('uint8'),dims, fx=downsample, fy=downsample, interpolation = cv2.INTER_AREA).astype(bool)
                    gt_vehicles = cv2.resize(gt[:,:,1].astype('uint8'),dims, fx=downsample, fy=downsample, interpolation = cv2.INTER_AREA).astype(bool)
                    gt_other = cv2.resize(gt[:,:,2].astype('uint8'),dims, fx=downsample, fy=downsample, interpolation = cv2.INTER_AREA).astype(bool)
                
                    gt = np.stack((gt_road, gt_vehicles, gt_other), axis=2)
                
                
                
                images.append(image) #64:576
                gt_images.append(gt) #64:576

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, image_paths, image_shape, crop):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in image_paths:
        image = scipy.misc.imread(image_file)
        street_im = anotate_image(sess, logits, keep_prob, image_pl, image, 
                                  image_shape, crop)

        yield os.path.basename(image_file), street_im
def anotate_image(sess, logits, keep_prob, image_pl, image, image_shape, crop):
    """
    Probably broken
    """
    mask_colors = [[0, 255, 0, 127],[255, 0, 0, 127]]    
    padding_t, padding_b, bottom = create_paddings(image_shape, crop)
    street_im = scipy.misc.toimage(image)
    segments = segment_image(sess, logits, keep_prob, image_pl, image[crop[0]:bottom], 
                             len(mask_colors))
    for i in range(len(segments)):
        segments[i] = pad_segment(segments[i], padding_t, padding_b)
    
    return mask_image(mask_colors, street_im, segments)
    
def mask_image(mask_colors, image, segments):
    for i in range(len(mask_colors)):
        mask = np.dot(np.expand_dims(segments[i], -1), np.array([mask_colors[i]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        image.paste(mask, box=None, mask=mask)
    return image
 
#import matplotlib.pyplot as plt
def segment_image(sess, logits, keep_prob, image_pl, image , num_segments):
    #image = image[:image_shape[0], :image_shape[1]]# scipy.misc.imresize(image, image_shape)
    image_shape = image.shape
    #print("segment image shape {}".format(image_shape))
    #plt.imshow(image)
    im_argmax = sess.run(
        [tf.argmax(logits, -1)],
        {keep_prob: 1.0, image_pl: [image]})
    #print(len(im_argmax), im_argmax, lgt)
    im_argmax = np.array(im_argmax).reshape(image_shape[0], image_shape[1])
    result = []
    for i in range(num_segments):
        segmentation = (im_argmax == i).astype('uint8') #.reshape(image_shape[0], image_shape[1])
        result.append(segmentation)
        #plt.imshow(segmentation)
    
    return result

def segment_images(sess, logits, keep_prob, image_pl, images , num_segments):#, options=None, run_metadata=None)
    #image = image[:image_shape[0], :image_shape[1]]# scipy.misc.imresize(image, image_shape)
    image_shape = images.shape
    #print("segment image shape {}".format(image_shape))
    #plt.imshow(image)
    start_time = time.perf_counter()
    im_argmax = sess.run(
        [logits], feed_dict={keep_prob: 1.0, image_pl: images}) #, options=options, run_metadata=run_metadata)
    print("Inference ran for {} seconds".format(time.perf_counter() - start_time), file=sys.stderr)
    #print(len(im_argmax), im_argmax, lgt)
    im_argmax = np.array(im_argmax).reshape(image_shape[0], image_shape[1], image_shape[2])
    result = []
    for i in range(num_segments):
        segmentation = (im_argmax == i).astype('uint8')
        result.append(segmentation)
        #plt.imshow(segmentation)
    
    return result

def create_paddings(image_shape, crop, stacks=None):
    padding_t, padding_b, bottom = None, None, None
    if crop[1]: 
        padding_b = np.zeros((crop[1], image_shape[1]), dtype='uint8')
        if stacks is not None:
            padding_b = np.array([padding_b for i in range(stacks)])
        bottom = -crop[1]
    if crop[0]: 
        padding_t = np.zeros((crop[0], image_shape[1]), dtype='uint8')
        if stacks is not None:
            padding_t = np.array([padding_t for i in range(stacks)])
    return padding_t, padding_b, bottom

def pad_segment(segment, padding_t, padding_b):
#    print(segment.shape, padding_t.shape)
    if padding_b is not None:
        if len(padding_b.shape)>2 and padding_b.shape[0] > segment.shape[0]:
            padding_b = padding_b[:segment.shape[0]]
        if padding_t is not None:
            if len(padding_t.shape)>2 and padding_t.shape[0] > segment.shape[0]:
                padding_t = padding_t[:segment.shape[0]]
            return np.concatenate((padding_t, segment, padding_b), axis=-2)
        else:
            return np.concatenate((segment, padding_b), axis=-2)
    elif padding_t is not None:
        if len(padding_t.shape)>2 and padding_t.shape[0] > segment.shape[0]:
            padding_t = padding_t[:segment.shape[0]]
        return np.concatenate((padding_t, segment), axis=-2)
    else:
        return segment

    

def save_inference_samples(runs_dir, image_paths, sess, image_shape, logits, keep_prob, input_image, crop):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image, 
                                    image_paths, image_shape, crop)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

#def process_video(data_dir, sess, image_shape, logits, keep_prob, input_image):
#    clip = VideoFileClip(os.path.join(data_dir,'driving.mp4'))
#    pipeline = lambda img: anotate_image(sess, logits, keep_prob, input_image,
#                                         img, image_shape)
#    new_clip = clip.fl_image(pipeline)
#    new_clip.write_videofile(os.path.join(data_dir, 'result.mp4'))
    
