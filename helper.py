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
from urllib.request import urlretrieve
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from sklearn.model_selection import train_test_split


class DLProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
    """
    Download and extract pretrained vgg model if it doesn't exist
    :param data_dir: Directory to download the model to
    """
    vgg_filename = 'vgg.zip'
    vgg_path = os.path.join(data_dir, 'vgg')
    vgg_files = [
        os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
        os.path.join(vgg_path, 'variables/variables.index'),
        os.path.join(vgg_path, 'saved_model.pb')]

    missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
    if missing_vgg_files:
        # Clean vgg dir
        if os.path.exists(vgg_path):
            shutil.rmtree(vgg_path)
        os.makedirs(vgg_path)

        # Download vgg
        print('Downloading pre-trained vgg model...')
        with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
            urlretrieve(
                'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
                os.path.join(vgg_path, vgg_filename),
                pbar.hook)

        # Extract vgg
        print('Extracting model...')
        zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
        zip_ref.extractall(data_dir)
        zip_ref.close()

        # Remove zip file to save space
        os.remove(os.path.join(vgg_path, vgg_filename))

def gen_train_val_folds(data_folder, tst_size=None, seed=None):
    image_paths = glob(os.path.join(data_folder, 'CameraRGB', '*.png'))
    label_paths = glob(os.path.join(data_folder, 'CameraSeg', '*.png'))
    indices = [i for i in range(len(image_paths))]
    if tst_size != None:
        indices = train_test_split(indices, test_size=tst_size, random_state=seed)
    return image_paths, label_paths, indices


def gen_batch_function(image_paths, label_paths, indexes=None):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :param indexes:
    :return:
    """
    print('Indexes',len(indexes))
    if indexes is None:
        indexes = [i for i in range(len(image_paths))]
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
#        background_color = np.array([255, 0, 0])
        random.shuffle(indexes)
        print("get_batches set size:", len(indexes))
        for batch_i in range(0, len(indexes), batch_size):
            images = []
            gt_images = []
            for i in indexes[batch_i:batch_i+batch_size]:
                image_file = image_paths[i]
                gt_image_file = label_paths[i]

                image = scipy.misc.imread(image_file) #, image_shape
                gt_image = scipy.misc.imread(gt_image_file) #, image_shape)
                gt_image = gt_image[:,:,0]

                gt_road = np.any(np.stack((gt_image == 6, gt_image==7), axis=2), axis=2)
                gt_vehicles = gt_image == 10
                gt_vehicles[496:] = False
                gt_objects = np.stack((gt_road, gt_vehicles), axis=2)
                gt_other = np.logical_not(np.any(gt_objects, axis=2))
                gt = np.stack((gt_road, gt_vehicles, gt_other), axis=2)
#                gt_bg = np.all(gt_image == background_color, axis=2)
#                gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
#                gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                images.append(image[:576])
                gt_images.append(gt[:576])

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, image_paths, image_shape):
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
        street_im = segment_image(sess, logits, keep_prob, image_pl, image, 
                                  image_shape)

        yield os.path.basename(image_file), street_im
import matplotlib.pyplot as plt
def segment_image(sess, logits, keep_prob, image_pl, image, image_shape):
    image = image[:image_shape[0], :image_shape[1]]# scipy.misc.imresize(image, image_shape)
    #plt.imshow(image)
    im_argmax = sess.run(
        [tf.argmax(logits, -1)],
        {keep_prob: 1.0, image_pl: [image]})
    #print(len(im_argmax), im_argmax, lgt)
    im_argmax = np.array(im_argmax).reshape(image_shape[0], image_shape[1])
    mask_colors = [[0, 255, 0, 127],[255, 0, 0, 127]]
    street_im = scipy.misc.toimage(image)
    for i in range(len(mask_colors)):
        segmentation = (im_argmax == i).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([mask_colors[i]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im.paste(mask, box=None, mask=mask)
    
    return np.array(street_im)
    

def save_inference_samples(runs_dir, image_paths, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(sess, logits, keep_prob, input_image, 
                                    image_paths, image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)

def process_video(data_dir, sess, image_shape, logits, keep_prob, input_image):
    clip = VideoFileClip(os.path.join(data_dir,'driving.mp4'))
    pipeline = lambda img: segment_image(sess, logits, keep_prob, input_image,
                                         img, image_shape)
    new_clip = clip.fl_image(pipeline)
    new_clip.write_videofile(os.path.join(data_dir, 'result.mp4'))
    
