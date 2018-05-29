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

KP, LRN_RATE, iou_op, iou_val = None, None, None, None


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    graph = tf.get_default_graph()
    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
#tests.test_load_vgg(load_vgg, tf)
#%%

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    l7_conv = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same', #activation=tf.nn.relu,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=.01))
    l7_conv_trn = tf.layers.conv2d_transpose(l7_conv, num_classes, 4, 2, padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=.01))
    l4_conv = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same', #activation=tf.nn.relu,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=.01))
    skip1 = tf.add(l7_conv_trn, l4_conv)
    l3_conv = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same', #activation=tf.nn.relu,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=.01))
    skip1_trn = tf.layers.conv2d_transpose(skip1, num_classes, 4, 2, padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=.01))
    skip2 = tf.add(l3_conv, skip1_trn)
    out = tf.layers.conv2d_transpose(skip2, num_classes, 16, 8, padding='same',
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3),
                                kernel_initializer=tf.truncated_normal_initializer(stddev=.01))
       
    return out
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes), name="logits")
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_loss = sum(reg_losses)
    train_op = optimizer.minimize(cross_entropy_loss + reg_loss)
    
    return logits, train_op, cross_entropy_loss#, reg_loss
#tests.test_optimize(optimize)
    
def metrics(nn_last_layer, correct_label, num_classes):
    lbls=tf.argmax(correct_label, axis=-1)
    preds=tf.argmax(nn_last_layer, axis=-1)
    iou_val, iou_op = tf.metrics.mean_iou(labels=lbls, predictions=preds, 
                                          num_classes=num_classes, name="iou_bloc")
    rcls = [None] * (num_classes-1)
    rcl_ops = [None] * (num_classes-1)
    precs = [None] * (num_classes-1)
    prec_ops = [None] * (num_classes-1)
    for i in range(num_classes-1):
        rcls[i], rcl_ops[i] = tf.metrics.recall(labels=tf.equal(lbls, i), 
                                                  predictions=tf.equal(preds, i), name="iou_bloc")
        precs[i], prec_ops[i] = tf.metrics.precision(labels=tf.equal(lbls, i), 
                                                  predictions=tf.equal(preds, i), name="iou_bloc")        
    return iou_val, iou_op, rcls, rcl_ops, precs, prec_ops

def f_beta(precision, recall, beta=1.):
    beta_s = beta**2
    return (1+beta_s) * precision * recall / (beta_s * precision + recall)
    
#%%
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, iou_value, iou_oper, 
             rcls, rcl_ops, precs, prec_ops, data_dir, val_batch_fn=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    global KP, LRN_RATE
    running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="iou_bloc")

    # Define initializer to initialize/reset running variables
    running_vars_initializer = tf.variables_initializer(var_list=running_vars)
    sess.run(tf.global_variables_initializer())
    t_bcount, v_bcount = None, None
    for ep in range(epochs):
        sess.run(running_vars_initializer)
        t_batch = 0
        xloss = 0
        for imgs, labels in get_batches_fn(batch_size):
            sess.run(train_op, feed_dict={input_image:imgs, correct_label:labels, keep_prob:KP,
                                          learning_rate:LRN_RATE})
            xloss += sess.run(cross_entropy_loss, feed_dict={input_image:imgs, correct_label:labels, keep_prob:1.,
                                          learning_rate:0.})
            if iou_oper is not None:
                sess.run([iou_oper] + rcl_ops + prec_ops, feed_dict={input_image:imgs, correct_label:labels, keep_prob:1.,
                                          learning_rate:0.})
            if t_batch % 10 == 0:
                print("Epoch:{} batch {}/{} xent loss:{}".format(ep, t_batch, t_bcount, xloss/10))
                xloss = 0
            t_batch += 1
        t_bcount = t_batch
        if iou_value is not None:
            ep_iou, rcl_r, rcl_c, prec_r, prec_c = sess.run([iou_value] + rcls + precs)
            f_cars = f_beta(prec_c, rcl_c, 2.)
            f_road = f_beta(prec_r, rcl_r, .5)
            print("Train for epoch {}: IOU {} F-c {} F-r {} F-avg {}".format(ep, 
                  ep_iou, f_cars, f_road, (f_cars+f_road)/2))
        if val_batch_fn:
            sess.run(running_vars_initializer)
            v_batch = 0
            xloss = 0
            for imgs, labels in val_batch_fn(batch_size):
                xloss += sess.run(cross_entropy_loss, feed_dict={input_image:imgs, correct_label:labels, keep_prob:1.,
                                          learning_rate:0.})
                if iou_oper is not None:
                    sess.run([iou_oper] + rcl_ops + prec_ops, feed_dict={input_image:imgs, correct_label:labels, keep_prob:1.,
                                          learning_rate:0.})
                if v_batch % 10 == 0:
                    print("Epoch:{} validation batch {}/{} xent loss:{}".format(ep, v_batch, v_bcount, xloss/10))
                    xloss = 0
                v_batch += 1
            if iou_value is not None:
                vep_iou, rcl_r, rcl_c, prec_r, prec_c = sess.run([iou_value] + rcls + precs)
                vf_cars = f_beta(prec_c, rcl_c, 2.)
                vf_road = f_beta(prec_r, rcl_r, .5)
                
                print("Epoch {}: Training IOU {} F-c {} F-r {} F-avg {}, Validation IOU {} F-c {} F-r {} F-avg {}".format(
                        ep, ep_iou, f_cars, f_road, (f_cars+f_road)/2, vep_iou, 
                        vf_cars, vf_road, (vf_cars+vf_road)/2))
        save_epoch(sess, data_dir, ep)
        
#tests.test_train_nn(train_nn)
def save_epoch(sess, data_dir, epoch):
    print('Saving epoch {}'.format(epoch))
    builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(data_dir, 'models', str(epoch)))
    builder.add_meta_graph_and_variables(sess, ['VGG_Trained'])
    builder.save()
        


#%%
def run():
    num_classes = 3
    #image_shape = (160, 576)
    crops = (206, 74) # trim 206 pixels from top and 74 from bottom
    data_dir = './'
    runs_dir = './runs'
    log_dir = './tf_log'
#    tests.test_for_kitti_dataset(data_dir)
    num_epochs = 14 #25
    batch_size = 5 #3 #8
    global KP, LRN_RATE
    KP = .5
    LRN_RATE = 1e-4

    # Download pretrained vgg model
    #helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        images, masks, [train_idxs, val_idxs] = helper.gen_train_val_folds(os.path.join(data_dir, 'Train'), .1, 42)
        get_batches_fn = helper.gen_batch_function(images, masks, train_idxs, crops)
        test_batches_fn = helper.gen_batch_function(images, masks, val_idxs, crops)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        final_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        labels = tf.placeholder(tf.int32, (None, None, None, num_classes), name='labels')
        learning_rate = tf.placeholder(tf.float32)
        logits, train_op, cross_entropy_loss = optimize(final_layer, labels, learning_rate, num_classes)
        iou_val, iou_op, rcls, rcl_ops, precs, prec_ops = metrics(final_layer, labels, num_classes)
        print("TRAINING")
        writter = tf.summary.FileWriter(log_dir, sess.graph)

        train_nn(sess, num_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, 
                 input_image, labels, keep_prob, learning_rate, iou_val, iou_op, 
                 rcls, rcl_ops, precs, prec_ops, data_dir, test_batches_fn)
        writter.close()
        
#        builder = tf.saved_model.builder.SavedModelBuilder(os.path.join(data_dir, 'trained'))
#        builder.add_meta_graph_and_variables(sess, ['VGG_Trained'])
#        builder.save()
#        print("Trained model saved")
        # Save inference data using helper.save_inference_samples
       # helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

