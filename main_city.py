import os.path
import tensorflow as tf
import helper
import helper_cityscapes
import warnings
import math
from distutils.version import LooseVersion
import project_tests as tests

from timeit import default_timer as timer
from tqdm import tqdm
import scipy.misc
import numpy as np

import argparse

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
L2_REG = 1e-5
STDEV = 1e-2
KEEP_PROB = 0.8
LEARNING_RATE = 1e-4
EPOCHS_KITI = 20
BATCH_SIZE_KITI = 8
IMAGE_SHAPE_KITI = (160, 576)
NUM_CLASSES_KITI = 2

DATA_DIR = './data'
RUNS_DIR = './runs'

EPOCHS_CITY = 40
BATCH_SIZE_CITY = 8
IMAGE_SHAPE_CITY = (256, 512)
NUM_CLASSES_CITY = 4


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


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    init = tf.truncated_normal_initializer(stddev=STDEV)
    reg = tf.contrib.layers.l2_regularizer(L2_REG)

    # as per fcn8s reference caffe implementation: but does not help at all ...
    # vgg_layer3_out_scaled = tf.multiply(vgg_layer3_out, 0.0001)
    # vgg_layer4_out_scaled = tf.multiply(vgg_layer4_out, 0.01)

    # reduce dimensions with conv1x1 filters
    conv1x1_l3 = tf.layers.conv2d(
        vgg_layer3_out,
        num_classes,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )
    conv1x1_l4 = tf.layers.conv2d(
        vgg_layer4_out,
        num_classes,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )
    conv1x1 = tf.layers.conv2d(
        vgg_layer7_out,
        num_classes,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )

    # upsample output of encoder by 2
    deconv_1 = tf.layers.conv2d_transpose(
        conv1x1,
        num_classes,
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )
    # add skip connection from layer 4
    deconv_1 = tf.add(deconv_1, conv1x1_l4)
    # upsample by 2
    deconv_2 = tf.layers.conv2d_transpose(
        deconv_1,
        num_classes,
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )
    # add skip connection from layer 3
    deconv_2 = tf.add(deconv_2, conv1x1_l3)
    # upsample by 8: so we are back to the original image size (that was downsampled by 32 in encoder)
    deconv_3 = tf.layers.conv2d_transpose(
        deconv_2,
        num_classes,
        kernel_size=16,
        strides=8,
        padding='same',
        kernel_initializer=init,
        kernel_regularizer=reg
    )

    # tf.Print(deconv_3, [tf.shape](deconv_3)])

    return deconv_3


def build_predictor(nn_last_layer):
    softmax_output = tf.nn.softmax(nn_last_layer)
    predictions_argmax = tf.argmax(softmax_output, axis=-1) #, output_type=tf.int64)
    return softmax_output, predictions_argmax


def build_metrics(correct_label, predictions_argmax, num_classes):
    labels_argmax = tf.argmax(correct_label, axis=-1) #, output_type=tf.int64)
    iou, iou_op = tf.metrics.mean_iou(labels_argmax, predictions_argmax, num_classes)
    return iou, iou_op


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    # # l2_reg does not help here apparently  ...
    # regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) # Scalar
    # cross_entropy_loss = cross_entropy_loss + regularization_loss

    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.99)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0)

    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


def train_nn(sess, epochs, batch_size, get_train_batches_fn, get_valid_batches_fn, train_op, cross_entropy_loss,
             input_image, correct_label, keep_prob, learning_rate, iou, iou_op, saver, n_train, n_valid, lr,
             early_stop, patience):
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
    print("Start training with lr {} ...".format(lr))
    fail = 0
    best_iou = 0
    for epoch in range(epochs):
        generator = get_train_batches_fn(batch_size)
        description = 'Train Epoch {:>2}/{}'.format(epoch + 1, epochs)
        start = timer()
        losses = []
        ious = []
        for image, label in tqdm(generator, total=n_train, desc=description, unit='batches'):
            _, loss, _ = sess.run([train_op, cross_entropy_loss, iou_op],
                                  feed_dict={input_image: image, correct_label: label,
                                             keep_prob: 0.8, learning_rate: lr})
            # print(loss)
            losses.append(loss)
            ious.append(sess.run(iou))
        end = timer()
        helper.plot_loss(RUNS_DIR, losses, "loss_graph_training")
        print("EPOCH {} with lr {} ...".format(epoch + 1, lr))
        print("  time {} ...".format(end - start))
        print("  Train Xentloss = {:.4f}".format(sum(losses) / len(losses)))
        print("  Train IOU = {:.4f}".format(sum(ious) / len(ious)))

        generator = get_valid_batches_fn(batch_size)
        description = 'Valid Epoch {:>2}/{}'.format(epoch + 1, epochs)
        losses = []
        ious = []
        for image, label in tqdm(generator, total=n_valid, desc=description, unit='batches'):
            loss, _ = sess.run([cross_entropy_loss, iou_op],
                               feed_dict={input_image: image, correct_label: label, keep_prob: 1})
            losses.append(loss)
            ious.append(sess.run(iou))
        helper.plot_loss(RUNS_DIR, losses, "loss_graph_validation")
        print("  Valid Xentloss = {:.4f}".format(sum(losses) / len(losses)))
        valid_iou = sum(ious) / len(ious)
        print("  Valid IOU = {:.4f}".format(valid_iou))

        if (valid_iou > best_iou):
            saver.save(sess, './models/fcn8s')
            saver.save(sess, './models/fcn8s.ckpt')
            with open("models/training.txt", "w") as text_file:
                text_file.write("models/fcn8s: epoch {}, lr {}, valid_iou {}".format(epoch + 1, lr, valid_iou))
            print("  model saved")
            best_iou = valid_iou
            fail = 0
        else:
            lr *= 0.5  # lr scheduling: halving on failure
            print("  no improvement => lr downscaled to {} ...".format(lr))
            fail += 1

        if (early_stop and fail >= patience):
            break


# get mean IOU over a reference test set
def test_nn(sess, batch_size, get_test_batches_fn, predictions_argmax, input_image, correct_label, keep_prob, iou,
            iou_op, n_batches):
    generator = get_test_batches_fn(batch_size)
    ious = []
    for image, label in tqdm(generator, total=n_batches, unit='batches'):
        labels, _ = sess.run([predictions_argmax, iou_op],
                             feed_dict={input_image: image, correct_label: label, keep_prob: 1})
        ious.append(sess.run(iou))
    print("Test IOU = {:.4f}".format(sum(ious) / len(ious)))


# process 1 image
# 62 ms for inference + 23 ms for image composition
def predict_nn(sess, test_image, predictions_argmax, input_image, keep_prob, image_shape, label_colors):
    start = timer()
    image = scipy.misc.imresize(test_image, image_shape)

    labels = sess.run([predictions_argmax], feed_dict={input_image: [image], keep_prob: 1})
    labels = labels[0].reshape(image_shape[0], image_shape[1])

    # create an overlay
    labels_colored = np.zeros((image_shape[0], image_shape[1], 4))  # 4 for RGBA
    for label in label_colors:
        label_mask = labels == label
        labels_colored[label_mask] = np.array((*label_colors[label], 127))

    mask = scipy.misc.toimage(labels_colored, mode="RGBA")
    pred_image = scipy.misc.toimage(image)
    pred_image.paste(mask, box=None, mask=mask)
    end = timer()
    print("predict time {}".format(end - start))
    return pred_image


def run():
    parser = argparse.ArgumentParser(description='Train FCN8s')
    parser.add_argument('--resume', type=bool, default=False, help='resume training')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')

    parser.add_argument('--epochs', type=int, default=6, help='epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='batch size')
    parser.add_argument('--early-stop', type=bool, default=True, help='early stop')
    parser.add_argument('--patience', type=int, default=3, help='epochs')
    args = parser.parse_args()

    resume = args.resume
    lr = args.lr

    epochs = args.epochs
    batch_size = args.batch_size
    early_stop = args.early_stop
    patience = args.patience

    # Download pre trained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    runs_dir = './runs_city'
    city_data_dir = './data'
    train_images, valid_images, test_images, num_classes, label_colors, image_shape = helper_cityscapes.load_data(city_data_dir)
    print("len: train_images {}, valid_images {}, test_images {}".format(len(train_images), len(valid_images), len(test_images)))

    # Create function to get batches
    get_train_batches_fn = helper_cityscapes.gen_batch_function(train_images, image_shape)
    get_valid_batches_fn = helper_cityscapes.gen_batch_function(valid_images, image_shape)
    get_test_batches_fn = helper_cityscapes.gen_batch_function(test_images, image_shape)

    correct_label = tf.placeholder(tf.float32, (None, image_shape[0], image_shape[1], num_classes))
    learning_rate = tf.placeholder(tf.float32, [])

    # # compile tensorflow from sources
    # does not help ...
    # # Create a TensorFlow configuration object. This will be
    # # passed as an argument to the session.
    # config = tf.ConfigProto()
    # # JIT level, this can be set to ON_1 or ON_2
    # jit_level = tf.OptimizerOptions.ON_2
    # config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        input_image, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        fcn8s_output = layers(layer3_out, layer4_out, layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(fcn8s_output, correct_label, learning_rate, num_classes)

        softmax_output, predictions_argmax = build_predictor(fcn8s_output)
        iou, iou_op = build_metrics(correct_label, predictions_argmax, num_classes)

        # WARNING run those initializer _BEFORE_ restore
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()

        if resume:
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            print("resume")

        n_train = int(math.ceil(len(train_images) / batch_size))
        n_valid = int(math.ceil(len(valid_images) / batch_size))
        train_nn(sess, epochs, batch_size, get_train_batches_fn, get_valid_batches_fn,
                 train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate, iou, iou_op, saver,
                 n_train, n_valid, lr, early_stop, patience)

        test_image = scipy.misc.imread("test_image.png")
        pred_image = predict_nn(sess, test_image, predictions_argmax, input_image, keep_prob, image_shape, label_colors)
        scipy.misc.imsave("pred_image.png", pred_image)

        n_batches = int(math.ceil(len(test_images) / batch_size))
        # batch_size 32 is ok (and faster) with GTX 1080 TI and 11 GB memory
        test_nn(sess, 32, get_test_batches_fn, predictions_argmax, input_image,
                correct_label, keep_prob, iou, iou_op, n_batches)

        if resume:
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            print("resume")

        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        helper_cityscapes.save_inference_samples(runs_dir, test_images, sess, image_shape, logits, keep_prob,
                                                 input_image, label_colors)

        output_node_names = 'Softmax'
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,  # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(),  # The graph_def is used to retrieve the nodes
            output_node_names.split(",")  # The output node names are used to select the usefull nodes
        )

        saver.save(sess, './runs/fcn8s.ckpt')
        tf.train.write_graph(tf.get_default_graph().as_graph_def(), '', './runs/base_graph.pb', False)
        tf.train.write_graph(output_graph_def, '', './runs/frozen_graph.pb', False)
        print("%d ops in the final graph." % len(output_graph_def.node))

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()