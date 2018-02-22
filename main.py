import os.path
import tensorflow as tf
import helper
import warnings
import time
from datetime import timedelta
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


L2_REG = 1e-5
STDEV = 1e-2
KEEP_PROB = 0.8
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 8
IMAGE_SHAPE = (160, 576)
NUM_CLASSES = 2

DATA_DIR = './data'
RUNS_DIR = './runs'
MODEL_DIR = './models_3col'


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    graph = tf.get_default_graph()
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)

    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return input_image, keep_prob, layer3, layer4, layer7


tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    conv_layer_7_1x1 = tf.layers.conv2d(
        inputs=vgg_layer7_out,
        filters=num_classes,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG)
    )
    deconv_layer_1_output = tf.layers.conv2d_transpose(
        inputs=conv_layer_7_1x1,
        filters=num_classes,
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG)
    )
    conv_layer_4_1x1 = tf.layers.conv2d(
        inputs=vgg_layer4_out,
        filters=num_classes,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG)
    )
    skip_connection_1 = tf.add(deconv_layer_1_output, conv_layer_4_1x1)
    deconv_layer_2 = tf.layers.conv2d_transpose(
        inputs=skip_connection_1,
        filters=num_classes,
        kernel_size=4,
        strides=2,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG)
    )
    layer3_conv_1x1 = tf.layers.conv2d(
        inputs=vgg_layer3_out,
        filters=num_classes,
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG)
    )
    skip_connection_2 = tf.add(deconv_layer_2, layer3_conv_1x1)
    output_conv_layer = tf.layers.conv2d_transpose(
        inputs=skip_connection_2,
        filters=num_classes,
        kernel_size=16,
        strides=8,
        padding='same',
        kernel_initializer=tf.random_normal_initializer(stddev=STDEV),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_REG)
    )

    return output_conv_layer


tests.test_layers(layers)


# Loss with weights
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

    cross_entropy_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    )
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cross_entropy_loss = cross_entropy_loss + sum(reg_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss


tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    losses = []
    for epoch in range(epochs):
        loss = None
        s_time = time.time()
        for image, labels in get_batches_fn(batch_size):
            _, loss = sess.run(
                [train_op, cross_entropy_loss],
                feed_dict={input_image: image,
                           correct_label: labels,
                           keep_prob: KEEP_PROB,
                           learning_rate: LEARNING_RATE}
            )
            losses.append(loss)
            # if epoch % 10 == 0:
            #     saver.save(sess, os.path.join(DATA_DIR, 'checkpoints/cont_epoch_' + str(epoch) + '.ckpt'))
        print("[Epoch: {0}/{1} Loss: {2:4f} Time: {3}]".format(epoch + 1, epochs, loss, str(timedelta(seconds=(time.time() - s_time)))))
    helper.plot_loss(RUNS_DIR, losses, "loss_graph")


tests.test_train_nn(train_nn)


def run():
    tests.test_for_kitti_dataset(DATA_DIR)
    # Download pre trained vgg model
    helper.maybe_download_pretrained_vgg(DATA_DIR)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    # https://www.cityscapes-dataset.com/

    print("Start training...")
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        # Path to vgg model
        vgg_path = os.path.join(DATA_DIR, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(DATA_DIR, 'data_road/training'), IMAGE_SHAPE)
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network
        # Add some augmentations, see helper.py
        input_image, keep_prob, layer3, layer4, layer7 = load_vgg(sess, vgg_path)
        output_layer = layers(layer3, layer4, layer7, NUM_CLASSES)

        correct_label = tf.placeholder(dtype=tf.float32, shape=(None, None, None, NUM_CLASSES))
        learning_rate = tf.placeholder(dtype=tf.float32)

        logits, train_op, cross_entropy_loss = optimize(output_layer, correct_label, learning_rate, NUM_CLASSES)

        sess.run(tf.global_variables_initializer())

        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(RUNS_DIR, DATA_DIR, sess, IMAGE_SHAPE, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
