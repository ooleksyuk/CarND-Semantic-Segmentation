import os.path
import scipy.misc
from scipy import ndimage
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
import matplotlib.pyplot as plt
plt.switch_backend('agg')


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


def img_size(image):
    return image.shape[0], image.shape[1]


def crop_image(image, gt_image):
    h, w = img_size(image)
    nw = random.randint(1150, w-5)  # Random crop size
    nh = int(nw / 3.3) # Keep original aspect ration
    x1 = random.randint(0, w - nw)  # Random position of crop
    y1 = random.randint(0, h - nh)
    return image[y1:(y1+nh), x1:(x1+nw), :], gt_image[y1:(y1+nh), x1:(x1+nw), :]


def flip_image(image, gt_image):
    return np.flip(image, axis=1), np.flip(gt_image, axis=1)


def bc_img(img, s=1.0, m=0.0):
    img = img.astype(np.int)
    img = img * s + m
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img


def process_gt_image(gt_image):
    background_color = np.array([255, 0, 0])

    gt_bg = np.all(gt_image == background_color, axis=2)
    gt_bg = gt_bg.reshape(gt_bg.shape[0], gt_bg.shape[1], 1)

    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
    return gt_image


def process_gt_city_images(gt_image):
    road_color = np.array([128, 64, 128, 255])
    car_color = np.array([0, 0, 142, 255])
    sign_color = np.array([220, 220, 0, 255])

    gt_road = np.all(gt_image == road_color, axis=2)
    gt_road = gt_road.reshape(gt_road.shape[0], gt_road.shape[1], 1)

    gt_car = np.all(gt_image == car_color, axis=2)
    gt_car = gt_car.reshape(gt_car.shape[0], gt_car.shape[1], 1)

    gt_sing = np.all(gt_image == sign_color, axis=2)
    gt_sing = gt_sing.reshape(gt_sing.shape[0], gt_sing.shape[1], 1)

    gt_obj = np.concatenate((gt_road, gt_car, gt_sing), axis=2)

    gt_bg = np.all(gt_obj == 0, axis=2)
    gt_bg.reshape(gt_bg.shape[0], gt_bg.shape[1], 1)

    # gt_image = np.concatenate((gt_bg, gt_obj), axis=2)

    return gt_image


def gen_batch_function_city(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """

    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        train_dataset_dir = os.path.join(data_folder, 'train_ds/')
        gt_dataset_dir = os.path.join(data_folder, 'gt_ds/')

        image_paths = os.listdir(gt_dataset_dir)
        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = os.path.join(gt_dataset_dir, image_file)

                image = scipy.misc.imread(os.path.join(train_dataset_dir, image_file))
                # image, gt_image = crop_image(image, gt_image)  # Random crop augmentation

                gt_image = scipy.misc.imread(gt_image_file)
                image2, gt_image2 = flip_image(image, gt_image)

                image = scipy.misc.imresize(image, image_shape)
                gt_image = scipy.misc.imresize(gt_image, image_shape)

                image2 = scipy.misc.imresize(image2, image_shape)
                gt_image2 = scipy.misc.imresize(gt_image2, image_shape)

                contr = random.uniform(0.85, 1.15)  # Contrast augmentation
                bright = random.randint(-40, 30)  # Brightness augmentation
                image = bc_img(image, contr, bright)

                gt_image = process_gt_city_images(gt_image)
                gt_image2 = process_gt_city_images(gt_image2)

                images.append(image)
                gt_images.append(gt_image)

                images.append(image2)
                gt_images.append(gt_image2)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn


def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}

        random.shuffle(image_paths)
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for image_file in image_paths[batch_i:batch_i + batch_size]:
                gt_image_file = label_paths[os.path.basename(image_file)]

                image = scipy.misc.imread(image_file)
                gt_image = scipy.misc.imread(gt_image_file)

                image2, gt_image2 = crop_image(image, gt_image)  # Random crop augmentation
                image3, gt_image3 = flip_image(image, gt_image)

                image = scipy.misc.imresize(image, image_shape)
                gt_image = scipy.misc.imresize(gt_image, image_shape)

                image2 = scipy.misc.imresize(image2, image_shape)
                gt_image2 = scipy.misc.imresize(gt_image2, image_shape)

                image3 = scipy.misc.imresize(image3, image_shape)
                gt_image3 = scipy.misc.imresize(gt_image3, image_shape)

                # image = cv2.imread(image_file)
                # gt_image = cv2.imread(gt_image_file)

                # image = cv2.resize(image, (image_shape[1], image_shape[0]))
                # gt_image = cv2.resize(gt_image, (image_shape[1], image_shape[0]))

                contrast = random.uniform(0.85, 1.15)  # Contrast augmentation
                bright = random.randint(-45, 30)  # Brightness augmentation
                image = bc_img(image, contrast, bright)

                gt_image = process_gt_image(gt_image)
                gt_image2 = process_gt_image(gt_image2)
                gt_image3 = process_gt_image(gt_image3)

                images.append(image)
                gt_images.append(gt_image)

                images.append(image2)
                gt_images.append(gt_image2)

                images.append(image3)
                gt_images.append(gt_image3)

            yield np.array(images), np.array(gt_images)

    return get_batches_fn


def denoise_img(img):
    eroded_img = ndimage.binary_erosion(img)
    return ndimage.binary_propagation(eroded_img, mask=img)


def paste_mask(street_im, im_soft_max, image_shape, color, obj_color_schema):
    im_soft_max_r = im_soft_max[0][:, color].reshape(image_shape[0], image_shape[1])
    segmentation_r = (im_soft_max_r > 0.5).reshape(image_shape[0], image_shape[1], 1)
    mask = np.dot(segmentation_r, np.array(obj_color_schema))
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im.paste(mask, box=None, mask=mask)

    return street_im


def gen_test_output_city(sess, logits, keep_prob, image_pl, data_folder, image_shape):
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

    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        street_im = scipy.misc.toimage(image)
        im_soft_max = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [image]})

        # Road
        street_im = paste_mask(street_im, im_soft_max, image_shape, 0, [[128, 64, 128, 64]])

        # Car
        street_im = paste_mask(street_im, im_soft_max, image_shape, 1, [[0, 0, 142, 64]])

        # Signs
        street_im = paste_mask(street_im, im_soft_max, image_shape, 3, [[220, 220, 0, 127]])

        yield os.path.basename(image_file), np.array(street_im)


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
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
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
        street_im = scipy.misc.toimage(image)

        im_softmax = sess.run([tf.nn.softmax(logits)], {keep_prob: 1.0, image_pl: [image]})

        street_im = paste_mask(street_im, im_softmax, image_shape, 1, [[0, 255, 0, 127]])

        yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape
    )
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)


def save_inference_samples_city(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output_city(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape
    )
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)


def plot_loss(runs_dir, loss, folder_name):
    _, axes = plt.subplots()
    plt.plot(range(0, len(loss)), loss)
    plt.title('Cross-entropy loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
    os.makedirs(runs_dir)

    output_file = os.path.join(runs_dir, folder_name + ".png")
    plt.savefig(output_file)
