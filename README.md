# Semantic Segmentation
### Introduction
In this project, you'll label the pixels of a road in images using a Fully Convolutional Network (FCN).

### Reflections. Rubric
#### Does the project load the pre-trained vgg model?
The function `load_vgg` loads `vgg` model.
#### Does the project learn the correct features from the images?
The project has `layers` function implemented.
This image visualizes the Original Skip Layer Architecture of the network:
![Original Skip Layer Architecture of the network](./3-Figure3-1.png)
#### Does the project optimize the neural network?
The project has `optimize` function implemented.
#### Does the project train the neural network?
The `train_nn` function is implemented and prints time and loss per epoch/epochs of training.
#### Does the project train the model correctly?
The project trains model correctly, about 48s per epoch, 48sx20 epochs in total.
#### Does the project use reasonable hyper parameters?
I have trained model many times to figure out reasonable set of params.
For KITI data set I used:
```python
L2_REG = 1e-5
STDEV = 1e-2
KEEP_PROB = 0.8
LEARNING_RATE = 1e-4
EPOCHS = 20
BATCH_SIZE = 8
IMAGE_SHAPE = (160, 576)
NUM_CLASSES = 2
```
For CityScapes dataset I used:
```python
L2_REG = 1e-5
STDEV = 1e-2
KEEP_PROB = 0.8
LEARNING_RATE = 5e-4
EPOCHS = 20
BATCH_SIZE = 8
IMAGE_SHAPE = (256, 512)
```
for CityScapes data set number of classes defined based on a list of classes provided by the set provider.
#### Does the project correctly label the road?
Yes, I've tested the project on images from the dataset and here is the result:
![KITI data set](./kiti-dataset-full.gif)
KITI data set
![CityScapes data set](./city-dataset.gif)
CityScapes data set

I was tracking the loss during the training and here is a graph that describes my results:
![Cross-entropy loss for KITI data set](./loss_graph_kiti.png)
Cross-entropy loss for KITI data set
![Cross-entropy loss for CityScapes data set](./loss_graph_city.png)
Cross-entropy loss for CityScapes data set

#### Reflections
To improve road recognition I've added image pre processing. In `helper.py` in `def gen_batch_function` I added image crop, image flip and changes to brightness and contrast of the image.
To increase an image data set, for one given image I've produced three additional once, cropped, flipped and with changed brightness and contrast, for each image I've kept the ground truth to be consistent. If the image was cropped, I've cropped ground truth image as well, if image was flipped, I flip the ground truth images as well.

This allowed me to add more variety to data set of images and improve road recognition in difficult places like shadows, bikes, sidewalks, road separators.

Skip connections are found to improve the segmentation accuracy, as discussed by the authors in the original [paper](./papers/1411.4038.pdf).

I used these resources to learn more about [weight initialization](http://cs231n.github.io/neural-networks-2/#init) and [regularization](http://cs231n.github.io/neural-networks-2/#reg).
To study more about Semantic Segmentation, I am going to use this resource - [A 2017 Guide to Semantic Segmentation with Deep Learning](http://blog.qure.ai/notes/semantic-segmentation-deep-learning-review).

I have process ~20 labels from [cityscapes data](https://www.cityscapes-dataset.com/), code is located in `main-city.py` and `helper-cityscapes.py`.
Code can be run `python3 main_city.py --epoch 20 --batch-size 16`. Code build using Python3.

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
##### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the data set in the `data` folder.  This will create the folder `data_road` with all the training a test images.
Download the [CityScapes dataset](https://www.cityscapes-dataset.com/) from [here](https://www.cityscapes-dataset.com/)

### Start
##### Implement
Implement the code in the `main.py` module indicated by the "TODO" comments.
The comments indicated with "OPTIONAL" tag are not required to complete.
##### Run
Run the following command to run the project:
```
python main.py
```
**Note** If running this in Jupyter Notebook system messages, such as those regarding test status, may appear in the terminal rather than the notebook.

### Submission
1. Ensure you've passed all the unit tests.
2. Ensure you pass all points on [the rubric](https://review.udacity.com/#!/rubrics/989/view).
3. Submit the following in a zip file.
 - `helper.py`
 - `main.py`
 - `project_tests.py`
 - Newest inference images from `runs` folder  (**all images from the most recent run**)
 
 ### Tips
- The link for the frozen `VGG16` model is hardcoded into `helper.py`.  The model can be found [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip)
- The model is not vanilla `VGG16`, but a fully convolutional version, which already contains the 1x1 convolutions to replace the fully connected layers. Please see this [forum post](https://discussions.udacity.com/t/here-is-some-advice-and-clarifications-about-the-semantic-segmentation-project/403100/8?u=subodh.malgonde) for more information.  A summary of additional points, follow. 
- The original FCN-8s was trained in stages. The authors later uploaded a version that was trained all at once to their GitHub repo.  The version in the GitHub repo has one important difference: The outputs of pooling layers 3 and 4 are scaled before they are fed into the 1x1 convolutions.  As a result, some students have found that the model learns much better with the scaling layers included. The model may not converge substantially faster, but may reach a higher IoU and accuracy. 
- When adding l2-regularization, setting a regularizer in the arguments of the `tf.layers` is not enough. Regularization loss terms must be manually added to your loss function. otherwise regularization is not implemented.
 
### Using GitHub and Creating Effective READMEs
If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about REAMDE files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/courses/ud777), as well. 

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.
