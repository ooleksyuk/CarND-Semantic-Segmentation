import tensorflow as tf
import numpy as np
import argparse
import sys
import time

from scipy.misc import imread, imresize
from glob import glob

FLAGS = None

def load_graph(graph_file, use_xla=False):
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        # ops = sess.graph.get_operations()
        # n_ops = len(ops)
        return sess #, ops

# images are (375, 1242, 3)
# need to reshape to (380, 1242, 3) or else we get off by 1 for some operations
def load_imgs():
    files = glob('./data/data_road/testing/image_2/um_*.png')
    imgs = []
    for f in files:
        imgs.append(imresize(imread(f), (380, 1242, 3)).reshape((1, 380, 1242, 3)))
    return imgs

def benchmark(sess, imgs, runs=5, binary=True):
    g = sess.graph
    x = g.get_tensor_by_name('image_input:0')
    keep_prob = g.get_tensor_by_name('keep_prob:0')
    out = g.get_tensor_by_name('adam_logit:0')

    # dummy run
    sess.run(out, {x: imgs[0], keep_prob: 1.0})

    times = []
    for i in range(runs):
        t0 = time.time()
        for img in imgs:
            t1 = time.time()
            sess.run(out, {x: img, keep_prob: 1.0})
            print(time.time() - t1)
        duration = time.time() - t0
        times.append(duration)
        print("Run {0} complete, time taken = {1}".format(i, duration))
    print("Mean = {0}, Stddev = {1}, Min = {2}, Max = {3}".format(np.mean(times), np.std(times), np.min(times), np.max(times)))
    return times

def main(_):
    imgs = load_imgs()
    print("XLA {0}".format(FLAGS.use_xla))
    sess, ops = load_graph(FLAGS.graph)
    times = benchmark(sess, imgs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--graph', type=str, help='Graph file (.pb)')
    parser.add_argument( '--use_xla', type=bool, help='Whether to use XLA JIT')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)