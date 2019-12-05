import os
import json
from glob import glob
import numpy as np
import seaborn as sns
import scipy.io as sio
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
from IPython.display import Image

filenames = sorted(glob('C:\\Users\\ajink\\Downloads\\ML\\Images\\*.jpg'))[:]
representations = []

with gfile.GFile('classify_image_graph_def.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

def forward_pass(fname, target_layer = 'inception/pool_3:0'):
    g = tf.Graph()
    image_data = tf.gfile.FastGFile(fname, 'rb').read()
    with tf.Session(graph = g) as sess:
        tf.import_graph_def(graph_def, name = 'inception')
        pool3 = sess.graph.get_tensor_by_name(target_layer)
        pool3 = sess.run(pool3, {'inception/DecodeJpeg/contents:0': image_data})
        return pool3.flatten()

for fname in filenames:
    print(fname)
    frame_repr = forward_pass(fname)
    representations.append(frame_repr.flatten())

fig, (ax1, ax2) = plt.subplots(ncols = 2)
default_heatmap_kwargs = dict(xticklabels = False, yticklabels = False, square = True, cbar = False)

GROUND_TRUTH_PATH = os.path.expanduser('C:\\Users\\ajink\\Downloads\\ML\\kitti05GroundTruth.mat')
gt_data = sio.loadmat(GROUND_TRUTH_PATH)['truth'][:]
sns.heatmap(gt_data, ax = ax1, **default_heatmap_kwargs)
ax1.set_title('Ground Truth')

def normalize(x):
    return x / np.linalg.norm(x)

def build_confusion_matrix():
    n_frames = len(representations)
    confusion_matrix = np.zeros((n_frames, n_frames))
    for i in range(n_frames):
        for j in range(n_frames):
            print(i, j)
            confusion_matrix[i][j] = 1.0 - np.sqrt(1.0 - np.dot(normalize(representations[i]), normalize(representations[j])))
    return confusion_matrix

confusion_matrix = build_confusion_matrix()
sns.heatmap(confusion_matrix, ax = ax2, **default_heatmap_kwargs)
ax2.set_title('CNN')

fig.show()
fig.savefig('KITTI.png')