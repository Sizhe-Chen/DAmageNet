from keras.models import Model
import numpy as np
import innvestigate.utils as iutils
import os
from keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
import PIL.Image
import matplotlib.pyplot as plt
import cv2

from interpreters import SGLRP
from utils import *


def visualize_lrp(analysis, size=440, get_signature=False):
    if get_signature: return cv2.resize(analysis / np.max(np.abs(analysis)) * 255, (size, size))
    heatmap(analysis.sum(axis=2))
    buffer_path = str(time.time() + np.random.choice(range(1000))).replace('.', '') + '.png'
    plt.savefig(buffer_path)
    img = PIL.Image.open(buffer_path).convert('RGB')
    os.remove(buffer_path)
    plt.clf()
    return cv2.resize(np.array(img)[18:458, 100:540, ...], (size, size))


def build_lrp(partial_model, out=None, out_ori=None):
    inp = partial_model.input
    if out is None: 
        assert out_ori is None
        out = partial_model.output
    if out_ori is not None: 
        indics_ori = tf.argmax(out_ori[0])
        out_modified = tf.where(tf.equal(tf.range(1000), tf.cast(indics_ori, tf.int32)), tf.zeros(1000), out[0])
        target_id = tf.argmax(out_modified)
    else:
        target_id = tf.argmax(out[0])
        
    analyzer = SGLRP(partial_model, target_id=target_id, relu=True, low=tf.reduce_min(inp), high=tf.reduce_max(inp))
    analysis = analyzer.analyze(inp)
    return analysis[0]


def test_lrp():
    for net_name in ['VGG19', 'ResNet50', 'DenseNet201']:
        sess = tf.InteractiveSession()
        model, preprocess_input = load_net(net_name) 
        analysis = build_lrp(Model(inputs=model.inputs, outputs=iutils.keras.graph.pre_softmax_tensors(model.outputs), name=model.name + '_partial'))
        
        image_size = int(model.input.shape[1])
        image_imagenet = preprocess_input(process_sample('demo/ImageNet_ILSVRC2012_val_00000046.JPEG', image_size))
        image_damagenet = preprocess_input(process_sample('demo/DAmageNet_ILSVRC2012_val_00000046.png', image_size))
        
        plot = Plot('.', n_img_x=2, img_w=440, img_h=440, img_c=3)
        plot.add_image(visualize_lrp(sess.run(analysis, {model.input: [image_imagenet]})))
        plot.add_image(visualize_lrp(sess.run(analysis, {model.input: [image_damagenet]})))
        img_path = 'demo/' + net_name + '.png'
        plot.save_images(img_path)
        sess.close()
        print('\n', img_path, '\n')


if __name__ == '__main__':
    test_lrp()