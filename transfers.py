import cv2
import PIL.Image
import numpy as np
import tensorflow as tf
from copy import deepcopy


def build_direction(loss, input_place, TI=False):
    import scipy.stats as st
    kernlen = 15; nsig = 3
    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel.astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
    stack_kernel = np.expand_dims(stack_kernel, 3)
    
    grad = tf.gradients(loss, input_place)[0]
    if TI: grad = tf.nn.depthwise_conv2d(grad, stack_kernel, strides=[1, 1, 1, 1], padding='SAME')
    direction = grad * tf.size(grad, out_type=tf.float32) / tf.norm(grad, ord=1)
    return direction


def random_padding(adv_image):
    original_size = (adv_image.shape[1], adv_image.shape[2])
    resized_factor = np.random.uniform(0.9, 1)
    new_size = (int(adv_image.shape[1]*resized_factor), int(adv_image.shape[2]*resized_factor))
    resized_image = cv2.resize(adv_image[0], (new_size[1], new_size[0]))[np.newaxis, ...]
    new_corner = (int(np.random.uniform(0, original_size[0] - new_size[0])), int(np.random.uniform(0, original_size[1] - new_size[1])))
    new_image = np.zeros(adv_image.shape)
    new_image[0, new_corner[0]:new_corner[0]+new_size[0], new_corner[1]:new_corner[1]+new_size[1], :] = resized_image[0]
    return new_image.astype(adv_image.dtype)


def update_sample(direction, sess, feed_dict, adv_image, input_placeholder, DI=False, SI=False): 
    direction_values = [sess.run(direction, feed_dict)]
    if SI: 
        for i in range(4):
            new_image = (adv_image * (0.5 ** i))
            feed_dict[input_placeholder] = new_image[np.newaxis, ...]
            direction_values.append(sess.run(direction, feed_dict))    
    if DI:
        for i in range(4):
            new_image = adv_image
            feed_dict[input_placeholder] = random_padding(new_image[np.newaxis, ...])
            direction_values.append(sess.run(direction, feed_dict))
    return sum(direction_values)/len(direction_values)