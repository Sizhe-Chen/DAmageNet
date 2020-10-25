from scipy.misc import imsave, imresize
import matplotlib.pyplot as plt
import tensorflow as tf
import PIL.Image
import numpy as np
import shutil
import time
import os
import cv2
import keras


paths = {
    'Data':   '../ILSVRC2012_img_val',
    'Label':  'val.txt'
}
for key in paths:
    assert os.path.exists(paths[key]), paths[key] + ' does not exist'
    if '.index' in paths[key]: paths[key] = paths[key].replace('.index', '')


def process_sample(sample_path, return_size):
    sample = PIL.Image.open(sample_path).convert('RGB')
    size, large_size, index = np.min(sample.size), np.max(sample.size), np.argmin(sample.size)
    if index: # long
        sample = sample.resize((int(return_size/size*large_size), return_size))
        cut_up, cut_down = int((np.max(sample.size) + return_size) / 2), int((np.max(sample.size) - return_size) / 2)
        sample = np.array(sample)[:, cut_down:cut_up, :] #sample.size = (a, b) -> np.array(sample).shape = (b, a, 3)
    else: # wide
        sample = sample.resize((return_size, int(return_size/size*large_size)))
        cut_up, cut_down = int((np.max(sample.size) + return_size) / 2), int((np.max(sample.size) - return_size) / 2)
        sample = np.array(sample)[cut_down:cut_up, :, :]
    
    sample = cv2.resize(sample, (return_size, return_size))
    return np.clip(sample.astype(np.float32), 0, 255)


def load_net(net_name, inp=None, return_net=True):
    size = {'InceptionV3': 299, 'Xception': 299, 'NASNetLarge': 331}.get(net_name, 224)
    if inp is None: inp = tf.placeholder(tf.float32, [1, size, size, 3])
    else:           inp = tf.image.resize_bilinear(inp, (size, size))
    if   net_name == 'ResNet50':        from keras.applications.resnet50 import ResNet50, preprocess_input; net = ResNet50(input_tensor=inp) if return_net else size
    elif net_name == 'ResNet101':       from keras_applications.resnet_v2 import ResNet101V2, preprocess_input; net = ResNet101V2(input_tensor=inp, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils) if return_net else size
    elif net_name == 'ResNet152':       from keras_applications.resnet_v2 import ResNet152V2, preprocess_input; net = ResNet152V2(input_tensor=inp, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils) if return_net else size
    elif net_name == 'InceptionResNetV2': from keras_applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input; net = InceptionResNetV2(input_tensor=inp, backend = keras.backend, layers = keras.layers, models = keras.models, utils = keras.utils) if return_net else size
    elif net_name == 'InceptionV3':     from keras.applications.inception_v3 import InceptionV3, preprocess_input; net = InceptionV3(input_tensor=inp) if return_net else size
    elif net_name == 'Xception':        from keras.applications.xception import Xception, preprocess_input; net = Xception(input_tensor=inp) if return_net else size
    elif net_name == 'VGG16':           from keras.applications.vgg16 import VGG16, preprocess_input; net = VGG16(input_tensor=inp) if return_net else size
    elif net_name == 'VGG19':           from keras.applications.vgg19 import VGG19, preprocess_input; net = VGG19(input_tensor=inp) if return_net else size
    elif net_name == 'DenseNet121':     from keras.applications.densenet import DenseNet121, preprocess_input; net = DenseNet121(input_tensor=inp) if return_net else size
    elif net_name == 'DenseNet169':     from keras.applications.densenet import DenseNet169, preprocess_input; net = DenseNet169(input_tensor=inp) if return_net else size
    elif net_name == 'DenseNet201':     from keras.applications.densenet import DenseNet201, preprocess_input; net = DenseNet201(input_tensor=inp) if return_net else size
    elif net_name == 'NASNetMobile':    from keras.applications.nasnet import NASNetMobile, preprocess_input; net = NASNetMobile(input_tensor=inp) if return_net else size
    elif net_name == 'NASNetLarge':     from keras.applications.nasnet import NASNetLarge, preprocess_input; net = NASNetLarge(input_tensor=inp) if return_net else size
    else: raise ValueError('Invalid Network Name')
    return net, preprocess_input


def get_time(deviation=0): return time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()-deviation))


def convert_second_to_time(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return "%02d:%02d:%02d" % (h, m, s)


def output(value_dict, stream=None, bit=3, prt=True, end='\n'):
    output_str = ''
    for key, value in value_dict.items():
        if isinstance(value, list): #value = value[-1]
            for i in range(len(value)): value[i] = round(value[i], bit)
        if isinstance(value, float) or isinstance(value, np.float32) or isinstance(value, np.float64): value = round(value, bit)
        output_str += '[ ' + str(key) + ' ' + str(value) + ' ] '
    if prt: print(output_str, end=end)
    if stream is not None: print(output_str, file=stream)


def copy_files(result_dir):
    for root, _, files in os.walk('.'):
        if '-' in root or '__pycache__' in root: continue
        for file in files:
            if '.py' not in file: continue
            destiny_path = result_dir + root[1:]
            os.makedirs(destiny_path, exist_ok=True)
            shutil.copyfile(root + '/' + file, destiny_path + '/' + file)


def crop_or_pad(sample, size):
    img = PIL.Image.fromarray(sample.astype(np.uint8))
    if img.size[0] > size:
        img = img.crop(((img.size[0] - size) / 2, (img.size[1] - size) / 2, (img.size[0] + size) / 2, (img.size[1] + size) / 2))
        img = img.resize((size, size))
    else:
        black = PIL.Image.fromarray(np.zeros((size, size, 3), dtype=np.uint8))
        black.paste(img, (int((size - img.size[0]) / 2), int((size - img.size[1]) / 2)))
        img = black
    return np.array(img)


def heatmap(heatmap, cmap="seismic", interpolation="none", colorbar=False, M=None):
    if M is None:
        M = np.abs(heatmap).max()
        if M == 0: M = 1
    plt.imshow(heatmap, cmap=cmap, vmax=M, vmin=-M, interpolation=interpolation)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if colorbar: plt.colorbar()


def initialize_uninitialized(sess):
    global_vars          = tf.global_variables()
    is_not_initialized   = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
    if len(not_initialized_vars): sess.run(tf.variables_initializer(not_initialized_vars))


def save_images(images, result_dir, name):
    if len(images) == 0: return
    assert images[0].dtype == np.uint8, 'images must be uint8'
    number = len(images)
    n_img_x = int(np.sqrt(number)) if int(np.sqrt(number)**2) == number else int(np.sqrt(number)) + 1
    plot = Plot(result_dir, n_img_x=n_img_x, img_w=images[0].shape[1], img_h=images[0].shape[0], img_c=images[0].shape[2])
    plot.add_image(images)
    plot.save_images(name)
    plot.clear()


class Plot:
    def __init__(self, directory, n_img_x, img_w, img_h, img_c=3, resize_factor=1, interval=1):
        self.directory = directory
        if not os.path.exists(directory): os.makedirs(directory)
        assert isinstance(interval, int)
        assert interval >= 1
        self.interval = interval
        assert n_img_x > 0
        self.n_img_x = n_img_x
        assert img_w > 0 and img_h > 0
        self.img_w = img_w
        self.img_h = img_h
        assert resize_factor > 0
        self.resize_factor = resize_factor
        assert img_c == 1 or img_c == 3
        self.img_c = img_c
        self.img_list = []

    def save_images(self, name='result.jpg'):
        imsave(os.path.join(self.directory, name), self._merge(self.img_list[::self.interval]))

    def _merge(self, image_list):
        size_y = len(image_list) // self.n_img_x + (1 if len(image_list) % self.n_img_x != 0 else 0)
        size_x = self.n_img_x
        h_ = int(self.img_h * self.resize_factor)
        w_ = int(self.img_w * self.resize_factor)
        img = np.zeros((h_ * size_y, w_ * size_x, self.img_c))
        
        for idx, image in enumerate(image_list):
            i = int(idx % size_x)
            j = int(idx / size_x)
            image_ = image#imresize(image, size=(w_,h_), interp='bicubic')
            img[j*h_:j*h_+h_, i*w_:i*w_+w_, :] = image_.reshape((self.img_w, self.img_h, self.img_c))
        return img.squeeze()

    def add_image(self, img):
        if isinstance(img, list): self.img_list += img
        else: self.img_list.append(img)
        return self

    def clear(self):
        self.img_list = []