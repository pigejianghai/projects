import tensorflow as tf
import numpy as np
from tensorflow import keras
import albumentations as A
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_file
from IPython.display import display

def read_npy_file(path, level):
    data = np.load(path)
    data = data.transpose(1, 2, 0)
    slice = data[level, ...]
    # return data.astype(np.float32)
    return slice.astype(np.float32)
def to_npy_array(id_list):
    array = []
    for i in id_list:
        array.append(read_npy_file(i))
    return array
def get_id_label(txt_path):
    id_list, label_list = [], []
    f = open(txt_path, 'r')
    for line in f:
        # id_list.append('/data1/jianghai/DECT/npy/dataset/'+ line.split()[0] + '_0.npy')
        id_list.append(line.split()[0])
        label_list.append(int(line.split()[1]))# * 2
    return id_list, label_list
def get_batch(image, label):#, batch_size, capacity
    # Step 1: queue and tensor generation
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    input_queue = tf.compat.v1.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.py_function(read_npy_file, [input_queue[0], [tf.float32,]])
    # Step 2: decode
    # Step 3: augmentation
    # Step 4:batch generation
    # image_batch, label_batch = tf.data.Dataset.batch([image_contents, label], batch_size=batch_size, num_threads=16, capacity=capacity)
    # label_batch = tf.reshape(label_batch, [batch_size])
    # image_batch = tf.cast(image_batch, tf.float32)

    return image_contents, label

def npy_load(path, level):
    array = np.load(path).astype(np.float32)
    array = array.transpose(1, 2, 0)
    for s in range(array.shape[2]):
        single = array[:, :, s]
        array[:, :, s] = (single - np.min(single)) / (np.max(single) - np.min(single))
    if level is not None:
        return np.expand_dims(array[..., level], -1)
    else:
        return array.astype(np.float32)

class DECTGenerator(keras.utils.Sequence):
    
    def __init__(self, IDs_list, label_list, num_classes=4, 
                 batch_size=32, augmentation=None, shuffle=True, 
                 image_size=(110, 110), n_channels=1, 
                 energy_level=None):
        self.IDs_list, self.label_list = IDs_list, label_list
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.shuffle = shuffle
        self.image_size = image_size
        self.n_channels = n_channels
        self.energy_level = energy_level
        self.on_epoch_end()
        self.label_dict_transfer()
        
    def __len__(self):
        return int(np.floor(len(self.IDs_list) / self.batch_size))
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.IDs_list[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y
    
    def label_dict_transfer(self):
        self.label_dict = {}
        for l in range(len(self.label_list)):
            self.label_dict[self.IDs_list[l]] = self.label_list[l]
            
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.IDs_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.image_size, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            if self.augmentation is None:
                X[i,] = npy_load('/data1/jianghai/DECT/npy/dataset/' + ID + '_0.npy', self.energy_level)
            else:
                X_sample = npy_load('/data1/jianghai/DECT/npy/dataset/' + ID + '_0.npy', self.energy_level)
                augmented = self.augmentation(image=X_sample)
                X[i,] = augmented["image"]
            y[i,] = self.label_dict[ID]
        # to one_hot labels
        return X, keras.utils.to_categorical(y, num_classes=self.num_classes)
##########################################################################################
##########################################################################################
##########################################################################################
# def vgg16_mura_model():
#     """Get a vgg16 model.

#     The model can classify bone X-rays into three categories:
#     wrist, shoulder and elbow.
#     It will download the weights automatically for the first time.

#     Return:
#         A tf.keras model object.
#     """
#     path_weights = get_file(
#         "tf_keras_vgg16_mura_model.h5",
#         WEIGHTS_PATH_VGG16_MURA,
#         cache_subdir="models")

#     model = load_model(path_weights)

#     return model
def preprocess_image(img_path, target_size=(224, 224)):
    """Preprocess the image by reshape and normalization.

    Args:
        img_path: A string.
        target_size: A tuple, reshape to this size.
    Return:
        An image array.
    """
    # img = image.load_img(img_path, target_size=target_size)
    img = np.load(img_path)
    img = img.transpose(1, 2, 0)
    # img = image.img_to_array(img)
    img /= 255

    return img


def show_imgwithheat(img_path, heatmap, alpha=0.4, return_array=False):
    """Show the image with heatmap.

    Args:
        img_path: string.
        heatmap: image array, get it by calling grad_cam().
        alpha: float, transparency of heatmap.
        return_array: bool, return a superimposed image array or not.
    Return:
        None or image array.
    """
    # img = cv2.imread(img_path)
    img = np.load(img_path)
    img = img.transpose(1, 2, 0)
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    imgwithheat = Image.fromarray(superimposed_img)
    try:
        display(imgwithheat)
    except NameError:
        imgwithheat.show()

    if return_array:
        return superimposed_img
##########################################################################################
##########################################################################################
##########################################################################################
if __name__ == '__main__':
    path = '/data1/jianghai/DECT/txt/multi/multi_test.txt'
    id_list, label_list = get_id_label(path)
    augmentation = A.Compose([
        A.Rotate(limit=18), 
        A.Flip(), 
    ])
    data_generator = DECTGenerator(IDs_list=id_list, label_list=label_list, 
                                   num_classes=6, 
                                   batch_size=4, augmentation=augmentation)
    # print(len(data_generator))
    for image_batch, label_batch in data_generator:
        print(image_batch.shape, label_batch)