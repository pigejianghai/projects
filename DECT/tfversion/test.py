import tensorflow as tf
import numpy as np
# from sklearn.model_selection import StratifiedGroupKFold, KFold

from utils import get_id_label, to_npy_array
from model import DecouplingConvNet

train_txt = '/data1/jianghai/DECT/txt/binary/train.txt'
test_txt = '/data1/jianghai/DECT/txt/binary/test.txt'
train_npy_path_list, train_label_list = get_id_label(train_txt, flag='train')
test_npy_path_list, test_label_list = get_id_label(test_txt, flag='test')

train_dataset = tf.data.Dataset.from_tensor_slices((to_npy_array(train_npy_path_list), train_label_list))
test_dataset = tf.data.Dataset.from_tensor_slices((to_npy_array(test_npy_path_list), test_label_list))
# for array, label in train_dataset:
#     print(array.shape, label)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model = DecouplingConvNet()

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')
train_auc = tf.keras.metrics.AUC(name='train_auc')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.Accuracy(name='test_accuracy')
test_auc = tf.keras.metrics.AUC(name='test_auc')

@tf.function
def train_step(array, labels):
    with tf.GradientTape() as tape:
        predictions = model(array, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
@tf.function
def test_step(array, labels):
    predictions = model(array, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)