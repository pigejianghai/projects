import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
tf.config.experimental_run_functions_eagerly(True)
import numpy as np
import random
########################################
########################################
# RANDOM_STATE = 42
# random.seed(RANDOM_STATE)
# np.random.seed(RANDOM_STATE)
# tf.random.set_seed(RANDOM_STATE)
# os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# tf.config.experimental_run_functions_eagerly(True)
# GPU_OPTIONS = tf.compat.v1.GPUOptions(allow_growth=True)
# SESS = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=GPU_OPTIONS))
# cudnn may brings deviation in the final indexes even if the random seed has been fixed
########################################
########################################
from tensorflow import keras
from utils import get_id_label, DECTGenerator
from sklearn.metrics import classification_report, confusion_matrix

# @tf.function
# def test_step(model, test_array, labels):
#     predictions = model.call(test_array)
#     loss = loss_object(labels, predictions)

#     test_loss(loss)
#     test_accuracy.update_state(labels, predictions)
#     test_auc.update_state(labels, predictions)


# model = DecouplingConvNet()
model = tf.saved_model.load("/data1/jianghai/DECT/code/tfversion/checkpoint/20240305_142448_fold_0_best_model")
# model = keras.models.load_model('/data1/jianghai/DECT/code/keras/checkpoint/20231214_200012_fold_0_best_model', compile=False)
# print(model)
# model.summary()
NUM_CLASSES = 4
BATCH_SIZE = 32
ROOT_PATH = '/data1/jianghai/DECT/'
test_txt = ROOT_PATH + 'txt/multi/multi_test.txt'
test_id_list, test_label_list = get_id_label(test_txt)

test_dataset = DECTGenerator(IDs_list=test_id_list, label_list=test_label_list, 
                             num_classes=NUM_CLASSES, batch_size=1, shuffle=False, 
                             augmentation=None)

loss_object = keras.losses.CategoricalCrossentropy(from_logits=True)
test_accuracy = keras.metrics.CategoricalAccuracy(name='test_accuracy')
test_auc = keras.metrics.AUC(name='test_auc')
test_loss = keras.metrics.Mean(name='test_loss', dtype=tf.float32)
# keras.metrics.


true = []
pred = []
for test_array, test_labels in test_dataset:
    predictions = model(test_array, False)
    loss = loss_object(test_labels, predictions)

    test_loss(loss)
    test_accuracy(test_labels, predictions)
    test_auc(test_labels, tf.nn.softmax(predictions))
    
    for t in test_labels.tolist():#tf.nn.softmax(test_labels):
        # true.append(np.argmax(t))
        true.append(t)
    for p in predictions.numpy().tolist():#tf.nn.softmax(predictions):
        # pred.append(np.argmax(p))
        pred.append(p)

print(f'Test AUC: {test_auc.result()}')
print(true)
# print(type(true[0]))
# print(true.shape)

print(pred)
# print(type(pred))
# print(classification_report(y_true=true, y_pred=pred))
# print(confusion_matrix(y_true=true, y_pred=pred))
