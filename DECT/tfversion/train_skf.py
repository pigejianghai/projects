import random
import tensorflow as tf
import datetime
import numpy as np
import albumentations as A
# import imblearn
import os
import tf_metrics
import argparse
################################################################################
################################################################################
parser = argparse.ArgumentParser(description='')

parser.add_argument('--energy_level', dest='ENERGY_LEVEL', type=int, default=None, 
                    help='# Energy Level')
parser.add_argument('--num_channels', dest='n_channels', type=int, default=11, 
                    help='# number of channels')
parser.add_argument('--num_class', dest='n_class', type=int, default=3, 
                    help='# number of class')
parser.add_argument('--gpu_num', dest='GPU_number', type=str, default='2', 
                    help='# CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
################################################################################
################################################################################
###### cudnn may brings deviation even if the random seed has been fixed #######
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_number
tf.config.experimental_run_functions_eagerly(True)
GPU_OPTIONS = tf.compat.v1.GPUOptions(allow_growth=True)
SESS = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=GPU_OPTIONS))
################################################################################
################################################################################
from tensorflow import keras
from sklearn.model_selection import StratifiedKFold

from utils import get_id_label, DECTGenerator
from model import DecouplingConvNet, Decoupling_SE_ConvNet
from loss_function import FocalLoss
from multiclass_metrics import MulticlassAUC
################################################################################
################################################################################
@tf.function
def train_step(model: keras.Model, optimizer, train_array, 
               labels, loss_object, 
               train_loss, train_accuracy, 
               train_auc):
            #    train_auc_0, train_auc_1, train_auc_2):
    with tf.GradientTape() as tape:
        predictions = model(train_array, training=True)
        loss = loss_object(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)
    train_auc(labels, tf.nn.softmax(predictions))
    # train_auc_0(labels, tf.nn.softmax(predictions))
    # train_auc_1(labels, tf.nn.softmax(predictions))
    # train_auc_2(labels, tf.nn.softmax(predictions))
@tf.function
def infer_step(model: keras.Model, infer_array, labels, 
               loss_object, infer_loss, 
               infer_accuracy, 
               infer_auc):
            #  infer_auc_0, infer_auc_1, infer_auc_2):
    predictions = model(infer_array)
    loss = loss_object(labels, predictions)

    infer_loss(loss)
    infer_accuracy(labels, predictions)
    infer_auc(labels, tf.nn.softmax(predictions))
    # infer_auc_0(labels, tf.nn.softmax(predictions))
    # infer_auc_1(labels, tf.nn.softmax(predictions))
    # infer_auc_2(labels, tf.nn.softmax(predictions))
################################################################################
################################################################################
ROOT_PATH ='/data1/jianghai/DECT/'
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 100
EPOCHS = 500
NUM_CLASSES = args.n_class
SE_ALPHA = 0.2

LEARNING_RATE = 1e-4
AUGMENTATION = A.Compose([
    A.Rotate(limit=18), 
    A.Flip(), 
])
################################################################################
################################################################################
train_txt = ROOT_PATH + 'txt/multi/multi_train.txt'
test_txt = ROOT_PATH + 'txt/multi/multi_test.txt'
id_list, label_list = get_id_label(train_txt)
test_id_list, test_label_list = get_id_label(test_txt)
################################################################################
################################################################################
test_dataset = DECTGenerator(IDs_list=test_id_list, label_list=test_label_list, 
                             num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, shuffle=False, 
                             n_channels=args.n_channels, energy_level=args.ENERGY_LEVEL, 
                             augmentation=None)

test_loss = keras.metrics.Mean(name='test_loss', dtype=tf.float32)
test_accuracy = keras.metrics.CategoricalAccuracy(name='test_accuracy')
test_auc = keras.metrics.AUC(name='test_auc')
# test_auc_0 = MulticlassAUC(name='test_auc_0', pos_label=0, sparse=False)
# test_auc_1 = MulticlassAUC(name='test_auc_1', pos_label=1, sparse=False)
# test_auc_2 = MulticlassAUC(name='test_auc_2', pos_label=2, sparse=False)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_auc', 
    verbose=1, 
    patience=50, 
    mode='max', 
    restore_best_weights=True
)

loss_object = keras.losses.CategoricalCrossentropy(from_logits=True)
# loss_object = FocalLoss(alpha=1.5, gamma=[0.6313, 0.2368, 0.1579])
optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

train_loss = keras.metrics.Mean(name='train_loss', dtype=tf.float32)
train_accuracy = keras.metrics.CategoricalAccuracy(name='train_accuracy')
train_auc = keras.metrics.AUC(name='train_auc')
# train_auc_0 = MulticlassAUC(name='train_auc_0', pos_label=0, sparse=False)
# train_auc_1 = MulticlassAUC(name='train_auc_1', pos_label=1, sparse=False)
# train_auc_2 = MulticlassAUC(name='train_auc_2', pos_label=2, sparse=False)

val_loss = keras.metrics.Mean(name='val_loss', dtype=tf.float32)
val_accuracy = keras.metrics.CategoricalAccuracy(name='val_accuracy')
val_auc = keras.metrics.AUC(name='val_auc')
# val_auc_0 = MulticlassAUC(name='val_auc_0', pos_label=0, sparse=False)
# val_auc_1 = MulticlassAUC(name='val_auc_1', pos_label=1, sparse=False)
# val_auc_2 = MulticlassAUC(name='val_auc_2', pos_label=2, sparse=False)
################################################################################
################################################################################
skf = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE, shuffle=True)
# ros = imblearn.over_sampling.RandomOverSampler(random_state=RANDOM_STATE)
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
for fold, (train_index, val_index) in enumerate(skf.split(id_list, label_list)):
    ############################################################################
    ############################################################################
    train_id_list, train_label_list = [], []
    val_id_list, val_label_list = [], []
    for t in train_index:
        train_id_list.append(id_list[t])
        train_label_list.append(label_list[t])
    for v in val_index:
        val_id_list.append(id_list[v])
        val_label_list.append(label_list[v])
    ############################################################################
    ############################################################################
    ## OverSampling
    # train_id_list = np.array(train_id_list).reshape(-1, 1)
    # train_id_list = np.reshape(-1, 1)
    # train_id_list, train_label_list = ros.fit_resample(train_id_list, train_label_list)
    # train_id_list = np.squeeze(train_id_list)
    # train_id_list = train_id_list.tolist()
    ############################################################################
    ############################################################################
    train_dataset = DECTGenerator(IDs_list=train_id_list, label_list=train_label_list, 
                                  num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, shuffle=True, 
                                  n_channels=args.n_channels, energy_level=args.ENERGY_LEVEL, 
                                  augmentation=AUGMENTATION)
    val_dataset = DECTGenerator(IDs_list=val_id_list, label_list=val_label_list, 
                                num_classes=NUM_CLASSES, batch_size=BATCH_SIZE, shuffle=False, 
                                n_channels=args.n_channels, energy_level=args.ENERGY_LEVEL, 
                                augmentation=None)

    model = Decoupling_SE_ConvNet(num_classes=NUM_CLASSES, se_alpha=SE_ALPHA)
    # model = DecouplingConvNet(num_classes=NUM_CLASSES)
    
    # model = keras.applications.vgg16.VGG16(weights=None, classes=NUM_CLASSES, input_shape=(110, 110, 11))
    # model = keras.applications.vgg19.VGG19(weights=None, classes=NUM_CLASSES, input_shape=(110, 110, 11))
    # model = keras.applications.ResNet50(weights=None, classes=NUM_CLASSES, input_shape=(110, 110, 11))
    # model = keras.applications.densenet.DenseNet121(weights=None, classes=NUM_CLASSES, input_shape=(110, 110, 11))
    
    train_log_dir = ROOT_PATH + 'code/tfversion/tensorboard/VC_' + current_time + '_fold_' + str(fold) + '/train'
    val_log_dir = ROOT_PATH + 'code/tfversion/tensorboard/VC_' + current_time + '_fold_' + str(fold) + '/val'
    test_log_dir = ROOT_PATH + 'code/tfversion/tensorboard/VC_' + current_time + '_fold_' + str(fold) + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    ############################################################################
    ############################################################################
    best_auc = 0.0
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        # train_auc_1.reset_states()
        # train_auc_2.reset_states()
        
        val_loss.reset_states()
        val_accuracy.reset_states()
        # val_auc_1.reset_states()
        # val_auc_2.reset_states()
        
        test_loss.reset_states()
        test_accuracy.reset_states()
        # test_auc_1.reset_states()
        # test_auc_2.reset_states()

        for train_array, train_labels in train_dataset:
            train_step(model, optimizer, train_array, train_labels, 
                       loss_object=loss_object, train_loss=train_loss, 
                       train_accuracy=train_accuracy, 
                       train_auc=train_auc)
                    #    train_auc_0=train_auc_0, train_auc_1=train_auc_1, train_auc_2=train_auc_2)
        with train_summary_writer.as_default():
            tf.summary.scalar('Loss/fold_' + str(fold), train_loss.result(), step=epoch)
            tf.summary.scalar('Acc/fold_' + str(fold), train_accuracy.result(), step=epoch)
            tf.summary.scalar('AUC/fold_' + str(fold), train_auc.result(), step=epoch)
            # tf.summary.scalar('AUC_0/fold_' + str(fold), train_auc_0.result(), step=epoch)
            # tf.summary.scalar('AUC_1/fold_' + str(fold), train_auc_1.result(), step=epoch)
            # tf.summary.scalar('AUC_2/fold_' + str(fold), train_auc_2.result(), step=epoch)

        for val_array, val_labels in val_dataset:
            infer_step(model, val_array, val_labels, 
                     loss_object=loss_object, infer_loss=val_loss, 
                     infer_accuracy=val_accuracy, 
                     infer_auc=val_auc)
                    #  infer_auc_0=val_auc_0, infer_auc_1=val_auc_1, infer_auc_2=val_auc_2)
            if best_auc < val_auc.result() and epoch > 5:
                best_auc = val_auc.result()
                best_model = model
        with val_summary_writer.as_default():
            tf.summary.scalar('Loss/fold_' + str(fold), val_loss.result(), step=epoch)
            tf.summary.scalar('Acc/fold_' + str(fold), val_accuracy.result(), step=epoch)
            tf.summary.scalar('AUC/fold_' + str(fold), val_auc.result(), step=epoch)
            # tf.summary.scalar('AUC_0/fold_' + str(fold), val_auc_0.result(), step=epoch)
            # tf.summary.scalar('AUC_1/fold_' + str(fold), val_auc_1.result(), step=epoch)
            # tf.summary.scalar('AUC_2/fold_' + str(fold), val_auc_2.result(), step=epoch)

        for test_array, test_labels in test_dataset:
            infer_step(model, test_array, test_labels, 
                      loss_object=loss_object, infer_loss=test_loss, 
                      infer_accuracy=test_accuracy, 
                      infer_auc=test_auc)
                    #   infer_auc_0=test_auc_0, infer_auc_1=test_auc_1, infer_auc_2=test_auc_2)
        with test_summary_writer.as_default():
            tf.summary.scalar('Loss/fold_' + str(fold), test_loss.result(), step=epoch)
            tf.summary.scalar('Acc/fold_' + str(fold), test_accuracy.result(), step=epoch)
            tf.summary.scalar('AUC/fold_' + str(fold), test_auc.result(), step=epoch)
            # tf.summary.scalar('AUC_0/fold_' + str(fold), test_auc_0.result(), step=epoch)
            # tf.summary.scalar('AUC_1/fold_' + str(fold), test_auc_1.result(), step=epoch)
            # tf.summary.scalar('AUC_2/fold_' + str(fold), test_auc_2.result(), step=epoch)
        
        print(
            f'Fold {fold},\t'
            f'Epoch {epoch + 1:0>4d},\t'
            # f'Train Loss: {train_loss.result():.4f},\t'
            # f'Train Accuracy: {train_accuracy.result():.4f},\t'
            f'Train_AUC: {train_auc.result():.4f},\t'
            # f'Train_AUC_Class_1: {train_auc_1.result():.4f},\t'
            # f'Train_AUC_Class_2: {train_auc_2.result():.4f},\t'
            # f'Test Loss: {test_loss.result():.4f},\t'
            # f'Test Accuracy: {test_accuracy.result():.4f},\t'
            f'Val Accuracy: {val_auc.result():.4f},\t'
            f'Test Accuracy: {test_auc.result():.4f},\t'
            # f'Val_AUC_Class_1: {val_auc_1.result():.4f},\t'
            # f'Val_AUC_Class_2: {val_auc_2.result():.4f},\t'
            # f'Test_AUC_Class_1: {test_auc_1.result():.4f},\t'
            # f'Test_AUC_Class_2: {test_auc_2.result():.4f}\t'
        )
    tf.saved_model.save(best_model, 
                        ROOT_PATH + "code/tfversion/checkpoint/" + current_time + "_fold_" + str(fold) + '_best_model')