NAME = '0_BASELINE'
PYTHON_VERSION = '3.9.7'
KERAS_VERSION = '2.9.0'
TENSOR_FLOW_GPU = '2.9.1'
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Input, Conv3D, AveragePooling3D, GlobalAveragePooling3D, MaxPool3D, BatchNormalization
from tensorflow.keras.layers import Convolution3D, MaxPooling3D, Convolution2D, AveragePooling2D, MaxPooling2D, ZeroPadding3D, ZeroPadding2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt
from classification_models_3D.tfkeras import Classifiers
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from traintestsplit import get_baseline_data, loaddata
from constants import *
from reportresults import *

import pandas as pd
import numpy as np


path = r'/mnt/beta/tjjmreintjes/Train/'

'''This is the baseline model, in which the train, validation and test data a retrieved from the traintestsplit.py are used in ResNet50.
This is done by using a data generator and the CSV file as ground_truth file. All data is stored with use of a pipeline from which a ROC and PR curve can be plotted.
Parameters can be adjusted in the constants.py file.'''


## Set up pipeline folder if missing
pipeline = os.path.join(path, 'pipeline', NAME)
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))
        for network in ['resnet']:
            os.makedirs(os.path.join(pipeline, folder, network))
            
            
def read_data(seed):
    global test_id, test_label_c, class_weights, train, validation
    global train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights

    ground_truth_file = os.path.join(path,'Data', 'total_nodule.csv')
    
    train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights = get_baseline_data(
        ground_truth_file, seed, VERBOSE)
    
    data1 = lambda: loaddata(train_id, train_label_c)
    data2 = lambda: loaddata(valid_id, valid_label_c)
    
    train = tf.data.Dataset.from_generator(data1, output_signature=(tf.TensorSpec(shape=[128, 128, 64, 1], dtype=tf.float32),
                                                                    tf.TensorSpec(shape=[], dtype=tf.int64))).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    validation = tf.data.Dataset.from_generator(data2, output_signature=(tf.TensorSpec(shape=[128, 128, 64, 1], dtype=tf.float32),
                                                                         tf.TensorSpec(shape=[], dtype=tf.int64))).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model():
    ResNet50, preprocess_input = Classifiers.get('resnet50')
    conv = ResNet50(include_top=False,
                     pooling=None,
                     input_shape=(128,128,64,1),
                     weights=None,
                     classes=1)
  
    x = tf.keras.layers.Flatten()(conv.output)
    out_class = tf.keras.layers.Dense(1, activation='sigmoid', name='out_class')(x)

    opt = tf.keras.optimizers.RMSprop(learning_rate= 0.000005)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic=True)
    model = keras.models.Model(conv.input, outputs=[out_class])
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    if VERBOSE:
        model.summary()
    return model


def fit_model(model):

    global history
    weights_filepath = os.path.join(r'/mnt/beta/tjjmreintjes/Train/pipeline/0_BASELINE/out/resnet/', str(seed)+'base.h5')
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min', min_delta = 0.001)
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [early_stopping, checkpoint]
    
    history = model.fit(
        train,
        epochs=EPOCHS,
        validation_data=validation,
        callbacks=callbacks_list,
        class_weight=class_weights)
    
def predict_model(model):

    data3 = lambda: loaddata(test_id, test_label_c)
    test = tf.data.Dataset.from_generator(data3, output_signature=(tf.TensorSpec(shape=[128, 128, 64, 1], dtype=tf.float32),
                                                                   tf.TensorSpec(shape=[], dtype=tf.int64))).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    predictions = model.predict(test)
    y_true = test_label_c
    delta_size = predictions.size - y_true.count()
    scores = np.resize(predictions, predictions.size - delta_size)
    y_true =y_true.to_numpy()
    y_true = y_true.astype('int')

    filename = get_output_filename(str(seed)+'predictions.csv')
    df = pd.DataFrame({'id': test_id, 'prediction': scores, 'true_label': y_true})
    with open(filename, mode='w') as f:
        df.to_csv(f, index=False)

    auc = roc_auc_score(y_true, scores)
    cf, ct, tresholds = roc_curve(y_true, scores)
    pf, pt, treshold = precision_recall_curve(y_true, scores)
    avscore = average_precision_score(y_true, scores)
    
    if PR == True:
        fig,ax = plt.subplots()
        ax.plot(pf, pt, color ='purple', label= 'seed['+str(seed) + '] - PS  = %0.3f' % avscore)
        ax.set_title('Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        ax.legend()
        plt.savefig(str(seed)+'PRcurve.png')
    
    if ROC == True:
        plt.plot(cf, ct, marker='.', label= 'seed['+str(seed) + '] - AUC  = %0.3f' % auc)
        plt.title('Receiver Operating Characteristic Curve')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive rate')
        plt.legend()
        plt.savefig(str(seed)+'ROC.png' )
    return auc
    

def get_output_filename(name):
    if NETWORK_SELECTED == NETWORK_TYPE.RESNET:
            filename = os.path.join(pipeline, 'out', 'resnet', name)
    return filename

def save_model(model, seed):
    model_json = model.to_json()

    filename = get_output_filename(str(seed)+'base')
    with open(filename + '.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights(filename + '.h5')
    
if VERBOSE:
    print_constants()
    
df_auc = pd.DataFrame(columns=['seed', 'auc'])
for seed in seeds:
    read_data(seed)

    model = build_model()

    fit_model(model)

    if SAVE_MODEL_WEIGHTS:
        save_model(model, seed)

    report_acc_and_loss(history, get_output_filename(str(seed)+'acc_and_loss.csv'))

    score = predict_model(model)
    df_auc = df_auc.append({'seed': seed, 'auc': score}, ignore_index=True)

report_auc(df_auc, get_output_filename('aucs.csv'))
