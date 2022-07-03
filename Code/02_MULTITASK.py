NAME = '1_MULTITASK'
PYTHON_VERSION = '3.9.7'
KERAS_VERSION = '2.9.0'
TENSOR_FLOW_GPU = '2.9.1'

import tensorflow as tf


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from matplotlib import pyplot as plt
from classification_models_3D.tfkeras import Classifiers
import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

from traintestsplit import get_multi_task_data, loaddata_multitask
from constants import *
from reportresults import *
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

import pandas as pd
import numpy as np


path = r'/mnt/beta/tjjmreintjes/Train/'

'''This is the multitask model, in which the train, validation and test data a retrieved from the traintestsplit.py are used in ResNet50. Everything is the same
as the baseline model, however, an annotation via the traintestsplit is added in model fit and compile function. All with use of a data generator and the CSV file 
as ground_truth file. Data is stored with use of a pipeline from which a ROC and PR curve can be plotted.
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
    global test_id, test_label_c, class_weights, train, validation, test_annotation
    global train_id, train_label_c, valid_id, valid_label_c, test_id, test_label_c, class_weights, train_annotation, valid_annotation
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    ground_truth_file = os.path.join(path,'Data', 'total_nodule.csv') #df_nodule
    
    train_id, train_label_c, train_annotation, valid_id, valid_label_c, valid_annotation, test_id, test_label_c, test_annotation, class_weights = get_multi_task_data(
        ground_truth_file, seed, annotation, VERBOSE)
    
    data1 = lambda: loaddata_multitask(train_id, train_label_c, train_annotation)
    data2 = lambda: loaddata_multitask(valid_id, valid_label_c, valid_annotation)
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train = tf.data.Dataset.from_generator(data1, output_types=(tf.float32, {'out_class': tf.int64, 'out_asymm': tf.float16}),
                                           output_shapes = (tf.TensorShape([128, 128, 64, 1]), {'out_class': tf.TensorShape([]),
                                                                                                'out_asymm': tf.TensorShape([])})).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    validation = tf.data.Dataset.from_generator(data2, output_types=(tf.float32, {'out_class': tf.int64, 'out_asymm': tf.float16}),
                                                output_shapes = (tf.TensorShape([128, 128, 64, 1]), {'out_class': tf.TensorShape([]),
                                                                                                     'out_asymm': tf.TensorShape([])})).batch(BATCH_SIZE).prefetch(AUTOTUNE)
def build_model():
    ResNet50, preprocess_input = Classifiers.get('resnet50')
    conv = ResNet50(include_top=False,
                     pooling=None,
                     input_shape=(128,128,64,1),
                     weights=None,
                     classes=1)
    
    x = tf.keras.layers.Flatten()(conv.output)
    x = keras.layers.Dense(256, activation='relu')(x)
    out_class = tf.keras.layers.Dense(1, activation='sigmoid', name='out_class')(x)
    out_asymm = tf.keras.layers.Dense(1, activation='linear', name='out_asymm')(x)

    opt = tf.keras.optimizers.RMSprop(learning_rate= 0.000005)
    opt = tf.keras.mixed_precision.LossScaleOptimizer(opt, dynamic=True)
    
    model = keras.models.Model(conv.input, outputs=[out_class, out_asymm])
    model.compile(loss={'out_class': 'binary_crossentropy', 'out_asymm': 'mse'}, loss_weights={'out_class': 0.5, 'out_asymm': 0.5},
                  optimizer=opt,
                  metrics={'out_class': 'accuracy'})

    if VERBOSE:
        model.summary()
    return model


def fit_model(model):

    global history
    weights_filepath = os.path.join(r'/mnt/beta/tjjmreintjes/Train/pipeline/1_MULTITASK/out/resnet/', str(seed)+'base.h5')
    early_stopping = EarlyStopping(monitor = 'val_loss', patience = 3, mode = 'min', min_delta = 0.001)
    checkpoint = ModelCheckpoint(weights_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [early_stopping, checkpoint]
    
    history = model.fit(
        train,
        epochs=EPOCHS,
        validation_data=validation,
        callbacks=callbacks_list)
    
def predict_model(model):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    weights_filepath = os.path.join(r'/mnt/beta/tjjmreintjes/Train/pipeline/1_MULTITASK/out/resnet/', str(seed)+'base.h5')
    model.load_weights(weights_filepath)
    data3 = lambda: loaddata_multitask(test_id, test_label_c, test_annotation)
    
    test = tf.data.Dataset.from_generator(data3, output_types=(tf.float32, {'out_class': tf.int64, 'out_asymm': tf.float16}), output_shapes=(tf.TensorShape([128, 128, 64, 1]), {'out_class': tf.TensorShape([]), 'out_asymm': tf.TensorShape([])})).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    predictions = model.predict(test)
    y_true = test_label_c
    delta_size = predictions[0].size - y_true.count()
    
    scores = np.resize(predictions[0], predictions[0].size - delta_size)

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
