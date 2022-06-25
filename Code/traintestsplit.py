PYTHON_VERSION = '3.9.7'


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import os


path = r'/mnt/beta/tjjmreintjes/Train/'

'''File is used to create 3D arrays and to create the train, validation and test data for both 
baseline and multitask model'''


def padding3D(array, xx, yy, zz):
    #:param array: numpy array
    #:param xx: desired height
    #:param yy: desirex width
    #:return: padded array

    h = array.shape[0]
    w = array.shape[1]
    d = array.shape[2]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    c = (zz - d) // 2
    cc = zz - c - d

    return np.pad(array, pad_width=((a, aa), (b, bb), (c, cc)), mode='constant')

def get_baseline_data(ground_truth_file, seed, verbose):
    df = pd.read_csv(ground_truth_file)
    class_label = df['malignant']
    class_id = df['nodule_idx']

    X_train, X_test, y_train, y_test = train_test_split(
        class_id,
        class_label,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=class_label)
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=y_train)
    
    class_weights = class_weight.compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
    class_weights = {i: class_weights[i] for i in range(2)}

    if verbose:
        print('in train set      = \n' + str(y_train.value_counts()))
        print('in validation set = \n' + str(y_validate.value_counts()))
        print('in test set       = \n' + str(y_test.value_counts()))
    
    return (X_train, y_train, X_validate, y_validate, X_test, y_test, class_weights)

       
def loaddata(noduleid, label):
    for i in range(len(noduleid)):
        nid = noduleid.iloc[i]
        y_label = label.iloc[i]
        x_label = np.load(os.path.join(path, 'Data', 'unsort', f'{nid[2:6]}{nid[-2:]}_array.npy'))
        nodule_padded = padding3D(x_label, 128, 128, 64)
        nodule_padded = np.resize(nodule_padded, (128, 128, 64, 1))

        yield nodule_padded, y_label

#-------------------------------------------------------------------------------------------------------------------------------


def get_multi_task_data(ground_truth_file, seed, annotation, verbose):#, statsOutputFilename): TODO wat doet statsoutput hier
    df = pd.read_csv(ground_truth_file)
    class_label = df['malignant']
    class_id = df['nodule_idx']

    X_train, X_test, y_train, y_test = train_test_split(
        class_id,
        class_label,
        test_size=0.125,
        random_state=seed,
        shuffle=True,
        stratify=class_label)
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
        stratify=y_train)

    df.set_index("nodule_idx", inplace=True)

    Train_list = X_train.tolist()
    ann_list = []
    for name in Train_list:
        ann_row = df.loc[name][annotation]
        ann_list.append(ann_row)
    annotation_train = pd.Series(ann_list)

    noduleid = X_validate.tolist()
    ann_list = []
    for name in noduleid:
        ann_row = df.loc[name][annotation]
        ann_list.append(ann_row)
    annotation_valid = pd.Series(ann_list)

    noduleid = X_test.tolist()
    ann_list = []
    for name in noduleid:
        ann_row = df.loc[name][annotation]
        ann_list.append(ann_row)
    annotation_test = pd.Series(ann_list)

    w = class_weight.compute_class_weight('balanced', classes = np.unique(y_train), y = y_train)
    
    class_weights = dict()
    class_weights[0] = w[0]
    class_weights[1] = w[1]

    return (X_train, y_train, annotation_train,
            X_validate, y_validate, annotation_valid,
            X_test, y_test, annotation_test, class_weights)

def loaddata_multitask(noduleid, label, annotation): #, sample_weight):
    for i in range(len(noduleid)):
        nid = noduleid.iloc[i]
        y_label = label.iloc[i]
        x_label = np.load(os.path.join(path, 'Data', 'unsort', f'{nid[2:6]}{nid[-2:]}_array.npy'))
        nodule_padded = padding3D(x_label, 128, 128, 64)
        nodule_padded = np.resize(nodule_padded, (128, 128, 64, 1))

        y_label = label.iloc[i]
        annotation = annotation.iloc[i]
        annotation = (annotation-min(annotation))/(max(annotation)-min(annotation)) #normalize annotation


        yield (nodule_padded, {'out_class': np.asarray(y_label), 'out_asymm': np.asarray(annotation)})
