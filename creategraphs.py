'''This file is used to manual plot raincloud, bar en boxplot, instead of running the whole model again'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import ptitprince as pt
var = pd.read_csv('total_nodule.csv')
var.head()

'''Make Raincloud plot for the annotation and diagnositic label'''

ytru = list(var['malignant'])
score = list(var['spiculation'])

ax = pt.RainCloud(x=ytru, y=score,
                  orient = 'h')

ax.set_title('Raincloud of spiculation and malignancy')
ax.set_ylabel('Malginant')
ax.set_xlabel('Spiculation')
ax.legend()
                  
'''Make bar chart with the distribution of train, validation and test test'''

datasets = ['Train', 'Validation', 'Test']
Malignant = [443,111,79]
Benign = [947, 237, 170]

x_axis = np.arange(len(datasets))

plt.bar(x_axis -0.2, Malignant, width=0.4, label = 'Malignant', color='orange')
plt.bar(x_axis +0.2, Benign, width = 0.4, label = 'Benign' , color='lightgreen')

plt.xticks(x_axis, datasets)

plt.legend()

plt.show()

'''Make boxplot to compare all AUC scores'''
allAUC = {'Baseline' : [0.927, 0.898, 0.882, 0.917, 0.919], 'Multitask' : [0.935, 0.898, 0.894, 0.915, 0.921]}

fig, ax = plt.subplots()
ax.boxplot(allAUC.values())

ax.set_xticklabels(allAUC.keys())
ax.set_ylabel( 'AUC score')
