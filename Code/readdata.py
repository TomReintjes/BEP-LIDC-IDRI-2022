"""
While pylidc has many functions for analyzing and querying only annotation data 
(which do not require DICOM image data access), pylidc also has many functions
 that do require access to the DICOM files associated with the LIDC dataset. 
 pylidc looks for a special configuration file that tells it where DICOM data 
 is located on your system. You can use pylidc without creating this configuration file,
 but of course, any functions that depend on CT image data will not be usable.

pylidc looks in your home folder for a configuration file called C:Users[User]pylidc.conf. With format:
    [dicom]
    path = /path/to/big_external_drive/datasets/LIDC-IDRI
    warn = True

https://pylidc.github.io/install.html
"""
import pylidc as pl
import numpy as np


import os

from export_np_from_annotations import *

'''The pylidc module can combine or cluster the annotations provided in the LIDC database '''

#load scans + annotations
DCOM = pl.query(pl.Scan).all()

DCOM[0].annotations
'''As we see, each annooation has an id, there are nodule ids, 
but these don't coincide accross annotations, while in reality,
 some annotations concern the same actual nodule. This data is 
 combined in the 'nodule_number' column, which numbers the nodules for each patient

'''
df_nodule = flatten_annotations_by_nodule(DCOM)
print(df_nodule)

df_nodule = df_nodule.assign(patient_number = df_nodule["patient_id"].str[-4:])
df_nodule = df_nodule.assign(nod_id = df_nodule[['patient_number', 'nodule_number']].apply(lambda x: 'P_'+str(x[0])+'_ND_'+str("%02d" % x[1]), axis = 1))

df_nodule.to_csv(os.path.join(r'C:\Users\20183282\Desktop\BEP MIA\BEP final\Data', "df_nodule.csv"))


df_nodule = df_nodule.assign(malignancy = (df_nodule.malignancy -3)/2)

total_nodule = df_nodule.groupby(["nodule_idx", "nodule_number",
                      "patient_id", "scan_id", "patient_number"], as_index = False).agg(
    {'malignancy': ['mean'], "annotation_id": 'count'}).rename(columns = {'annotation_id':'n_annotations'})
                          
                          
total_nodule = flatten_multiindex_columns(total_nodule)
total_nodule = total_nodule.assign(
    borderline = (np.sign(total_nodule.malignancy_min) == -np.sign(total_nodule.malignancy_max)) | (
                np.abs(total_nodule.malignancy_mean) <= 0.5 / 4),
    malignant = total_nodule.malignancy_mean > 0)
    
total_nodule = total_nodule.assign(borderline = (total_nodule.malignancy > 2) & (total_nodule.malignancy < 4))
total_nodule = total_nodule.assign(malignant = total_nodule.malignancy > 3)


total_nodule.to_csv(os.path.join(r'C:\Users\20183282\Desktop\BEP MIA\BEP final\Data', "total_nodule2.csv"))

total_nodule.head()

datadir = r'C:\Users\20183282\Desktop\BEP MIA\BEP final\Data\lidc'
os.listdir(datadir)
if not os.path.exists(os.path.join(datadir, 'unsort')):
    os.makedirs(os.path.join(datadir, 'unsort'))

resample_and_crop(DCOM, path = os.path.join(datadir, 'unsort'))

    
                         
