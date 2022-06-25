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
import pandas as pd
from pylidc.utils import consensus

import os

'''The pylidc module can combine or cluster the annotations provided in the LIDC database '''

#load scans + annotations
DCOM = pl.query(pl.Scan).all()

DCOM[0].annotations
'''As we see, each annooation has an id, there are nodule ids, 
but these don't coincide accross annotations, while in reality,
 some annotations concern the same actual nodule. This data is 
 combined in the 'nodule_number' column, which numbers the nodules for each patient

'''
def nodule_annotation(dcm):
    ''' Takes al dcm-files to a list and return a pandas DataFrame'''

    
    df = flatten(dcm[0].annotations[0]).iloc[0:0]
    df.assign(nodnumber = np.empty(0, dtype = "int32"))
    
    
    for file in dcm:
        for i, nod in enumerate(dcm.cluster_annotations()):
            allnodules = flatten(nod)
            allnodules = allnodules.assign(nodnumber = i+1)
            df = pd.concat([df, allnodules], axis = 0)
    return(df)

df_nodule = nodule_annotation(DCOM)
print(df_nodule)

df_nodule = df_nodule.assign(patient_number = df_nodule["patient_id"].str[-4:])
df_nodule = df_nodule.assign(nod_id = df_nodule[['patient_number', 'nodule_number']].apply(lambda x: 'P_'+str(x[0])+'_ND_'+str("%02d" % x[1]), axis = 1))

df_nodule.to_csv(os.path.join(r'C:\Users\20183282\Desktop\BEP MIA\BEP final\Data', "df_nodule.csv"))


df_nodule = df_nodule.assign(malignancy = (df_nodule.malignancy -3)/2)

total_nodule = df_nodule.groupby(["nodule_idx", "nodule_number",
                      "patient_id", "scan_id", "patient_number"], as_index = False).agg(
    {'malignancy': ['mean'], "annotation_id": 'count'}).rename(columns = {'annotation_id':'n_annotations'})
                          
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

def convert_to_numpy(scans, path, size_mm = 5, export_mask = False):
    ''' Convert the list of scans to numpy arrays and export mask, which can form a voxel'''

    for scan in scans:
        patient_id = scan.patient_id
        patient_number = patient_id[-4:]
        print(patient_id, end = "")
        nodules = scan.cluster_annotations()
        
        for i, nodule in enumerate(nodules):
            nodule_number = i+1
            nod_id = str(patient_number)+'_ND_'+str("%02d" % nodule_number)
            print(" nodule " +str(nodule_number), end = "")

            volume = nodule[0].scan.to_volume(verbose=0)
            
            mask,vol, x = consensus(nodule, clevel=0.5)             
            voxel = volume[vol]
            mask3d= mask
            try:
                np.save(file = os.path.join(path, 'P_'+str(nod_id)+"_array.npy"), arr = voxel)
            
                if export_mask:
                    np.save(file = os.path.join(path, 'P_'+str(nod_id)+"_mask.npy"), arr = mask3d)
                print("")
            except:
                print("-failed")
                
convert_to_numpy(DCOM, path = os.path.join(datadir, 'unsort'))



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------

def flatten(ann):
    ''' Take a list of annotations, return a pandas DataFrame '''

    nodid = np.zeros((len(ann), 
                       single_flatten(ann[0])[0].shape[0]), dtype = "<U14")
    feature = np.zeros((len(ann), 
                       single_flatten(ann[0])[1].shape[0]), dtype = "int64")
    
    for i, ann in enumerate(ann):
        ann_id, ann_val = single_flatten(ann)
        nodid[i,:] = ann_id
        feature[i,:] = ann_val
    
    df_nodid = pd.DataFrame(nodid, columns = ["patient_id", "nodule_idx", "annotation_id", "scan_id"])
    df_feature = pd.DataFrame(feature, columns = [
                                         'sublety', 'internalstructure', 'calcification',
                                         'sphericity', 'margin', 'lobulation', 'spiculation',
                                         'texture', 'malignancy'])
    df = pd.concat([df_nodid, df_feature], axis = 1)
    return(df)

def single_flatten(ann):
    '''Create a single row of the annotations '''
    
    ann_id = np.array([
        ann.scan.patient_id,
        ann._nodule_id,
        ann.id,
        ann.scan_id], 
        dtype = '<U14')
    ann_val = ann.ann_val()
    return(ann_id, ann_val)



