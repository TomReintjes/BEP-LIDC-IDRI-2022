TRIAL = False
SANITY_CHECK = False
VERBOSE = True
BATCH_SIZE = 10
INPUT_SHAPE = (128, 128, 64, 1)
SAVE_MODEL_WEIGHTS = True
MODEL_NAME = 'RESNET'
annotation = 'spiculation'
ROC = False
PR= True

path = r'C:\Users\20183282\Desktop\BEP MIA\BEP final'

if TRIAL:
    STEPS_PER_EPOCH = 2
    EPOCHS = 2
    seeds = [1970, 1972]
else:
    STEPS_PER_EPOCH = 20
    EPOCHS = 20
    seeds = [1970, 1972, 2008, 2019, 2020]

def print_constants():
    print('*************************************************************')
    print('SANITY_CHECK           = ', str(SANITY_CHECK))
    print('SAVE_MODEL_WEIGHTS     = ', str(SAVE_MODEL_WEIGHTS))
    print('TRIAL                  = ', str(TRIAL))
    print('*************************************************************')
