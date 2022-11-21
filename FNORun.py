###
#Imports
###

from torch import nn,optim,save,cuda,from_numpy,manual_seed
from torch.utils.data import DataLoader,TensorDataset
from torchvision import transforms
from os.path import join
from numpy import load,random,save,array
from UtilityFunctions import LpLoss
from simvue import Run
import time
import wandb
import gc

#from UtilityFunctions import LogitPercentConverter,H5ToNumpy as LPC,H2N

###
#Globals/Constants
###

#Overall Info
PROJECT = "FNOGeoffTest1"
MODEL_NAME = "CAMFNOV1"

DEVICE = "cuda" if cuda.is_available() else "cpu"

#Hyper Parameters
SHUFFLE = True
WORKERS = 0
BATCH_SIZE = 8

EPOCHS = 1
LOSS_FN = "LpLoss"
OPTIMIZER = "Adam"
LEARNING_RATE = 0.001

T_IN = 10
T_OUT = 50
STEP = 1

CUTOFF = 200

SEED = None
if not SEED:
    SEED = random.randint(0,9223372036854775807)

#Data Locations
H5_TRAINING_FILE_LOCATION = "/home/dbren/VSCode/DataStore/RBB_FILES/Training"
H5_VALIDATION_FILE_LOCATION = "/home/dbren/VSCode/DataStore/RBB_FILES/Validation"
H5_TESTING_FILE_LOCATION = "/home/dbren/VSCode/DataStore/RBB_FILES/Testing"

IMAGE_SIZE = "448 x 640"

#Optional Modes
EPOCH_SAVE_INTERVAL = 0 #0 for off
FINAL_SAVING = True

TESTING = True

#Outputs
HEURISTICS_SAVE_PATH = "/home/dbren/VSCode/DataStore/Heuristics"

###
#SimVue setup
###

configuration = {"Model": MODEL_NAME,
                 "Epochs": EPOCHS,
                 "Batch Size": BATCH_SIZE,
                 "Optimizer": OPTIMIZER,
                 "Loss Function": LOSS_FN,
                 "Learning Rate": LEARNING_RATE,
                 "Device" : DEVICE,
                 "Epoch Save Interval": "Off" if EPOCH_SAVE_INTERVAL == 0 else EPOCH_SAVE_INTERVAL,
                 "Image Size": IMAGE_SIZE,
                 "Seed" : SEED,
                 "T_in": T_IN,
                 "T_Out" : T_OUT,
                 "Step" : STEP,
                 "Cutoff" : CUTOFF
                }

run = Run()


#run.init(folder="/HPC-AI", tags=['OpenFOAM', 'Turbulent', 'Laminar', 'Surrogate', 'FNO'], metadata=configuration)
run.init(folder = "/Geoff",tags = ['FNO','Test','Geoff'], metadata = configuration)

manual_seed(SEED)

###
#Model 
###

from Models import FNO2d
Net = FNO2d(16,16,16).to(DEVICE)

###
#Data input
###
from DataSets import FNOH5DataSet

TrainingData = FNOH5DataSet(H5_TRAINING_FILE_LOCATION,T_IN,CUTOFF)
TrainLoader = DataLoader(TrainingData,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=WORKERS)

ValidationData = FNOH5DataSet(H5_VALIDATION_FILE_LOCATION,T_IN,CUTOFF)
ValidationLoader = DataLoader(ValidationData,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=WORKERS)

TestingData = FNOH5DataSet(H5_TESTING_FILE_LOCATION,T_IN,CUTOFF)
TestLoader = DataLoader(TestingData,batch_size=BATCH_SIZE,shuffle=SHUFFLE,num_workers=WORKERS)
###
#Training loop
###

from TrainingLoops import FNOTrainingLoop
from TestingLoops import FNOTestingLoop

LossFn = LpLoss(size_average=False)
Optimizer = optim.Adam(Net.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

for Epoch in range(EPOCHS):
    TrainingLoss = FNOTrainingLoop(DEVICE,TrainLoader,Net,LossFn,Optimizer,T_OUT,STEP)
    ValidationLoss = FNOTestingLoop(DEVICE,ValidationLoader,Net,LossFn,Optimizer,T_OUT,STEP)

    TrainingLoss /= len(TrainingData)
    ValidationLoss /= len(ValidationData)

    #Logging
    run.log_metrics({'Training Loss' : TrainingLoss, 'Validation Loss' : ValidationLoss})

    if EPOCH_SAVE_INTERVAL:
        if Epoch%EPOCH_SAVE_INTERVAL == 0:
            save(Net.state_dict(), join(HEURISTICS_SAVE_PATH,(PROJECT + "_" + time.time() + "_Epoch_" + str(Epoch) + ".pth")))

run.update_metadata({'Training Loss' : TrainingLoss, 'Validation Loss' : ValidationLoss})

###
#Final Testing Loop
###

if TESTING:
    TestingLoss,AllPreds = FNOTestingLoop(DEVICE,TestLoader,Net,LossFn,Optimizer,T_OUT,STEP,True)
    TestingLoss /= len(TestingData)
    run.update_metadata({'Testing Loss' : TestingLoss})
    save('Predictions.npy',array(AllPreds))
    run.save('Predictions.npy','Final Predictions')


###
#Outputs
###

if FINAL_SAVING:
    save(Net.state_dict(), join(HEURISTICS_SAVE_PATH,("{}_{}_Full.pth").format(PROJECT,time.time)))

run.close()