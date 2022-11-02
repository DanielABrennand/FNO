###
#Imports
###

from torch import nn,optim,save,cuda,from_numpy,manual_seed
from torch.utils.data import DataLoader,TensorDataset
from torchvision import transforms
from os.path import join
from numpy import load,random
import wandb
import gc

#from UtilityFunctions import LogitPercentConverter,H5ToNumpy as LPC,H2N

###
#Globals/Constants
###

#Overall Info
PROJECT = "LHClassifierGeoffSaveLoadTest"
MODEL_NAME = "AlexNetV2"

DEVICE = "cuda" if cuda.is_available() else "cpu"

#Hyper Parameters
SHUFFLE = True
WORKERS = 0
BATCH_SIZE = 20

EPOCHS = 100
LOSS_FN = "BCELoss"
OPTIMIZER = "SGD"
MOMENTUM = 0.9
LEARNING_RATE = 0.001

CONFIDENCE_THRESHOLD = 0.8

SEED = None
if not SEED:
    SEED = random.randint(0,9223372036854775807)

#Data Locations
TRAINING_DATA_ROOT = "/home/dbren/VSCode/DataStore/NumpyData/Training"
TRAINING_MODES_PATH = "/home/dbren/VSCode/DataStore/NumpyData/Training/TrainingModes.npy"

VALIDATION_DATA_ROOT = "/home/dbren/VSCode/DataStore/NumpyData/Validation"
VALIDATION_MODES_PATH = "/home/dbren/VSCode/DataStore/NumpyData/Validation/ValidationModes.npy"

TESTING_DATA_ROOT = "/home/dbren/VSCode/DataStore/NumpyData/Testing"
TESTING_MODES_PATH = "/home/dbren/VSCode/DataStore/NumpyData/Testing/TestingModes.npy"

IMAGE_SIZE = 512
#Optional Modes
EPOCH_SAVE_INTERVAL = 0 #0 for off
FINAL_SAVING = True

TESTING = True

#Outputs
HEURISTICS_SAVE_PATH = "/home/dbren/VSCode/DataStore/Heuristics"

###
#WandB setup
###

configuration = {"Model": MODEL_NAME,
                 "Epochs": EPOCHS,
                 "Batch Size": BATCH_SIZE,
                 "Optimizer": OPTIMIZER,
                 "Loss Function": LOSS_FN,
                 "Learning Rate": LEARNING_RATE,
                 "Momentum": MOMENTUM,
                 "Device" : DEVICE,
                 "Epoch Save Interval": "Off" if EPOCH_SAVE_INTERVAL == 0 else EPOCH_SAVE_INTERVAL,
                 "Image Size": IMAGE_SIZE,
                 "Seed" : SEED,
                 "Confidence Threshold": CONFIDENCE_THRESHOLD
                }

run = wandb.init(project=PROJECT,
                 notes='',
                 config=configuration)

manual_seed(SEED)

###
#Model
###

from Models import AlexNet
Net = AlexNet().to(DEVICE)

###
#Data input
###
'''
TrainingDataFrames = from_numpy(load(TRAINING_DATA_ROOT))
TrainingDataModes = from_numpy(load(TRAINING_MODES_PATH))
TrainingDataSet = TensorDataset(TrainingDataFrames,TrainingDataModes)
TrainLoader = DataLoader(TrainingDataSet, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)
del TrainingDataFrames, TrainingDataModes

ValidationDataFrames = from_numpy(load(VALIDATION_DATA_ROOT))
ValidationDataModes = from_numpy(load(VALIDATION_MODES_PATH))
ValidationDataSet = TensorDataset(ValidationDataFrames,ValidationDataModes)
ValidationLoader = DataLoader(ValidationDataSet, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)
del ValidationDataFrames, ValidationDataModes

TestingDataFrames = from_numpy(load(TESTING_DATA_ROOT))
TestingDataMode = from_numpy(load(TESTING_MODES_PATH))
TestingDataSet = TensorDataset(TestingDataFrames,TestingDataMode)
TestLoader = DataLoader(TestingDataSet, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)
del TestingDataFrames, TestingDataMode

gc.collect()
'''
from DataSets import NumpyDataSet

TrainingDataSet = NumpyDataSet(TRAINING_DATA_ROOT,TRAINING_MODES_PATH)
TrainLoader = DataLoader(TrainingDataSet, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)

ValidationDataSet = NumpyDataSet(VALIDATION_DATA_ROOT,VALIDATION_MODES_PATH)
ValidationLoader = DataLoader(ValidationDataSet, batch_size=BATCH_SIZE, shuffle=SHUFFLE, num_workers=WORKERS)

TestingDataSet = NumpyDataSet(TESTING_DATA_ROOT,TESTING_MODES_PATH)
TestLoader = DataLoader(TestingDataSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

###
#Training loop
###

from TrainingLoops import BinaryClassiferTrainingLoop
from TestingLoops import BinaryClassiferTestingLoop

#LossFn = nn.BCELoss()
LossFn = nn.BCEWithLogitsLoss()
Optimizer = optim.SGD(Net.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
#Optimizer = optim.Adam(Net.parameters(),lr=LEARNING_RATE)

for Epoch in range(EPOCHS):
    TrainingLoss = BinaryClassiferTrainingLoop(DEVICE,TrainLoader,Net,LossFn,Optimizer)
    ValidationLoss,Correct = BinaryClassiferTestingLoop(DEVICE,ValidationLoader,Net,LossFn,CONFIDENCE_THRESHOLD)

    wandb.log({'Train Loss': TrainingLoss, 
               'Validation Loss': ValidationLoss,
               'Validation Correct': Correct})

    if EPOCH_SAVE_INTERVAL:
        if Epoch%EPOCH_SAVE_INTERVAL == 0:
            save(Net.state_dict(), join(HEURISTICS_SAVE_PATH,(PROJECT + "_" + wandb.run.name + "_Epoch_" + str(Epoch) + ".pth")))

del TrainingDataSet, TrainLoader, ValidationDataSet, ValidationLoader
gc.collect()

###
#Final Testing Loop
###

if TESTING:
    TestingLoss,Correct = BinaryClassiferTestingLoop(DEVICE,TestLoader,Net,LossFn,CONFIDENCE_THRESHOLD,True)
    wandb.log({'Final Test Loss':TestingLoss,
                'Final Test Correct': Correct})

###
#Outputs
###

if FINAL_SAVING:
    save(Net.state_dict(), join(HEURISTICS_SAVE_PATH,("{}_{}_Full.pth").format(PROJECT,wandb.run.name)))
