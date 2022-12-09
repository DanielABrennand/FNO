from torch import no_grad, squeeze, float,cat,zeros
from UtilityFunctions import LogitPercentConverter as L2P
import wandb
import simvue
import numpy as np

def BinaryClassiferTestingLoop(Device,DataLoader,Model,LossFn,Threshold = 0.8,Final = False):
    Threshold = L2P.ToLogit(Threshold)
    Size = len(DataLoader.dataset)
    Batches = len(DataLoader)
    Model.eval()
    TestLoss, Correct = 0, 0
    with no_grad():
        for sample in DataLoader:

            xx = sample['xx']
            yy = sample['yy']

            BatchSize = xx.shape[0]
            ImageSize = xx.shape[1]

            #Images = Data['image'].float().to(Device)
            #Modes = Data['mode'].to(Device)

            xx = xx.reshape((BatchSize,1,ImageSize,ImageSize)).float().to(Device)
            yy =yy.to(Device).float()

            #Outputs = squeeze(Model(Images))
            out = squeeze(Model(xx)).float()

            if Final:
                for pred in out:
                    wandb.log({'Final Test Output': pred})

            #TestLoss += LossFn(Outputs,Modes).item()
            TestLoss  += LossFn(out,yy).item()

            

            #Correct += (Outputs.argmax(1) == Modes).type(float).sum().item()
            Prediction = (out>Threshold)
            #Correct  += (out.argmax(1) == yy).type(float).sum().item()
            Correct += (Prediction == yy).sum().item()

            #wandb.log({'Testing Batch': Batch})
        TestLoss /= Batches
        Correct /= Size

    return TestLoss,Correct

def FNOTestingLoop(Device,DataLoader,Model,LossFn,Optimizer,T,Step,Final = False):
    Model.eval()
    test_l2_step = 0
    test_l2_full = 0
    AllPreds = []
    with no_grad():
        for sample in DataLoader:

            xx = sample['xx'].to(Device).float()
            yy = sample['yy'].to(Device).float()

            loss = 0
            batch_size = xx.shape[0]

            for t in range(0,T,Step):
                y = yy[...,t:t+Step]

                im = Model(xx)

                loss += LossFn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

                if t == 0: 
                    pred = im
                else:
                    pred = cat((pred, im), -1)

                xx = cat((xx[..., Step:], im), dim=-1)

            test_l2_step += loss.item()
            l2_full = LossFn(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
            test_l2_full += l2_full.item()

            if Final:
                pred = np.array(pred.cpu())
                if len(AllPreds) == 0:
                    AllPreds = pred
                else:
                    AllPreds = [AllPreds,pred]

    if Final:
        return test_l2_full, AllPreds
    else:       
        return test_l2_full