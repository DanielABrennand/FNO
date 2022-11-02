from torch import no_grad, squeeze, float
from UtilityFunctions import LogitPercentConverter as L2P
import wandb
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
