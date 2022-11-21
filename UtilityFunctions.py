import numpy as np
import h5py
from PIL import Image
from torch import norm, mean, sum
class LogitPercentConverter:
    #Takes a logit and returns the corresponding probability (0 to 1) or vice versa
    @staticmethod
    def ToPercent(Logit):
        return 1/(1+np.exp(-Logit))
    @staticmethod
    def ToLogit(Percent):
        return np.log((Percent)/(1-Percent))

class H5ToNumpy:
    @staticmethod
    def ConvertH5(FilePath,Resolution = 0,CutOff = 65536):
        #Takes an H5 file and returns a numpy array of all of the frame data in the form [:,:,frame#], (dtype = uint8 for most rbb cameraas)
        #Also performs an image transformation to a given resolution (if 0 native resoltion is to be used)
        #Also only converts up to a cutoff point
        F = h5py.File(FilePath)
        AllKeys = list(F.keys())
        NumFrames = min(len(AllKeys)-1,CutOff) #-1 for the time dataset
        if not Resolution:
            FrameRes = np.array(F[AllKeys[0]].shape)
        else:
            FrameRes = (Resolution,Resolution)
        Data = np.zeros([FrameRes[0],FrameRes[1],NumFrames])
        for n,Key in enumerate(AllKeys):
            if Key != "time":
                img = np.array(F[Key])
                if Resolution:
                    img = Image.fromarray(img).resize((Resolution,Resolution))
                Data[:,:,n] = img
        return Data

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return mean(all_norms)
            else:
                return sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return mean(diff_norms/y_norms)
            else:
                return sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)