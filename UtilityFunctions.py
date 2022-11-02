import numpy as np
import h5py
from PIL import Image
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
