from torch import squeeze,cat
import wandb
def BinaryClassiferTrainingLoop(Device,DataLoader,Model,LossFn,Optimizer):
    Batches = len(DataLoader)
    Model.train()
    Running_Loss = 0
    for sample in DataLoader:

        xx = sample['xx'].float()
        yy = sample['yy'].float()

        BatchSize = xx.shape[0]
        ImageSize = xx.shape[1]

        #Images = Data['image'].float().to(Device)
        #Modes = Data['mode'].to(Device)

        xx = xx.reshape((BatchSize,1,ImageSize,ImageSize)).float().to(Device)
        yy = yy.to(Device).float()

        Optimizer.zero_grad()

        #Outputs = squeeze(Model(Images))
        out = squeeze(Model(xx)).float()

        #Loss = LossFn(Outputs, Modes)
        Loss = LossFn(out,yy)


        Loss.backward()
        Optimizer.step()

        Running_Loss += Loss.item()
        #wandb.log({'Training Batch': Batch})
            
    return Running_Loss/Batches  



def FNOTrainingLoop(Device,DataLoader,Model,LossFn,Optimizer,T,Step):
    Model.train()
    train_l2_step = 0
    train_l2_full = 0
    for sample in DataLoader:

        xx = sample['xx'].to(Device).float()
        yy = sample['yy'].to(Device).float()

        loss = 0
        batch_size = xx.shape[0]

        for t in range(0,T,Step):
            y = yy[...,t:t+Step]

            print(xx.shape)
            
            im = Model(xx)

            loss += LossFn(im.reshape(batch_size, -1), y.reshape(batch_size, -1))

            if t == 0:
                pred = im
            else:
                pred = cat((pred, im), -1)

            xx = cat((xx[..., Step:], im), dim=-1)

        train_l2_step += loss.item()
        l2_full = LossFn(pred.reshape(batch_size, -1), yy.reshape(batch_size, -1))
        train_l2_full += l2_full.item()

        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()
    return train_l2_full