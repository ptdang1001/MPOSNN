import sys

import torch
from torch import nn

# import torchmetrics
from pytorch_lightning import LightningModule


class LitFCN(LightningModule):
    def __init__(self,dim_in):
        super().__init__()
        #self.save_hyperparameters()
        #self.accuracy = torchmetrics.Accuracy()
        self.dim_in = dim_in

        
        self.predictor=None
        if self.dim_in<=5:
            self.predictor=nn.Sequential(
                nn.Linear(self.dim_in,2),
                nn.Tanhshrink(),
                nn.Linear(2,1),
            )
        else:
            layers=[]
            n_in,n_out=self.dim_in,int(self.dim_in//2)
            while n_out>=1:
                if n_out!=1:
                    layers.extend([nn.Linear(n_in,n_out), nn.Tanhshrink()])
                else:
                    layers.extend([nn.Linear(n_in,n_out)])
                n_in,n_out=n_out,int(n_out//2)

            self.predictor=nn.Sequential(*layers)


    def forward(self,x):
        flux=self.predictor(x)
        return flux

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=8e-2)
        #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.0001)
        #return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        return {"optimizer": optimizer}



    def training_step(self, batch,batch_idx):
        x, y = batch
        y=y.unsqueeze(1)
        x = x.view(x.size(0), -1)

        y_hat = self.forward(x)
        loss_func=nn.MSELoss()
        loss = loss_func(y_hat, y)
        '''
        l1_lambda = 0.2
        l2_lambda = 0.0
        l1_norm = sum(torch.linalg.norm(p, 1) for p in self.parameters())
        l2_norm = sum(torch.linalg.norm(p,2) for p in self.parameters())
        loss = loss + l1_lambda * l1_norm+l2_lambda*l2_norm
        '''
        self.log('train_loss', loss)
        return {"loss":loss}

    def validation_step(self,batch,batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y=y.unsqueeze(1)

        y_hat = self.forward(x)
        loss_func = nn.MSELoss()
        loss = loss_func(y_hat, y)
        '''
        l1_lambda = 0.2
        l2_lambda = 0.0
        l1_norm = sum(torch.linalg.norm(p, 1) for p in self.parameters())
        l2_norm = sum(torch.linalg.norm(p, 2) for p in self.parameters())
        loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
        '''

        self.log('val_loss', loss)
        return {"loss":loss}

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x = x.view(x.size(0), -1)
        y = y.unsqueeze(1)

        y_hat = self.forward(x)
        loss_func = nn.MSELoss()
        loss = loss_func(y_hat, y)

        '''
        l1_lambda = 0.2
        l2_lambda = 0.0
        l1_norm = sum(torch.linalg.norm(p, 1) for p in self.parameters())
        l2_norm = sum(torch.linalg.norm(p, 2) for p in self.parameters())
        loss = loss + l1_lambda * l1_norm + l2_lambda * l2_norm
        '''

        self.log_dict({"test_loss": loss})
        return loss

    def predict_step(self,data_batch,batch_idx=1):
        x=data_batch
        x=x.view(x.size(0),-1)
        y_hat=self.forward(x)
        return y_hat
