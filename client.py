import flwr as fl
from ecg import load_data
from model import Net
import pytorch_lightning as pl
from collections import OrderedDict
import numpy as np


import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

def check_weights_dict(weigths_dict):
    a = weigths_dict.copy()
    for k, v in a.items():
        if v.shape == torch.Size([0]):
            del weigths_dict[k]
    return weigths_dict

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.generator = model.netG
        self.discriminator = model.netD
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # leghts of NN parameters to send and receive
        self.g_w_l = len(self.generator.state_dict().items())
        self.d_w_l = len(self.discriminator.state_dict().items())


    # def get_parameters(self):
    #     generator_params = _get_parameters(self.model.netG)
    #     discriminator_params = _get_parameters(self.model.netD)
    #     return generator_params + discriminator_params

    # def set_parameters(self, parameters):
    #     _set_parameters(self.model.netG, parameters)
    #     _set_parameters(self.model.netD, parameters)
    
    def get_parameters(self):
        g_par = np.array([val.cpu().numpy()
                          for _, val in self.generator.state_dict().items()], dtype=object)
        d_par = np.array([val.cpu().numpy()
                          for _, val in self.discriminator.state_dict().items()], dtype=object)
        parameters = np.concatenate([g_par, d_par], axis=0)
        return parameters

    def set_parameters(self, parameters):
        # generator
        g_par = parameters[:self.g_w_l].copy()
        params_dict = zip(self.generator.state_dict().keys(), g_par)
        g_state_dict = OrderedDict({k: torch.Tensor(v)
                                   for k, v in params_dict})
        # discriminator
        d_par = parameters[self.g_w_l:int(self.g_w_l+self.d_w_l)].copy()
        params_dict = zip(self.discriminator.state_dict().keys(), d_par)
        d_state_dict = OrderedDict({k: torch.Tensor(v)
                                   for k, v in params_dict})
       
        # checking for null weights
        g_state_dict = check_weights_dict(g_state_dict)
        d_state_dict = check_weights_dict(d_state_dict)
        # assigning weights
        self.generator.load_state_dict(g_state_dict, strict=True)
        self.discriminator.load_state_dict(d_state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=0)
        trainer.fit(self.model, self.train_loader)

        return self.get_parameters(), 55000, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        # trainer = pl.Trainer(progress_bar_refresh_rate=0)
        # results = trainer.test(self.model, self.val_loader)
        # loss = results[0]["test_loss"]
        # skip evaluation, return dummy value
        loss = 0.9

        return loss, 10000, {"loss": loss}


# def _get_parameters(model):
#     return [val.cpu().numpy() for _, val in model.state_dict().items()]


# def _set_parameters(model, parameters):
#     params_dict = zip(model.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     model.load_state_dict(state_dict, strict=True)


def main() -> None:
    # Model and data
    model = Net(100, lr=1e-3)
    train_loader, val_loader = load_data(data_dir="./data/", batch_size=32)

    # Flower client
    client = FlowerClient(model, train_loader, val_loader)
    fl.client.start_numpy_client("[::]:8080", client)


if __name__ == "__main__":
    main()