import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

import torchmetrics


# Based on 1D GAN signal generation from https://github.com/LixiangHan/GANs-for-1D-Signal

clip_value = 0.01


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # input 1824 
            nn.Conv1d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 912
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 456
            nn.Conv1d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 228
            nn.Conv1d(256, 512, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # state size 114
            nn.Conv1d(512, 1, kernel_size=114, stride=1, padding=0, bias=False),
        )

    def forward(self, x, y=None):
        x = self.model(x)
        return x


class Generator(nn.Module):
    def __init__(self, nz):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose1d(nz, 512, 114, 1, 0, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(True),

            nn.ConvTranspose1d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),

            nn.ConvTranspose1d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),

            nn.ConvTranspose1d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(True),

            nn.ConvTranspose1d(64, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.model(x)
        return x
      
      
# model wrapper
class Net(pl.LightningModule):
  def __init__(self, nz, lr):
    super(Net, self).__init__()
    
    self.nz = nz
    self.lr = lr
    
    # init netD and netG
    self.netD = Discriminator()
    self.netD.apply(weights_init)

    self.netG = Generator(nz)
    self.netG.apply(weights_init)

    # self.val_z = torch.randn(64, nz)
    self.val_z = torch.randn(64, nz, 1)

    # used for visualizing training process
    # self.fixed_noise = torch.randn(16, nz, 1)

    # optimizers
    self.optimizerD = torch.optim.RMSprop(self.netD.parameters(), lr=lr)
    self.optimizerG = torch.optim.RMSprop(self.netG.parameters(), lr=lr)

    self.accuracy = torchmetrics.Accuracy()
    self.val_accuracy = torchmetrics.Accuracy()

  def forward(self, z):
    return self.netG(z)

  def configure_optimizers(self):
    return [self.optimizerG, self.optimizerD], []
  
  def training_step(self, batch, batch_idx, optimizer_idx):
    real, y = batch
    real = real.unsqueeze(1)
    # train discriminator
    if optimizer_idx == 1:
      noise = torch.randn(real.size(0), self.nz, 1)
      noise = noise.type_as(real)
      fake = self.netG(noise)
      fake = fake.type_as(real)
      loss_D = -torch.mean(self.netD(real)) + torch.mean(self.netD(fake))
      for p in self.netD.parameters():
          p.data.clamp_(-clip_value, clip_value)
      self.log("loss_D", loss_D)
      return loss_D

    # train generator
    if optimizer_idx == 0:
        # training netG
        noise = torch.randn(real.size(0), self.nz, 1)
        noise = noise.type_as(real)

        fake = self.netG(noise)
        fake = fake.type_as(real)
        loss_G = -torch.mean(self.netD(fake))
        self.log("loss_G", loss_G)
        return loss_G


