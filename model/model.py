import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_mimicry.nets.dcgan import dcgan_base

from torchvision.models import resnet18, resnet34, resnet50

def Resnet18CIFAR():
    backbone = resnet18()
    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    backbone.fc = nn.Identity()
    return backbone


class Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.normalize(x)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        


class GBlock_start(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=False,
                 upsample=False,
                 num_repeat=2
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.upsample = upsample
        self.num_repeat = num_repeat

        if upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

        if self.hidden_channels > 0:

            layers = []
            for i in range(self.hidden_channels):
                layers.extend(
                    [
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )

            self.identity = nn.Sequential(*layers)

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        if self.hidden_channels > 0:
            h = self.identity(h)
        return h


class GBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=False,
                 upsample=False,
                 num_repeat=2
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.upsample = upsample
        self.num_repeat = num_repeat

        if upsample:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(True)

        if self.hidden_channels > 0:

            layers = []
            for i in range(self.hidden_channels):
                layers.extend(
                    [
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2, inplace=True),
                    ]
                )

            self.identity = nn.Sequential(*layers)

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        if self.hidden_channels > 0:
            h = self.identity(h)
        return h


class DBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=0,
                 downsample=False,
                 BN=True,
                 shortcut=False
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.downsample = downsample

        stride = 2 if self.downsample else 1
        padding = 1 if self.downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

        if BN:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Sequential()

        self.relu = nn.LeakyReLU(0.2)

        if self.hidden_channels > 0:

            layers = []
            for i in range(self.hidden_channels):
                layers.extend(
                    [
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2),
                    ]
                )

            self.identity = nn.Sequential(*layers)
        else:
            self.identity = nn.Sequential()

        self.shortcut = nn.Sequential()
        if shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        h = self.relu(self.bn1(self.conv1(x)))
        if self.hidden_channels > 0:
            h = self.identity(h)
        h += self.shortcut(x)
        h = self.relu(h)

        return h


class DBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=0,
                 downsample=False,
                 BN=True,
                 shortcut=False
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.downsample = downsample

        stride = 2 if self.downsample else 1
        padding = 1 if self.downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

        if BN:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.Sequential()

        self.relu = nn.LeakyReLU(0.2)

        if self.hidden_channels > 0:

            layers = []
            for i in range(self.hidden_channels):
                layers.extend(
                    [
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2),
                    ]
                )

            self.identity = nn.Sequential(*layers)
        else:
            self.identity = nn.Sequential()

        self.shortcut = nn.Sequential()
        if shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=4, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        h = self.relu(self.bn1(self.conv1(x)))
        if self.hidden_channels > 0:
            h = self.identity(h)
        h += self.shortcut(x)
        h = self.relu(h)

        return h


class GeneratorCIFAR(dcgan_base.DCGANBaseGenerator):

    def __init__(self, nz=128, ngf=64, iden=None, bottom_width=4, **kwargs):
        super().__init__(nz=nz, ngf=ngf, bottom_width=bottom_width, **kwargs)

        self.main = nn.Sequential(
            GBlock_start(nz, ngf * 8, hidden_channels=iden, upsample=False),
            GBlock(ngf * 8, ngf * 4, hidden_channels=iden, upsample=True),
            #GBlock(ngf * 4, ngf * 2, hidden_channels=iden, upsample=True),
            
        )
        self.end = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, 3, 4, 2, 2, bias=False),
            nn.Tanh()
        )


    def forward(self, x):
        output = self.main(x.view(x.shape[0], -1, 1, 1))
        
        #print(output.shape)
        output = self.end(output)
        return output


class DiscriminatorCIFAR(dcgan_base.DCGANBaseDiscriminator):

    def __init__(self, nz=128, ndf=64, iden=0, shortcut=False, **kwargs):
        super().__init__(ndf=ndf, **kwargs)

        self.nz = nz
        self.main = nn.Sequential(
            DBlock(3, ndf, hidden_channels=iden, downsample=True, BN=False, shortcut=shortcut),
            DBlock(ndf, ndf * 2, hidden_channels=iden, downsample=True, shortcut=shortcut),
            DBlock(ndf * 2, ndf * 4, hidden_channels=iden, downsample=True, shortcut=shortcut),
        )

        self.end = nn.Sequential(
            nn.Conv2d(ndf * 4, nz, 4, 1, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, x):
    
      

        print(x.shape)
        h = self.main(x)
        print(h.shape)
        
        return self.end(h)

class encoder(nn.Module):
     def __init__(self,z_dim=128,normalize=True,norm_p=2):
        super().__init__()
        feature_dim = 512
        backbone = Resnet18CIFAR()
        self.backbone = backbone
        self.norm_p = norm_p
        self.pre_feature = nn.Sequential(nn.Linear(feature_dim,4096),
                                         nn.BatchNorm1d(4096),
                                         nn.ReLU()
                                        )
        self.normalize = normalize
        self.projection = nn.Linear(4096,z_dim)
     def forward(self, x):
        feature = self.backbone(x)
        if self.normalize:
            feature = self.pre_feature(feature)
            z = F.normalize(self.projection(feature),p=self.norm_p)
            return feature, z
        else:
            return self.projection(self.drop(self.pre_feature(feature)))

        


class autoencoder(dcgan_base.DCGANBaseDiscriminator):

    def __init__(self, latent_dim=4096, nz=256, ndf=64, iden=0, **kwargs):
        super().__init__(ndf=ndf, **kwargs)
        #self.encoder = DiscriminatorCIFAR(nz=latent_dim, iden=iden, shortcut=True)
        self.encoder = encoder(z_dim = nz)
        self.decoder = GeneratorCIFAR(nz=nz, iden=iden)
        
        self.fc_mu = nn.Linear(latent_dim, nz)
        self.fc_var = nn.Linear(latent_dim, nz)

    
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        z, z_proj = self.encoder(x)
        
        mu = self.fc_mu(z)
        log_var = self.fc_var(z)
        
        z_proj = self.reparameterize(mu, log_var)
        
        x_recon = self.decoder(z_proj)

        return z_proj, x_recon, mu, log_var
