############
## Import ##
############
import argparse
import torch.nn as nn
import torch.optim as optim
import os
import yaml
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
from model.model import autoencoder
from dataset.datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torchvision.transforms.functional as FF
from tqdm import tqdm
import torch
import torchvision.utils as vutils
from torchvision.datasets import CIFAR10
from loss import TotalCodingRate, SimCLR, Z_loss


######################
## Testing Accuracy ##
######################
# test using a knn monitor
num_patches = 16







def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size


def knn(train_features, train_labels, test_features, test_labels):
    """Perform k-Nearest Neighbor classification using cosine similaristy as metric.
    Options:
        k (int): top k features for kNN
    
    """
    sim_mat = train_features @ test_features.T
    topk = sim_mat.topk(k=11, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = topk_pred.mode(0).values.detach()
    acc = compute_accuracy(test_pred.numpy(), test_labels.numpy())
    print("kNN: {}".format(acc))
    return acc


 
def test(net, train_loader, test_data_loader, epoch):
    
    
    train_z_list, train_y_list, test_z_list, test_y_list = [], [], [], []
    train_x_list, train_logits_list, test_x_list, test_logits_list = [], [], [], []
    with torch.no_grad():
        n_iter = max(1,1)
        for i in range(n_iter):
            #print('collect train features and labels')
            for x, y in train_loader:
                x = x.to(device)
             
                z, _, _, _ = net(x)
                z = z.detach().cpu()
              
                train_z_list.append(z)
                if i==0:
                    train_y_list.append(y)
                    train_x_list.append(x.detach().cpu())
            #print('collect test features and labels')
            for x, y in test_loader:
                x = x.to(device)
               
                z, _ , _, _= net(x)
                z = z.detach().cpu()
              
                test_z_list.append(z)
                if i==0:
                    test_y_list.append(y)
                    test_x_list.append(x.detach().cpu())
            
    train_features, train_labels, test_features, test_labels = torch.cat(train_z_list,dim=0), torch.cat(train_y_list,dim=0), torch.cat(test_z_list,dim=0), torch.cat(test_y_list,dim=0)
    
    #train_features, train_labels, test_features, test_labels = torch.cat(train_z_list,dim=0), torch.cat(train_y_list,dim=0), torch.cat(test_z_list,dim=0), torch.cat(test_y_list,dim=0)
    
    
    acc = knn(train_features, train_labels, test_features, test_labels)
    
    return acc
    
    
    
    
    
    


dir_name = "./images_patchvae"

#####################
## Helper Function ##
#####################
def show(imgs, epoch, name):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = FF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    plt.savefig(f"./images_patchvae/{epoch:07d}_{name}.png", bbox_inches="tight")
    plt.close()

def chunk_avg(x,n_chunks=2,normalize=False):
    x_list = x.chunk(n_chunks,dim=0)
    x = torch.stack(x_list,dim=0)
    if not normalize:
        return x.mean(0)
    else:
        return F.normalize(x.mean(0),dim=1)


class Similarity_Loss(nn.Module):
    def __init__(self, ):
        super().__init__()
        pass

    def forward(self, z_list, z_avg):
        z_sim = 0
        
        #print("number of patches is:", len(z_list))
        
        num_patch = len(z_list)
        choices = np.random.choice(num_patch, 2, replace=False)
        I, II = choices[0], choices[1]
        
        #I, II = 0, 3
        
        
        #for z in z_list:
            #print("shape of z is:", z.shape, z_avg.shape)
        #    z_sim += F.cosine_similarity(z, z_avg, dim=1).mean()
            
        #z_sim = z_sim / len(z_list)
        
        z_sim = F.cosine_similarity(z_list[I], z_list[II], dim=1).mean()
        
        z_sim_out = z_sim.clone().detach()
        
        #z_sim = 30*z_sim - (criterion(z_list[I]) + criterion(z_list[II]))/2  
    
        
        
        
        return -z_sim, z_sim_out
######################
## Prepare Training ##
######################

#Get Dataset
train_dataset = load_dataset("cifar10", "cifar_patch", use_baseline=False, train=True, into_patches=True, add_gaussian=False, num_patch = 16)
dataloader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True,num_workers=16)


memory_dataset = load_dataset("cifar10", "cifar_patch", use_baseline=False, train=True, into_patches=True, add_gaussian=False, num_patch = 1)
memory_loader = DataLoader(memory_dataset, batch_size=1000, shuffle=True, drop_last=True,num_workers=16)



test_data = load_dataset("cifar10", "cifar_patch", use_baseline=False, train=False, into_patches=True, add_gaussian=False, num_patch = 1)
test_loader = DataLoader(test_data, batch_size=2000, shuffle=True, num_workers=16, pin_memory=True)


# Define models and optimizers
lr = 0.0001
epochs = 500

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
    
    
net = autoencoder()


net = nn.DataParallel(net)
net.cuda()


opt = optim.Adam(net.parameters(), lr, betas=(0.5, 0.999))
opt_decoder = optim.Adam(net.module.decoder.parameters(), lr, betas=(0.5, 0.999))


contractive_loss = Similarity_Loss()
reconstruction_loss = torch.nn.MSELoss()

criterion = TotalCodingRate(eps=0.2)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        
        #print(tensor.size())
    
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
        


add_Gaussian = AddGaussianNoise(0, 0.05)

##############
## Training ##
##############




def kld(mu, log_var, kld_weight = 1) -> dict:
    """
    Computes the VAE loss function.
    KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
    :param args:
    :param kwargs:
    :return:
    """


    kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    
    return kld_loss


def main():
    for epoch in range(epochs):
        test(net, memory_loader, test_loader, epoch)
    
        
        for step, (data, label) in tqdm(enumerate(dataloader)):
            net.zero_grad()
            opt.zero_grad()
            
            

            data = torch.cat(data, dim=0) 
            data = data.cuda()
           
           
            z_proj, x_recon, mu, log_var = net(data)
            
            
            z_proj_noised = add_Gaussian(z_proj.detach().cpu()).cuda()
            
            
           
            data_list = data.chunk(num_patches, dim=0)
            data_list = torch.stack(data_list,dim=0)
          

            #print(z_proj.shape)
            z_list = z_proj.chunk(num_patches, dim=0)
            z_avg = chunk_avg(z_proj, num_patches)
            
            
            z_list = mu.chunk(num_patches, dim=0)
            z_avg = chunk_avg(mu, num_patches)
            
            

            #Contractive Loss
            loss_contract, _ = contractive_loss(z_list, z_avg)
            
            
            
            #Reconstruction Loss
            loss_recon = reconstruction_loss(data, x_recon)
            
            
            kl_divergence = kld(mu, log_var)
            #Total Loss
            loss = loss_contract + 30*(loss_recon + 0.000125*kl_divergence)
            
            #loss = loss_recon + 0.00025*kl_divergence
            #loss = loss_contract 
            #loss = loss_recon
            #
            
            x_recon_list = x_recon.chunk(num_patches, dim=0)
            x_recon_list = torch.stack(x_recon_list,dim=0)
            
           
            
            
            #Update
            loss.backward()
            opt.step()
            
            
            #Update Gaussian Z
            """
           
            net.module.decoder.zero_grad()
            opt_decoder.zero_grad()
            
            
            #print(len(data_list))
            
            #print(z_proj_noised.shape)
            
            x_recon_noised = net.module.decoder(z_proj_noised)
            loss = reconstruction_loss(data, x_recon_noised)
            
            x_recon_list_Gau = x_recon_noised.chunk(num_patches, dim=0)
            x_recon_list_Gau = torch.stack(x_recon_list_Gau,dim=0)
            
            #Update
            loss.backward()
            opt_decoder.step()
            """
            
            
            
            """
            
            z_proj, x_recon = net(data)
            
            data_list = data.chunk(2, dim=0)
            data, data_withGau = data_list[0], data_list[1]
            
            data_list = data.chunk(num_patches, dim=0)
            data_list = torch.stack(data_list,dim=0)
            
            datawithGau_list = data_withGau.chunk(num_patches, dim=0)
            datawithGau_list = torch.stack(datawithGau_list,dim=0)
            
            
            z = z_proj.chunk(2, dim = 0)
            z_proj = z[0]
            
            #print(z_proj.shape)
            z_list = z_proj.chunk(num_patches, dim=0)
            z_avg = chunk_avg(z_proj, num_patches)

            #Contractive Loss
            loss_contract, _ = contractive_loss(z_list, z_avg)
            #Reconstruction Loss
            
            
            
            x_recon_list = x_recon.chunk(2, dim=0)
            x_recon_noGau = x_recon_list[0]
            x_recon_Gau = x_recon_list[1]
            
            loss_recon = reconstruction_loss(data, x_recon_Gau) + reconstruction_loss(data, x_recon_noGau)
            #Total Loss
            loss = loss_contract + loss_recon
            
            x_recon_list = x_recon_Gau.chunk(num_patches, dim=0)
            x_recon_list = torch.stack(x_recon_list,dim=0)
            
            x_recon_list_noGau = x_recon_noGau.chunk(num_patches, dim=0)
            x_recon_list_noGau = torch.stack(x_recon_list_noGau,dim=0)
            """
            
            

            

            if step%50 == 0:
                show(vutils.make_grid(x_recon_list[:, 0, :, :, :], padding=2, normalize=True, nrow = 4), epoch, str(step)+"_cifar10_recon")
                
                #show(vutils.make_grid(x_recon_list_Gau[:, 0, :, :, :], padding=2, normalize=True, nrow =4), epoch, str(step)+"_cifar10_recon_withGau")
                
                show(vutils.make_grid(data_list[:, 0, :, :, :], padding=2, normalize=True, nrow =4), epoch, str(step)+"_cifar10")
                
                #show(vutils.make_grid(datawithGau_list[:, 0, :, :, :], padding=2, normalize=True, nrow =4), epoch, str(step)+"_cifar10_gaussian")
            
                print(np.linalg.norm(z_proj[0].cpu().numpy(), 2), np.linalg.norm(mu[0].cpu().numpy(), 2))
                
                
        print("At epoch:", epoch, "loss similarity is", loss_contract.item(), ", loss reconstruction is", loss_recon.item(), ",kld:", (-0.00025*kl_divergence).item())
        #print("At epoch:", epoch, "loss similarity is", loss_contract.item())
        
        
        

                


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
