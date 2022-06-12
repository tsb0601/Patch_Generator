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

######################
## Testing Accuracy ##
######################
# test using a knn monitor
num_patches = 16

def test(net, memory_data_loader, test_data_loader, epoch):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
        
        
        

            data = torch.cat(data, dim=0) 
            data = data.cuda()
      
            z_proj, x_recon = net(data)
            
            feature = chunk_avg(z_proj, num_patches)
            
           
            #feature, _ = net(data.cuda(non_blocking=True))
            #feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        test_bar = tqdm(test_data_loader)
        for data, target in test_bar:
        
            data = torch.cat(data, dim=0) 
            
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            z_proj, x_recon = net(data)
            
            feature = chunk_avg(z_proj, num_patches)
            
            
         
            #feature, _ = net(data)
            #feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, 200, 0.1)
            #print(pred_labels.shape)
            
            
            
            
            

            total_num += data.size(0)/num_patches
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
            
            #print(target[:10])
            #print(pred_labels[:,0][:10])
            

            
            test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}%'.format(epoch, 500, total_top1 / total_num * 100))

    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels 



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
    if not os.path.exists("./images"):
        os.makedirs("./images")
    plt.savefig(f"./images/{epoch:07d}_{name}.png", bbox_inches="tight")
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
        
        for z in z_list:
            #print("shape of z is:", z.shape, z_avg.shape)
            z_sim += F.cosine_similarity(z, z_avg, dim=1).mean()
            
        z_sim = z_sim / len(z_list)
        z_sim_out = z_sim.clone().detach()
        return -z_sim, z_sim_out
######################
## Prepare Training ##
######################

#Get Dataset
train_dataset = load_dataset("cifar10", "cifar_patch", use_baseline=False, train=True, into_patches=True)
dataloader = DataLoader(train_dataset, batch_size=1000, shuffle=True, drop_last=True,num_workers=16)


test_data = load_dataset("cifar10", "cifar_patch", use_baseline=False, train=False, into_patches=True)
test_loader = DataLoader(test_data, batch_size=2000, shuffle=True, num_workers=16, pin_memory=True)


# Define models and optimizers
lr = 0.0001
epochs = 500

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
    
    
net = autoencoder()
net.cuda()
opt = optim.Adam(net.parameters(), lr, betas=(0.5, 0.999))
contractive_loss = Similarity_Loss()
reconstruction_loss = torch.nn.MSELoss()



##############
## Training ##
##############
def main():
    for epoch in range(epochs):
    
        test(net, dataloader, test_loader, epoch)
        
        for step, (data, label) in tqdm(enumerate(dataloader)):
            net.zero_grad()
            opt.zero_grad()
            
            

            data = torch.cat(data, dim=0) 
            data = data.cuda()
            
            data_list = data.chunk(num_patches, dim=0)
            
            data_list = torch.stack(data_list,dim=0)
            #print(len(data_list))

            z_proj, x_recon = net(data)
            
            #print(z_proj.shape)
            z_list = z_proj.chunk(num_patches, dim=0)
            z_avg = chunk_avg(z_proj, num_patches)

            #Contractive Loss
            loss_contract, _ = contractive_loss(z_list, z_avg)
            #Reconstruction Loss
            loss_recon = reconstruction_loss(data, x_recon)
            #Total Loss
            loss = loss_contract + 10*loss_recon
            
            x_recon_list = x_recon.chunk(num_patches, dim=0)
            x_recon_list = torch.stack(x_recon_list,dim=0)
            
            

            #Update
            loss.backward()
            opt.step()

            if step%50 == 0:
                show(vutils.make_grid(x_recon_list[:, 0, :, :, :], padding=2, normalize=True, nrow =4), epoch, str(step)+"_cifar10_recon")
                show(vutils.make_grid(data_list[:, 0, :, :, :], padding=2, normalize=True, nrow =4), epoch, str(step)+"_cifar10")
                
        print("At epoch:", epoch, "loss similarity is", loss_contract.item(), ", loss reconstruction is", loss_recon.item(), ",totally:", loss.item())
        

                


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
