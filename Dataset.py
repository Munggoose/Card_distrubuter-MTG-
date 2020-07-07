import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
import os
import subprocess as subp
import typing

batch_size=8

path = './Magic_card/origin_data'

def resizepath(path,width=140,_height=200):
    trans = transforms.Compose([
        transforms.Resize((_height,width)),
    ])
    datas = torchvision.datasets.ImageFolder(root = path, transform= trans)
    for num, value in enumerate(datas):
        data, label =value

        if not os.path.exists('./train_data'):
            os.makedirs('./train_data')
            os.makedirs('./train_data/border_less')
            os.makedirs('./train_data/border')

        if(label == 0):
            data.save('train_data/border_less/%d_%d.jpeg'%(num,label))
        else:
            data.save('train_data/border/%d_%d.jpeg'%(num,label))


def load_data(path,width=140,_height=200):
    trans = transforms.Compose([
        transforms.Resize((_height,width)),
        transforms.ToTensor()
    ])
    data = torchvision.datasets.ImageFolder(root = path, transform= trans)
    #print('load_data',data)
    return data



def custom_loader(trg_set,batch_size_val):
    data_loader = torch.utils.data.DataLoader(dataset=trg_set,batch_size=batch_size_val, shuffle = True, drop_last = True, num_workers=2)
    #data_loader
    return data_loader


if __name__ == "__main__":
    resizepath(path)