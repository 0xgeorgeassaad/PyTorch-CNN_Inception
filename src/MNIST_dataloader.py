import torch
import torchvision
train_dataset = torchvision.datasets.MNIST(root='../data/',
                               train=True,
                               transform=torchvision.transforms.ToTensor(),
                               download=True)

test_dataset = torchvision.datasets.MNIST(root='../data/',
                              train=False,
                              transform=torchvision.transforms.ToTensor()
                              )

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=64,
                                          shuffle=False)