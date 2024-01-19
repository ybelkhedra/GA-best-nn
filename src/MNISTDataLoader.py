#import MNIST dataset from pytorch
import torch
from torchvision import datasets, transforms

class MNISTDataLoader:
    def __init__(self, batch_size, shuffle, num_workers):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def load(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,)),
                                        ])
        trainset = datasets.MNIST('data/', download=True, train=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                  num_workers=self.num_workers)

        testset = datasets.MNIST('data/', download=True, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size, shuffle=self.shuffle,
                                                 num_workers=self.num_workers)

        return trainloader, testloader
    
if __name__ == '__main__':
    trainloader, testloader = MNISTDataLoader(64, True, 0).load()
