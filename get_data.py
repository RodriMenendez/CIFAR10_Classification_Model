from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class CIFAR10():
    def __init__(self, path='./data', batch_size = 32):
        super().__init__()
        self.path = path
        self.batch_size = batch_size

    def download_data(self):
        """
        Downloads CIFAR10 train and test dataset, and createstheir respective trainloaders
        Returns: train_data, trainloader, test_data, testloader
        """
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_data = CIFAR10(root=self.path+'/cifar10', train=True, transform=transform, download=True)
        trainloader = DataLoader(train_data, batch_size=self.batch_size,
                                                shuffle=True, num_workers=2)

        test_data = CIFAR10(root=self.path+'/cifar10', train=False, transform=transform, download=True)
        testloader = DataLoader(test_data, batch_size=self.batch_size,
                                                shuffle=False, num_workers=2)
        
        return train_data, trainloader, test_data, testloader

