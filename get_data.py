from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms



def get_CIFAR10(path='./data', batch_size=32):
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
    root = path + '/cifar10'
    train_data = CIFAR10(root=root, train=True, transform=transform, download=True)
    trainloader = DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    test_data = CIFAR10(root=root, train=False, transform=transform, download=True)
    testloader = DataLoader(test_data, batch_size=batch_size,
                                            shuffle=False, num_workers=2)
    
    return train_data, trainloader, test_data, testloader

