import torch
import torchvision
from torchvision import transforms


def get_celebA_dataset(batch_size, image_size):
    image_path = "../data/"
    transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.ImageFolder(image_path + 'celeba', transformation)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    train_indices, test_indices = indices[:10000], indices[200000:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader


def get_ffhq_thumbnails(batch_size, image_size):
    image_path = "../data/"
    transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.ImageFolder(image_path + 'ffhq/thumbnails128x128', transformation)
    num_train = len(train_dataset)
    indices = list(range(num_train))
    train_indices, test_indices = indices[:60000], indices[60000:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=test_sampler)
    return train_loader, test_loader
