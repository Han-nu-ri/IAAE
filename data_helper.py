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
    #train_indices, test_indices = indices[:10000], indices[200000:]
    train_indices, test_indices = indices, indices[200000:]
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

def get_ffhq_thumbnails_tensorflow(batch_size, image_size):
    # TODO: 여기 구현하기
    return None


def get_cifar_dataset(batch_size, img_size):
    image_path = "../data/"
    dataset = torchvision.datasets.CIFAR10(root=image_path + 'cifar',  download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize((img_size, img_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, None


def get_e_mnist(batch_size, image_size):
    image_path = "../data/"
    train_set = torchvision.datasets.EMNIST(
        root=image_path + 'emnist',
        split='balanced',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader, None


def get_fashion_mnist(batch_size, image_size):
    image_path = "../data/"
    train_set = torchvision.datasets.FashionMNIST(
        root=image_path + 'fashion_mnist',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader, None


def get_mnist(batch_size, image_size):
    image_path = "../data/"
    train_set = torchvision.datasets.MNIST(
        root=image_path + 'mnist',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Grayscale(3),
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_loader, None
