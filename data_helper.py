import torch
import torchvision
from torchvision import transforms


def get_celebA_dataset(batch_size, image_size, environment):
    if environment == 'nuri':
        image_path = "../data/"
    else:
        image_path = '../../dataset/'
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


def get_ffhq_thumbnails(batch_size, image_size, environment):
    if environment == 'nuri':
        image_path = "../data/"
    else:
        image_path = '../../dataset/'
    transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(image_size),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])
    train_dataset = torchvision.datasets.ImageFolder(image_path + 'FFHQ', transformation)
    if environment == 'nuri':
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24, pin_memory=True)
    return train_loader, None


def get_ffhq_thumbnails_raw_images(image_size, environment):
    if environment == 'nuri':
        image_path = "../data/"
    else:
        image_path = '../../dataset/'
    transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.CenterCrop(image_size),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5)),
    ])
    dataset = torchvision.datasets.ImageFolder(image_path + 'ffhq/thumbnails128x128', transformation)
    return dataset


def get_cifar_dataset(batch_size, img_size, environment):
    if environment == 'nuri':
        image_path = "../data/"
    else:
        image_path = '../../dataset/'
    dataset = torchvision.datasets.CIFAR10(root=image_path + 'cifar',  download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize((img_size, img_size)),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                           ]))
    if environment == 'nuri':
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return train_loader, None


def get_e_mnist(batch_size, image_size, environment):
    if environment == 'nuri':
        image_path = "../data/"
    else:
        image_path = '../../dataset/'
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


def get_fashion_mnist(batch_size, image_size, environment):
    if environment == 'nuri':
        image_path = "../data/"
    else:
        image_path = '../../dataset/'
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


def get_mnist(batch_size, image_size, environment):
    if environment == 'nuri':
        image_path = "../data/"
    else:
        image_path = '../../dataset/'
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


def get_data(dataset, batch_size, image_size, environment):
    if dataset == 'ffhq':
        return get_ffhq_thumbnails(batch_size, image_size, environment)
    elif dataset == 'cifar':
        return get_cifar_dataset(batch_size, image_size, environment)
    elif dataset == 'emnist':
        return get_e_mnist(batch_size, image_size, environment)
    elif dataset == 'mnist':
        return get_mnist(batch_size, image_size, environment)
    elif dataset == 'mnist_fashion':
        return get_fashion_mnist(batch_size, image_size, environment)
    else:
        return get_celebA_dataset(batch_size, image_size, environment)
