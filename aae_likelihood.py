import argparse
import numpy as np
import pandas as pd
import data_helper
import prior_factory
import torch
import aae_for_various_prior
import time
import wandb


_start_time = time.time()
def tic():
    global _start_time
    _start_time = time.time()


def tac(rounding=True):
    if rounding:
        t_sec = round(time.time() - _start_time)
    else:
        t_sec = time.time() - _start_time
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour, t_min) = divmod(t_min,60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


def train_z_with_decoder(decoder, latent_dim, dataset, image_size):
    z_list = []
    decoder.eval()
    if dataset == 'ffhq32':
        dataset = data_helper.get_ffhq_thumbnails_raw_images(image_size)
    else:
        dataset = data_helper.get_e_mnist_raw_images(image_size)

    epoch = 1000
    count = 0
    tic()
    for each_real_image in dataset:
        z = torch.zeros([1, latent_dim], requires_grad=True).cuda()
        z = torch.nn.Parameter(z)
        optimizer = torch.optim.Adam([z], lr=1e-3)
        mse = torch.nn.MSELoss()
        for each_epoch in range(0, epoch):
            predict_image = decoder(z)
            loss = mse(predict_image[0], each_real_image[0].cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        z_list.append(z.cpu().detach().numpy())
        count += 1
        print(f"{100*(count/len(dataset))}% done")
    tac(rounding=True)
    return z_list


def calculate_likelihood_of_z_from_distribution(z_list, distribution):
    likelihoods = np.array([prior_factory.get_pdf(distribution, each_z) for each_z in np.array(z_list)])
    return np.average(likelihoods, axis=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        choices=['ffhq32', 'ffhq64', 'cifar', 'mnist', 'mnist_fashion', 'emnist'])
    parser.add_argument('--distribution', type=str,
                        choices=['standard_normal', 'uniform', 'gamma', 'beta', 'chi', 'dirichlet', 'laplace'])
    args = parser.parse_args()
    latent_dim = 100
    image_size = 32
    image_shape = [3, image_size, image_size]
    decoder = aae_for_various_prior.Decoder(latent_dim, image_shape).cuda()
    decoder.load_state_dict(torch.load(f'model/decoder_{args.dataset}_{args.distribution}'))
    z_list = train_z_with_decoder(decoder, latent_dim, args.dataset, image_size)
    likelihoods = calculate_likelihood_of_z_from_distribution(z_list, args.distribution)
    wandb.login()
    wandb.init(project="AAE_likelihoods",
               config={
                   "distribution": args.distribution,
                   "dataset": args.dataset
               })
    wandb.log({f"Likelihoods {args}": wandb.Table(dataframe=pd.DataFrame(likelihoods, columns=["likelihood"]))})


if __name__ == "__main__":
    main()
