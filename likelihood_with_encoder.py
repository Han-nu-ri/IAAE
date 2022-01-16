import argparse
import time
import matplotlib
import numpy as np
import torch
import tqdm
from torch.autograd import Variable
import data_helper
import model
import prior_factory
matplotlib.use('TkAgg')


_start_time = time.time()
def tic():
    global _start_time
    _start_time = time.time()


def tac(rounding=True):
    if rounding:
        t_sec = round(time.time() - _start_time)
    else:
        t_sec = time.time() - _start_time
    (t_min, t_sec) = divmod(t_sec, 60)
    (t_hour, t_min) = divmod(t_min, 60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


def print_percentage_of_data_that_is_difficult_to_generate(z_array, distribution):
    likelihoods = np.array([prior_factory.get_pdf(distribution, each_z) for each_z in z_array])
    percentage_of_data_that_is_difficult_to_generate = (np.any(likelihoods == 0, axis=1).sum() / likelihoods.shape[0])*100
    print(f"percentage_of_data_that_is_difficult_to_generate in {distribution}: "
          f"{percentage_of_data_that_is_difficult_to_generate}%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        choices=['ffhq', 'cifar', 'mnist', 'mnist_fashion', 'emnist'])
    parser.add_argument('--image_size', type=int, choices=[32, 64, 128])
    args = parser.parse_args()
    latent_dim = 32
    image_size = args.image_size
    image_shape = [3, image_size, image_size]
    batch_size = 512
    train_loader, _ = data_helper.get_data(args.dataset, batch_size, image_size)

    for each_distribution in ['standard_normal', 'uniform', 'gamma', 'beta', 'chi', 'laplace']:
        encoder = model.Encoder(latent_dim, image_shape).cuda()
        encoder.load_state_dict(torch.load(f'model/image_size_128_epoch_500_test_1/encoder_{args.dataset}_{args.image_size}_{each_distribution}'))

        z_array = None
        for each_batch in tqdm.tqdm(train_loader):
            each_batch = Variable(each_batch[0]).cuda()
            each_z_batch = encoder(each_batch)
            if z_array is None:
                z_array = each_z_batch.cpu().detach().numpy()
            else:
                z_array = np.concatenate((z_array, (each_z_batch.cpu().detach().numpy())))
        print_percentage_of_data_that_is_difficult_to_generate(z_array, each_distribution)


if __name__ == "__main__":
    main()
