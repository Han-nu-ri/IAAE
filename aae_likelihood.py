import argparse
import numpy as np
import pandas as pd
import data_helper
import model
import prior_factory
import torch
import time
import wandb
import matplotlib
import matplotlib.pyplot as plt
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
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour, t_min) = divmod(t_min,60)
    print('Time passed: {}hour:{}min:{}sec'.format(t_hour, t_min, t_sec))


def image_show(each_epoch, predict_image, real_image):
    predict_image_reshape = np.array([])
    predict_image_reshape = predict_image_reshape.reshape(32, 32, 0)
    real_image_reshape = np.array([])
    real_image_reshape = real_image_reshape.reshape(32, 32, 0)
    for i in range(0, 3):
        predict_image_reshape = np.append(predict_image_reshape, predict_image[i, :, :].reshape(32, 32, 1), axis=2)
        real_image_reshape = np.append(real_image_reshape, real_image[i, :, :].numpy().reshape(32, 32, 1), axis=2)
    fig = plt.figure(figsize=(3, 1))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(predict_image_reshape, cmap='gray')
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(real_image_reshape, cmap='gray')
    fig.show()


def train_z_with_decoder(decoder, latent_dim, dataset, image_size, distribution):
    z_list = []
    decoder.eval()
    if dataset == 'ffhq32':
        dataset = data_helper.get_ffhq_thumbnails_raw_images(image_size)
    else:
        dataset = data_helper.get_e_mnist_raw_images(image_size)

    epoch = 1000
    tic()
    mse = torch.nn.MSELoss()
    # TODO: 아래는 batch로 z를 업데이트하는 코드로, 불필요하다고 의사결정이 나면 삭제해야 함.
    # batch_size = 1024
    # z = torch.tensor(prior_factory.get_sample(distribution, batch_size, latent_dim), requires_grad=True,
    #                  dtype=torch.float).cuda()
    # z = torch.nn.Parameter(z)
    # optimizer = torch.optim.Adam([z], lr=1e-3)
    # for i in range(0, epoch):
    #     iteration_per_epoch = math.ceil(len(dataset) / batch_size)
    #     for j in range(0, iteration_per_epoch):
    #         batch_end_index = (j + 1) * batch_size
    #         if batch_end_index >= len(dataset):
    #             batch_end_index = len(dataset)
    #
    #         z_batch = z[j*batch_size:batch_end_index]
    #         real_image_batch = dataset[j*batch_size:batch_end_index]
    #         predict_image_batch = decoder(z_batch)
    #         loss = mse(predict_image_batch, real_image_batch)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print(f"{100 * (i / epoch)}% done")

    count = 0
    batch_size = 1
    for each_real_image in dataset:
        z = torch.tensor(prior_factory.get_sample(distribution, batch_size, latent_dim), requires_grad=True,
                         dtype=torch.float).cuda()
        z = torch.nn.Parameter(z)
        optimizer = torch.optim.Adam([z], lr=1e-2)
        mse = torch.nn.MSELoss()
        for each_epoch in range(0, epoch):
            predict_image = decoder(z)
            loss = mse(predict_image[0], each_real_image[0].cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # TODO: epcoh 테스트 용, 불필요해지면 주석하거나 삭제하기.
            if each_epoch % 100 == 0:
                image_show(each_epoch, predict_image[0].cpu().detach().numpy(), each_real_image[0])
        z_list.append(z.cpu().detach().numpy())
        count += 1
        print(f"{100*(count/len(dataset))}% done")
        # TODO: 작업 실행 전 아래 코드 삭제
        if count == 2:
            break
    tac(rounding=True)
    return z_list


def calculate_likelihood_of_z_from_distribution(z_list, distribution):
    likelihoods = np.array([prior_factory.get_pdf(distribution, each_z) for each_z in np.array(z_list)])
    return np.sum(-np.log(likelihoods + 0.001), axis=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        choices=['ffhq32', 'ffhq64', 'cifar', 'mnist', 'mnist_fashion', 'emnist'])
    parser.add_argument('--distribution', type=str,
                        choices=['standard_normal', 'uniform', 'gamma', 'beta', 'chi', 'dirichlet', 'laplace'])
    args = parser.parse_args()
    latent_dim = 32
    image_size = 32
    image_shape = [3, image_size, image_size]
    decoder = model.Decoder(latent_dim, image_shape).cuda()
    decoder.load_state_dict(torch.load(f'model/decoder_{args.dataset}_{args.distribution}'))
    z_list = train_z_with_decoder(decoder, latent_dim, args.dataset, image_size, args.distribution)
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
