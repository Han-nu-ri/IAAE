import argparse
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import generative_model_score
import matplotlib
import wandb
from torch.autograd import Variable
import tqdm
import os
import hashlib
from PIL import Image
import matplotlib.pyplot as plt
import data_helper
import prior_factory
matplotlib.use('Agg')


class Encoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(image_shape)), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.z_posterior = nn.Linear(64, latent_dim)
        self.latent_dim = latent_dim

    def forward(self, x):
        x_flat = x.view(x.shape[0], -1)
        z_posterior = self.encode(x_flat)
        return z_posterior

    def encode(self, x):
        x = self.model(x)
        z_posterior = self.z_posterior(x)
        return z_posterior


class Decoder(nn.Module):
    def __init__(self, latent_dim, image_shape):
        super(Decoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, int(np.prod(image_shape))),
            nn.Tanh(),
        )
        self.image_shape = image_shape

    def forward(self, z_posterior):
        decoded_flat = self.model(z_posterior)
        decoded = decoded_flat.view(decoded_flat.shape[0], *self.image_shape)
        return decoded


class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        validity = self.model(z)
        return validity


def sample_image(encoder, decoder, x):
    z = encoder(x)
    return decoder(z)


def inference_image(decoder, batch_size, latent_dim, distribution):
    z = torch.FloatTensor(prior_factory.get_sample(distribution, batch_size, latent_dim)).cuda()
    return decoder(z)


def update_autoencoder(ae_optimizer, X_train_batch, encoder, decoder):
    ae_optimizer.zero_grad()
    z_posterior = encoder(X_train_batch)
    X_decoded = decoder(z_posterior)
    pixelwise_loss = torch.nn.L1Loss()
    r_loss = pixelwise_loss(X_decoded, X_train_batch)
    r_loss.backward()
    ae_optimizer.step()
    return r_loss


def update_discriminator(d_optimizer, X_train_batch, encoder, discriminator, latent_dim, distribution):
    d_optimizer.zero_grad()
    batch_size = X_train_batch.size(0)
    z_prior = Variable(torch.FloatTensor(prior_factory.get_sample(distribution, batch_size, latent_dim))).cuda()
    z_posterior = encoder(X_train_batch)
    d_loss = -torch.mean(torch.log(discriminator(z_prior)) + torch.log(1 - discriminator(z_posterior)))
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data


def update_generator(g_optimizer, X_train_batch, encoder, discriminator):
    g_optimizer.zero_grad()
    z_posterior = encoder(X_train_batch)
    g_loss = -torch.mean(torch.log(discriminator(z_posterior)))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data


def load_inception_model(train_loader):
    # load real images info or generate real images info
    inception_model_score = generative_model_score.GenerativeModelScore()
    inception_model_score.lazy_mode(True)
    real_images_info_file_name = hashlib.md5(str(train_loader.dataset).encode()).hexdigest() + '.pickle'
    os.makedirs('inception_model_info', exist_ok=True)
    if os.path.exists('./inception_model_info/' + real_images_info_file_name):
        print("Using generated real image info.")
        print(train_loader.dataset)
        inception_model_score.load_real_images_info('./inception_model_info/' + real_images_info_file_name)
    else:
        inception_model_score.model_to('cuda')
        # put real image
        for each_batch in train_loader:
            X_train_batch = each_batch[0]
            inception_model_score.put_real(X_train_batch)

        # generate real images info
        inception_model_score.lazy_forward(batch_size=32, device='cuda', real_forward=True)
        inception_model_score.calculate_real_image_statistics()
        # save real images info for next experiments
        inception_model_score.save_real_images_info('./inception_model_info/' + real_images_info_file_name)
        # offload inception_model
        inception_model_score.model_to('cpu')
    inception_model_score.freeze_layers()
    return inception_model_score


def write_pca_data(dataset, distribution, epoch, real_pca, fake_pca):
    folder_name = 'pca_data/'
    os.makedirs(folder_name, exist_ok=True)
    pca_df = pd.DataFrame({"real_pca_x": real_pca[:, 0], "real_pca_y": real_pca[:, 1],
                           "fake_pca_x": fake_pca[:, 0], "fake_pca_y": fake_pca[:, 1]})
    pca_df.to_csv(f"{folder_name}{dataset}_{distribution}_{epoch}_pca_data.csv")


def log_index_with_inception_model(d_loss, decoder, discriminator, encoder, g_loss, i, inception_model_score, r_loss):
    # offload all GAN model to cpu and onload inception model to gpu
    encoder = encoder.to('cpu')
    decoder = decoder.to('cpu')
    discriminator = discriminator.to('cpu')
    inception_model_score.model_to('cuda')
    # generate fake images info
    inception_model_score.lazy_forward(batch_size=32, device='cuda', fake_forward=True)
    inception_model_score.calculate_fake_image_statistics()
    metrics, feature_pca_plot, real_pca, fake_pca = \
        inception_model_score.calculate_generative_score(feature_pca_plot=True)
    # onload all GAN model to gpu and offload inception model to cpu
    inception_model_score.model_to('cpu')
    encoder = encoder.to('cuda')
    decoder = decoder.to('cuda')
    discriminator = discriminator.to('cuda')
    precision, recall, fid, inception_score_real, inception_score_fake, density, coverage = \
        metrics['precision'], metrics['recall'], metrics['fid'], metrics['real_is'], metrics['fake_is'], \
        metrics['density'], metrics['coverage']
    wandb.log({'IsNet feature': wandb.Image(feature_pca_plot),
               "r_loss": r_loss,
               "d_loss": d_loss,
               "g_loss": g_loss,
               "precision": precision,
               "recall": recall,
               "fid": fid,
               "inception_score_real": inception_score_real,
               "inception_score_fake": inception_score_fake,
               "density": density,
               "coverage": coverage},
              step=i)
    inception_model_score.clear_fake()
    return decoder, discriminator, encoder, real_pca, fake_pca


def get_model_and_optimizer(image_shape, latent_dim):
    encoder = Encoder(latent_dim, image_shape).cuda()
    decoder = Decoder(latent_dim, image_shape).cuda()
    discriminator = Discriminator(latent_dim).cuda()
    ae_optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    return ae_optimizer, d_optimizer, decoder, discriminator, encoder, g_optimizer


def save_images(each_epoch, images):
    images = images.numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    folder_name = 'generated_images'
    os.makedirs(folder_name, exist_ok=True)
    plt.figure(figsize=(5, 5))
    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow((images[i, :, :, :] * 255).astype('uint8'))
        plt.axis('off')
    generated_image_file = folder_name + '/image_at_epoch_' + str(each_epoch + 1) + '.png'
    plt.savefig(generated_image_file)
    plt.close()
    return Image.open(generated_image_file)


def generate_image(decoder, encoder, i, train_loader):
    for each_batch in tqdm.tqdm(train_loader):
        X_train_batch = Variable(each_batch[0]).cuda()
        sampled_images = sample_image(encoder, decoder, X_train_batch).detach().cpu()
        break
    generated_image_file = save_images(i, sampled_images.data[0:25].cpu())
    wandb.log({'image': wandb.Image(generated_image_file, caption='%s_epochs' % i)}, step=i)


def log_z_posterior(encoder, i):
    wandb.log({"z_posterior": wandb.Histogram(np.average(encoder.z_posterior.weight.data.cpu().numpy(), axis=1))},
              step=i)
    wandb.log({"z_posterior_avg": np.average(encoder.z_posterior.weight.data.cpu().numpy())}, step=i)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        choices=['ffhq32', 'ffhq64', 'cifar', 'mnist', 'mnist_fashion', 'emnist'])
    parser.add_argument('--distribution', type=str,
                        choices=['standard_normal', 'uniform', 'gamma', 'beta', 'chi', 'dirichlet', 'laplace'])
    args = parser.parse_args()
    batch_size = 1024
    epochs = 200
    image_size = 32
    flag_log_index_with_inceptiom_module = True
    flag_use_mu_and_sigma_of_train_data = False
    if args.dataset == 'ffhq32':
        train_loader, test_loader = data_helper.get_ffhq_thumbnails(batch_size, image_size)
    elif args.dataset == 'ffhq64':
        batch_size = 256
        image_size = 64
        train_loader, test_loader = data_helper.get_ffhq_thumbnails(batch_size, image_size)
    elif args.dataset == 'cifar':
        train_loader, test_loader = data_helper.get_cifar_dataset(batch_size, image_size)
    elif args.dataset == 'emnist':
        train_loader, test_loader = data_helper.get_e_mnist(batch_size, image_size)
    elif args.dataset == 'mnist':
        train_loader, test_loader = data_helper.get_mnist(batch_size, image_size)
    elif args.dataset == 'mnist_fashion':
        train_loader, test_loader = data_helper.get_fashion_mnist(batch_size, image_size)
    else:
        train_loader, test_loader = data_helper.get_celebA_dataset(batch_size, image_size)
    latent_dim = 100
    save_image_interval = loss_calculation_interval = 5
    image_shape = [3, image_size, image_size]
    wandb.login()
    wandb.init(project="AAE",
               config={
                   "batch_size": batch_size,
                   "epochs": epochs,
                   "img_size": image_size,
                   "save_image_interval": save_image_interval,
                   "loss_calculation_interval": loss_calculation_interval,
                   "latent_dim": latent_dim,
               })
    if flag_log_index_with_inceptiom_module:
        inception_model_score = load_inception_model(train_loader)

    ae_optimizer, d_optimizer, decoder, discriminator, encoder, g_optimizer = get_model_and_optimizer(image_shape,
                                                                                                      latent_dim)
    for i in range(0, epochs):
        for each_batch in tqdm.tqdm(train_loader):
            X_train_batch = Variable(each_batch[0]).cuda()
            r_loss = update_autoencoder(ae_optimizer, X_train_batch, encoder, decoder)
            d_loss = update_discriminator(d_optimizer, X_train_batch, encoder, discriminator, latent_dim,
                                          args.distribution)
            g_loss = update_generator(g_optimizer, X_train_batch, encoder, discriminator)

            if i % loss_calculation_interval == 0:
                if flag_use_mu_and_sigma_of_train_data:
                    sampled_images = sample_image(encoder, decoder, X_train_batch).detach().cpu()
                else:
                    sampled_images = inference_image(decoder, batch_size, latent_dim, args.distribution).detach().cpu()
                if flag_log_index_with_inceptiom_module:
                    inception_model_score.put_fake(sampled_images)

        if i % save_image_interval == 0:
            generate_image(decoder, encoder, i, train_loader)

        if i % loss_calculation_interval == 0:
            if flag_log_index_with_inceptiom_module:
                decoder, discriminator, encoder, real_pca, fake_pca = \
                    log_index_with_inception_model(d_loss, decoder, discriminator, encoder, g_loss, i,
                                                   inception_model_score, r_loss)
                write_pca_data(args.dataset, args.distribution, i, real_pca, fake_pca)
            log_z_posterior(encoder, i)

    os.makedirs('model', exist_ok=True)
    torch.save(decoder.state_dict(), f'model/decoder_{args.dataset}_{args.distribution}')
    wandb.finish()


if __name__ == "__main__":
    main()
