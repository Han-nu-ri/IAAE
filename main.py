import argparse
import numpy as np
import pandas as pd
import torch
import generative_model_score
inception_model_score = generative_model_score.GenerativeModelScore()
import matplotlib
import matplotlib.pyplot as plt
import wandb
from torch.autograd import Variable
import tqdm
import os
import hashlib
from PIL import Image
import data_helper
import prior_factory
import model
from torchvision.utils import save_image
matplotlib.use('Agg')


def load_inception_model(train_loader, dataset, image_size):
    # TODO: 경로 args로 받도록 개선
    #icm_path = '../../../inception_model_info/'
    icm_path = './inception_model_info/'
    inception_model_score.lazy_mode(True)
    real_images_info_file_name = hashlib.md5(str(train_loader.dataset).encode()).hexdigest() + '.pickle'
    os.makedirs('inception_model_info', exist_ok=True)
    if os.path.exists(icm_path + real_images_info_file_name):
        print("Using generated real image info.")
        inception_model_score.load_real_images_info(icm_path + real_images_info_file_name)
    else:
        inception_model_score.model_to('cuda')
        inception_model_score.lazy_forward(dataset, image_size, real_forward=True, device='cuda')
        inception_model_score.calculate_real_image_statistics()
        inception_model_score.save_real_images_info(icm_path + real_images_info_file_name)
        inception_model_score.model_to('cpu')
    inception_model_score.freeze_layers()
    return inception_model_score


def inference_image(model_name, mapper, decoder, batch_size, latent_dim, distribution):
    with torch.no_grad():
        if model_name == 'non-prior' :
            z = torch.randn(batch_size, latent_dim, device='cuda')
            result = decoder(mapper(z)).cpu()
        if model_name == 'mimic':
            z = torch.rand(batch_size, latent_dim).cuda() * 2 - 1
            result = decoder(mapper(z)).cpu()
        else:
            z = Variable(torch.FloatTensor(prior_factory.get_sample(distribution, batch_size, latent_dim))).cuda()
            if decoder.has_mask_layer:
                z = torch.mul(z, decoder.mask_vector)
            result = decoder(z).cpu()
    return result





def write_pca_data(model_name, dataset, image_size, distribution, epoch, real_pca, fake_pca):
    folder_name = 'pca_data/'
    os.makedirs(folder_name, exist_ok=True)
    pca_df = pd.DataFrame({"real_pca_x": real_pca[:, 0], "real_pca_y": real_pca[:, 1], "real_pca_z": real_pca[:, 2],
                           "fake_pca_x": fake_pca[:, 0], "fake_pca_y": fake_pca[:, 1], "fake_pca_z": fake_pca[:, 2]})
    pca_df.to_csv(f"{folder_name}{model_name}_{dataset}_{image_size}_{distribution}_{epoch}_pca_data.csv")


def log_index_with_inception_model(args, d_loss, decoder, discriminator, encoder, g_loss, i, inception_model_score, r_loss,
                                   dataset, distribution, latent_dim, model_name, mapper=None):
    decoder, discriminator, encoder, mapper, inception_model_score = \
        swap_gpu_between_generative_model_and_inception_model(decoder, discriminator, encoder, inception_model_score,
                                                              mapper, generative_model_gpu=False)
    inception_model_score.lazy_forward(dataset, decoder=decoder, distribution=distribution, latent_dim=latent_dim,\
                                       real_forward=False, device='cuda', model_name=model_name, mapper=mapper, \
                                       gen_image_in_gpu=args.gen_image_in_gpu, batch_size=args.isnet_batch_size)
    inception_model_score.calculate_fake_image_statistics()
    metrics, feature_pca_plot, real_pca, fake_pca = \
        inception_model_score.calculate_generative_score(feature_pca_plot=True)
    decoder, discriminator, encoder, mapper, inception_model_score = \
        swap_gpu_between_generative_model_and_inception_model(decoder, discriminator, encoder, inception_model_score,
                                                              mapper, generative_model_gpu=True)

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
    if decoder.has_mask_layer:
        mask_vector_2d_list = [[each_mask_element] for each_mask_element in decoder.mask_vector.cpu().detach().numpy()]
        mask_table = wandb.Table(data=mask_vector_2d_list, columns=["mask_value"])
        wandb.log({"mask_vector_hist": wandb.plot.histogram(mask_table, "mask_value", title="mask_vector_hist")}, step=i)
    inception_model_score.clear_fake()
    return encoder, decoder, discriminator, mapper, real_pca, fake_pca


def swap_gpu_between_generative_model_and_inception_model(decoder, discriminator, encoder, inception_model_score,
                                                          mapper, generative_model_gpu):
    if not generative_model_gpu:
        # offload all GAN model to cpu and onload inception model to gpu
        encoder = encoder.to('cpu')
        decoder = decoder.to('cpu')
        discriminator = discriminator.to('cpu')
        if mapper is not None:
            mapper = mapper.to('cpu')
        inception_model_score.model_to('cuda')
    else:
        inception_model_score.model_to('cpu')
        encoder = encoder.to('cuda')
        decoder = decoder.to('cuda')
        if mapper is not None:
            mapper = mapper.to('cuda')
        discriminator = discriminator.to('cuda')
    return decoder, discriminator, encoder, mapper, inception_model_score


def generate_image_and_save_in_wandb(mapper, decoder, each_epoch, model_name, latent_dim, distribution, dataset):
    number_of_image = 100
    sampled_images = inference_image(model_name, mapper, decoder, number_of_image, latent_dim, distribution)
    folder_name = '%s_%s' % (dataset, model_name)
    os.makedirs('images/%s' % folder_name, exist_ok=True)
    image_name = "images/%s/%d_epoch.png" % (folder_name, each_epoch)
    save_image(sampled_images.data, image_name, nrow=10, normalize=True)
    generated_image_file = Image.open(image_name)
    wandb.log({'image': wandb.Image(generated_image_file, caption='%s_epochs' % each_epoch)}, step=each_epoch)


def log_z_posterior(encoder, i):
    wandb.log({"z_posterior": wandb.Histogram(np.average(encoder.z_posterior.weight.data.cpu().numpy(), axis=1))},
              step=i)
    wandb.log({"z_posterior_avg": np.average(encoder.z_posterior.weight.data.cpu().numpy())}, step=i)


def write_feature(model_name, dataset, image_size, distribution, epoch, real_feature_np, fake_feature_np):
    folder_name = 'feature_data/'
    os.makedirs(folder_name, exist_ok=True)
    np.savetxt(f"{folder_name}{model_name}_{dataset}_{image_size}_{distribution}_{epoch}_real_feature_data.csv",
               real_feature_np,
               delimiter=",")
    np.savetxt(f"{folder_name}{model_name}_{dataset}_{image_size}_{distribution}_{epoch}_fake_feature_data.csv",
               fake_feature_np,
               delimiter=",")


def save_models(args, decoder, encoder, mapper):
    os.makedirs('model', exist_ok=True)
    torch.save(encoder.state_dict(),
               f'model/encoder_{args.model_name}_{args.dataset}_{args.image_size}_{args.distribution}')
    torch.save(decoder.state_dict(),
               f'model/decoder_{args.model_name}_{args.dataset}_{args.image_size}_{args.distribution}')
    if args.model_name == 'mimic':
        torch.save(mapper.state_dict(),
                   f'model/mapper_{args.model_name}_{args.dataset}_{args.image_size}_{args.distribution}')


def load_pretrain_autoencoder_model(args, encoder, decoder, pretrain_epoch):
    is_pretrained = False
    if os.path.exists(f'./model/pretrained_encoder_{args.dataset}_{args.image_size}_{pretrain_epoch}') and \
            os.path.exists(f'./model/pretrained_decoder_{args.dataset}_{args.image_size}_{pretrain_epoch}'):
        encoder.load_state_dict(torch.load(f'./model/pretrained_encoder_{args.dataset}_{args.image_size}_{pretrain_epoch}'))
        decoder.load_state_dict(torch.load(f'./model/pretrained_decoder_{args.dataset}_{args.image_size}_{pretrain_epoch}'))
        is_pretrained = True
    return encoder, decoder, is_pretrained


def save_pretrain_autoencoder_model(args, decoder, encoder, pretrain_epoch):
    os.makedirs('model', exist_ok=True)
    torch.save(encoder.state_dict(), f'./model/pretrained_encoder_{args.dataset}_{args.image_size}_{pretrain_epoch}')
    torch.save(decoder.state_dict(), f'./model/pretrained_decoder_{args.dataset}_{args.image_size}_{pretrain_epoch}')


def pretrain_autoencoder(ae_optimizer, args, decoder, encoder, train_loader):
    encoder, decoder, is_pretrained = load_pretrain_autoencoder_model(args, encoder, decoder, args.pretrain_epoch)
    if is_pretrained:
        print("Using pretrained autoencoder.")
    else:
        print("There are no pretrained autoencoder, start pretraining.")
        for i in range(0, args.pretrain_epoch):
            loss_r = 0
            for each_batch in tqdm.tqdm(train_loader):
                each_batch = each_batch[0].to('cuda')
                loss_r = model.update_autoencoder(ae_optimizer, each_batch, encoder, decoder)
            print(f"epoch: {i}, loss: {loss_r}")
        save_pretrain_autoencoder_model(args, decoder, encoder, args.pretrain_epoch)
    return decoder, encoder


def log_and_write_pca(args, d_loss, decoder, discriminator, encoder, g_loss, i, inception_model_score, mapper, r_loss):
    if i % args.log_interval == 0 or i == (args.epochs - 1):
        generate_image_and_save_in_wandb(mapper, decoder, i, args.model_name, args.latent_dim, args.distribution,
                                         args.dataset)
        encoder, decoder, discriminator, mapper, real_pca, fake_pca = \
            log_index_with_inception_model(args, d_loss, decoder, discriminator, encoder, g_loss, i,
                                           inception_model_score, r_loss, args.dataset, args.distribution,
                                           args.latent_dim, args.model_name, mapper)
        if i == (args.epochs - 1):
            write_pca_data(args.model_name, args.dataset, args.image_size, args.distribution, i, real_pca, fake_pca)
            write_feature(args.model_name, args.dataset, args.image_size, args.distribution, i,
                          inception_model_score.real_feature_np,
                          inception_model_score.fake_feature_np)
    return decoder, discriminator, encoder, mapper


def main(args):
    train_loader, _ = data_helper.get_data(args.dataset, args.batch_size, args.image_size)
    if args.wandb:
        wandb_name = "%s[%d]_%s" % (args.dataset, args.image_size, args.model_name)
        wandb.login()
        wandb.init(project="AAE", config=args, name=wandb_name)
    inception_model_score = load_inception_model(train_loader, args.dataset, args.image_size)
    ae_optimizer, d_optimizer, decoder, discriminator, encoder, g_optimizer, mapper = \
        model.get_aae_model_and_optimizer(args.latent_dim, args.image_size, args.has_mask_layer)
    if args.model_name == 'mimic':
        mapper = model.Mimic(args.latent_dim, args.latent_dim, args.mapper_inter_nz, args.mapper_inter_layer).to(
            args.device)
        decoder, encoder = pretrain_autoencoder(ae_optimizer, args, decoder, encoder, train_loader)
    if args.model_name == 'non-prior':
        mapper, m_optimizer = model.get_nonprior_model_and_optimizer(args.latent_dim, args.mapper_inter_nz, args.mapper_inter_layer)

    for i in range(0, args.epochs):
        d_loss, g_loss, r_loss = 0, 0, 0
        encoded_feature_list = []
        for each_batch in tqdm.tqdm(train_loader, desc="train[%d/%d]" % (i, args.epochs)):
            each_batch = each_batch[0].cuda()
            if args.model_name in ['aae', 'mask_aae']:
                d_loss, g_loss, r_loss = model.update_aae(ae_optimizer, args, d_optimizer, decoder, discriminator,
                                                          each_batch, encoder, g_optimizer, args.latent_dim)
            elif args.model_name == 'mimic':
                r_loss, encoded_feature = \
                    model.update_autoencoder(ae_optimizer, each_batch, encoder, decoder, return_encoded_feature=True)
                encoded_feature_list.append(encoded_feature)                
            elif args.model_name == 'non-prior':
                r_loss, encoded_feature = model.update_autoencoder(ae_optimizer, each_batch, encoder, decoder, return_encoded_feature_gpu=True)
                d_loss, m_loss = model.update_posterior_part(mapper, discriminator, m_optimizer, d_optimizer, encoded_feature, args.latent_dim)

        if args.model_name == 'mimic':
            g_loss = model.train_mapper(encoder, mapper, args.device, args.lr, args.batch_size, encoded_feature_list)

        if args.wandb:
            decoder, discriminator, encoder, mapper = log_and_write_pca(args, d_loss, decoder, discriminator, encoder,
                                                                    g_loss, i, inception_model_score, mapper, r_loss)
    save_models(args, decoder, encoder, mapper)
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    
    '''
    vanilla command : 
    python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae --batch_size=128 --epochs=100 --latent_dim=32 --log_interval=10 --mapper_inter_nz=32 --mapper_inter_layer=1 --wandb=True --gen_image_in_gpu=True
    
    '''
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dataset', type=str, choices=['ffhq', 'cifar', 'mnist', 'mnist_fashion', 'emnist'])
    parser.add_argument('--distribution', type=str,
                        choices=['standard_normal', 'uniform', 'gamma', 'beta', 'chi', 'dirichlet', 'laplace'],
                        default='standard_normal')
    parser.add_argument('--image_size', type=int, choices=[32, 64, 128], default=128)
    parser.add_argument('--model_name', type=str, choices=['aae', 'mimic', 'mask_aae', 'non-prior'])
    parser.add_argument('--has_mask_layer', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--isnet_batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pretrain_epoch', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--mapper_inter_nz', type=int, default=32)
    parser.add_argument('--mapper_inter_layer', type=int, default=1)
    parser.add_argument('--wandb', type=bool, default=False)
    parser.add_argument('--gen_image_in_gpu', type=bool, default=False)
    args = parser.parse_args()

    if args.model_name == 'mask_aae':
        args.has_mask_layer = True
    main(args)
