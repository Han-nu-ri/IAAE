import argparse
import distutils
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
from utils import *
import seaborn as sns
import time
matplotlib.use('Agg')


def main(args):
    global inception_model_score
    
    train_loader, _ = data_helper.get_data(args.dataset, args.batch_size, args.image_size, args.environment)
    if args.wandb:
        wandb_name = "%s[%d]_%s" % (args.dataset, args.image_size, args.model_name)
        wandb.login()
        wandb.init(project="AAE", config=args, name=wandb_name)
    inception_model_score = load_inception_model(inception_model_score, train_loader, args.dataset, args.image_size, args.environment)
    ae_optimizer, d_optimizer, decoder, discriminator, encoder, g_optimizer, mapper = \
        model.get_aae_model_and_optimizer(args)
    if args.model_name == 'mimic':
        mapper = model.Mimic(args.latent_dim, args.latent_dim, args.mapper_inter_nz, args.mapper_inter_layer).to(
            args.device)
        decoder, encoder = pretrain_autoencoder(ae_optimizer, args, decoder, encoder, train_loader)
    if args.model_name == 'non-prior':
        mapper, m_optimizer = model.get_nonprior_model_and_optimizer(args)
    if args.model_name == 'learning-prior':
        mapper, m_optimizer, discriminator_forpl, dpl_optimizer = \
            model.get_learning_prior_model_and_optimizer(args)
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
        
        
    
    args.start_time = time.time()
    
    if args.pretrain_epoch > 0:
        pretrain_autoencoder(ae_optimizer, args, decoder, encoder, train_loader)
    
    log_dict, log, log2 = {}, {}, {}
    for i in range(0, args.epochs):
        if args.time_limit and timeout(args.time_limit, args.start_time) : break
        d_loss, g_loss, r_loss = 0, 0, 0
        encoded_feature_list = []
        for each_batch in tqdm.tqdm(train_loader, desc="train[%d/%d]" % (i, args.epochs)):
            each_batch = each_batch[0].to(args.device)
            if args.model_name in ['aae', 'mask_aae']:
                log = model.update_aae(ae_optimizer, args, d_optimizer, decoder, discriminator,
                                                          each_batch, encoder, g_optimizer, args.latent_dim)

            elif args.model_name == 'mimic':
                log, encoded_feature = \
                    model.update_autoencoder(ae_optimizer, each_batch, encoder, decoder, return_encoded_feature=True)
                encoded_feature_list.append(encoded_feature)         

            elif args.model_name == 'non-prior':
                log, encoded_feature = model.update_autoencoder(ae_optimizer, each_batch, encoder, decoder, return_encoded_feature_gpu=True, flag_retain_graph=False)
                log2 = model.update_posterior_part(args, mapper, discriminator, m_optimizer, d_optimizer, encoded_feature) 

            elif args.model_name == 'learning-prior':
                log = model.update_aae_with_mappedz(args, ae_optimizer, d_optimizer, decoder, discriminator, mapper, each_batch, encoder, g_optimizer)
                log2= model.update_mapper_with_discriminator_forpl(args, dpl_optimizer, decoder_optimizer, m_optimizer, discriminator_forpl, decoder, mapper, each_batch)

            if args.model_name == 'mimic':
                log = model.train_mapper(args, encoder, mapper, args.device, args.lr, args.batch_size, encoded_feature_list)
        
        log_dict.update(log)
        log_dict.update(log2)
        
        # wandb log를 남기고, time_check와 time_limit 옵션이 둘다 없을때만, log interval마다 기록을 남김
        if args.wandb and not args.time_check and not args.time_limit:
            decoder, discriminator, encoder, mapper = log_and_write_pca(args, wandb, decoder, discriminator, encoder,
                                                                     i, inception_model_score, mapper, log_dict)
            
    # wandb log를 남기고, time_check 또는 time_limit 옵션 둘 중 하나라도 있으면, 최후에 기록을 남김
    if args.wandb and (args.time_check or args.time_limit):
            decoder, discriminator, encoder, mapper = force_log_and_write_pca(args, wandb, decoder, discriminator, encoder,
                                                                    i, inception_model_score, mapper, log_dict)
        
    save_models(args, decoder, encoder, mapper)
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    
    '''
    vanilla command : 
    python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=aae --batch_size=128 --epochs=100 --pretrain_epoch=0 --latent_dim=32 --log_interval=100 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_check=True
    
    
    non-prior command : 
    python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=non-prior --batch_size=128 --epochs=300 --pretrain_epoch=10 --latent_dim=32 --log_interval=300 --mapper_inter_nz=32 --mapper_inter_layer=1 --wandb=True --gen_image_in_gpu=True --isnet_batch_size=128 --time_limit=9999
    
    learning-prior command : 
    python3 main.py --device=cuda:0 --dataset=ffhq --image_size=32 --model_name=learning-prior --batch_size=128 --epochs=300 --pretrain_epoch=0 --latent_dim=32 --log_interval=300 --mapper_inter_nz=32 --mapper_inter_layer=1 --gen_image_in_gpu=True --isnet_batch_size=128 --time_limit=3248 --wandb=True
    
    '''
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset', type=str, choices=['ffhq', 'cifar', 'mnist', 'mnist_fashion', 'emnist'])
    parser.add_argument('--distribution', type=str,
                        choices=['standard_normal', 'uniform', 'gamma', 'beta', 'chi', 'dirichlet', 'laplace'],
                        default='standard_normal')
    parser.add_argument('--image_size', type=int, choices=[32, 64, 128], default=128)
    parser.add_argument('--model_name', type=str, choices=['aae', 'mimic', 'mask_aae', 'non-prior', 'learning-prior'])
    parser.add_argument('--has_mask_layer', type=str2bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--isnet_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--pretrain_epoch', type=int, default=10)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--mapper_inter_nz', type=int, default=32)
    parser.add_argument('--mapper_inter_layer', type=int, default=1)
    parser.add_argument('--load_pretrain_model', type=str2bool, default=False)
    parser.add_argument('--wandb', type=str2bool, default=False)
    parser.add_argument('--gen_image_in_gpu', type=str2bool, default=False)
    parser.add_argument('--time_check', type=str2bool, default=False)
    parser.add_argument('--time_limit', type=int, default=0)
    parser.add_argument('--environment', type=str, default='yhs')
    
    
    args = parser.parse_args()

    if args.model_name == 'mask_aae':
        args.has_mask_layer = True
        
    if args.time_check or args.time_limit : 
        assert args.log_interval == args.epochs, \
            "if you use time_check or time_limit option, metric cannot be calculated in the middle and can only be calculated at the end[Recomendation : set log_interval=epochs]"
    
    global device
    device = args.device
    
    main(args)
