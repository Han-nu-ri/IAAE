import distutils
import os
import hashlib
from PIL import Image
import data_helper
import prior_factory
import model
from torchvision.utils import save_image
import seaborn as sns
import time
import tqdm
import torch
import torchvision
import matplotlib.pyplot as plt


def load_inception_model(inception_model_score, train_loader, dataset, image_size, environment):
    global device
    if environment == 'nuri':
        icm_path = './inception_model_info/'
    else:
        icm_path = '../../../inception_model_info/'
    inception_model_score.lazy_mode(True)
    real_images_info_file_name = hashlib.md5(str(train_loader.dataset).encode()).hexdigest() + '.pickle'
    os.makedirs('inception_model_info', exist_ok=True)
    if os.path.exists(icm_path + real_images_info_file_name):
        print("Using generated real image info.")
        inception_model_score.load_real_images_info(icm_path + real_images_info_file_name)
    else:
        inception_model_score.model_to(device)
        inception_model_score.lazy_forward(args, decoder, True, mapper)
        inception_model_score.calculate_real_image_statistics()
        inception_model_score.save_real_images_info(icm_path + real_images_info_file_name)
        inception_model_score.model_to('cpu')
    inception_model_score.freeze_layers()
    return inception_model_score


def inference_image(args, model_name, mapper, decoder, batch_size, latent_dim, distribution):
    device = args.device 
    model_name = args.model_name
    
    with torch.no_grad():
        if model_name == 'non-prior' :
            z = torch.randn(batch_size, latent_dim, device=device)
            result = decoder(mapper(z)).cpu()
        if model_name == 'mimic':
            z = torch.rand(batch_size, latent_dim).to(device) * 2 - 1
            result = decoder(mapper(z)).cpu()
        else:
            z = torch.tensor(prior_factory.get_sample(distribution, batch_size, latent_dim)).float().to(device)
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

def aae_latent_plt(z, E_x, fnum) : 
    fig1, ax1 = plt.subplots()
    sns.kdeplot(E_x, label='E(x):optimize', ax=ax1, alpha=0.6, color='r')
    sns.kdeplot(z, label='z:target', ax=ax1, alpha=0.6, color='b')
    ax1.legend()
    ax1.set_title('feature ' + str(fnum))
    return fig1    

def nonprior_latent_plt(z, E_x, M_z, fnum) : 
    fig1, ax1 = plt.subplots()
    sns.kdeplot(M_z, label='M(z):optimize',  ax=ax1, alpha=0.6, color='r')
    sns.kdeplot(E_x, label='E(x):target',  ax=ax1, alpha=0.6, color='b')
    sns.kdeplot(z, label='z:input', ax=ax1, alpha=0.6, color='g')
    ax1.legend()
    ax1.set_title('feature ' + str(fnum))
    return fig1 

def learningprior_latent_plt(z, E_x, M_z, fnum) : 
    fig1, ax1 = plt.subplots()
    sns.kdeplot(E_x, label='E(x):optimize',  ax=ax1, alpha=0.6, color='r')
    sns.kdeplot(M_z, label='M(z):target',  ax=ax1, alpha=0.6, color='b')
    sns.kdeplot(z, label='z:input', ax=ax1, alpha=0.6, color='g')
    ax1.legend()
    ax1.set_title('feature ' + str(fnum))
    return fig1 

def aae_latent_pca_plt(z, E_x, dim=1) : 
    fig1, ax1 = plt.subplots()
    all_data = torch.cat([z, E_x])
    U, S, V = torch.pca_lowrank(all_data)
    pca_z = torch.matmul(z, V[:, :dim])
    pca_Ex = torch.matmul(E_x, V[:, :dim])
    sns.kdeplot(pca_Ex.flatten().numpy(), label='E(x):optimize', alpha=0.6, color='r', ax=ax1)
    sns.kdeplot(pca_z.flatten().numpy(), label='Z:target', alpha=0.6, color='b', ax=ax1)
    ax1.legend()
    ax1.set_title('latent kde in pca to %d dim ' % dim)
    return fig1

def nonprior_latent_pca_plt(z, E_x, M_z, dim=1) : 
    fig1, ax1 = plt.subplots()
    all_data = torch.cat([E_x, M_z])
    U, S, V = torch.pca_lowrank(all_data)
    pca_z = torch.matmul(z, V[:, :dim])
    pca_Ex = torch.matmul(E_x, V[:, :dim])
    pca_Mz = torch.matmul(M_z, V[:, :dim])
    sns.kdeplot(pca_Mz.flatten().numpy(), label='M(z):optimize', alpha=0.6, color='r', ax=ax1)
    sns.kdeplot(pca_Ex.flatten().numpy(), label='E(x):target', alpha=0.6, color='b', ax=ax1)
    sns.kdeplot(pca_z.flatten().numpy(), label='z:input', alpha=0.6, color='g', ax=ax1)
    ax1.legend()
    ax1.set_title('latent kde in pca to %d dim ' % dim)
    return fig1

def learningprior_latent_pca_plt(z, E_x, M_z, dim=1) : 
    fig1, ax1 = plt.subplots()
    all_data = torch.cat([E_x, M_z])
    U, S, V = torch.pca_lowrank(all_data)
    pca_z = torch.matmul(z, V[:, :dim])
    pca_Ex = torch.matmul(E_x, V[:, :dim])
    pca_Mz = torch.matmul(M_z, V[:, :dim])
    sns.kdeplot(pca_Ex.flatten().numpy(), label='E(x):optimize', alpha=0.6, color='r', ax=ax1)
    sns.kdeplot(pca_Mz.flatten().numpy(), label='M(z):target', alpha=0.6, color='b', ax=ax1)
    sns.kdeplot(pca_z.flatten().numpy(), label='z:input', alpha=0.6, color='g', ax=ax1)
    ax1.legend()
    ax1.set_title('latent kde in pca to %d dim ' % dim)
    return fig1
    
def latent_plot(args, dataset, encoder, mapper, num=2048) :
    # gather x sample, E(x)
    train_loader, _ = data_helper.get_data(dataset, args.batch_size, args.image_size, args.environment)
    ex_list = []
    x_list = []
    for each_batch in train_loader : 
        with torch.no_grad() : 
            ex_list.append(encoder(each_batch[0].to(args.device)).cpu())
            x_list.append(each_batch[0])
        if args.batch_size * len(ex_list) >= num : break
    ex_tensor = torch.cat(ex_list)
    x_tensor = torch.cat(x_list)
    
    
    plt_list = []
    #case aae. plot{E(x), Gaussian}. E(x) imitate Gaussian. 
    if args.model_name in ['aae', 'mask_aae'] : 
        z = torch.randn(*ex_tensor.shape)
        pca_plt = aae_latent_pca_plt(z, ex_tensor)
        for i in range(z.size(1)) :
            plt_list.append(aae_latent_plt(z[:,i],ex_tensor[:,i], i))
       
    
    #case non-prior. plot{Gaussian, M(z), E(x)} M(z) imitate E(x).
    elif args.model_name in ['non-prior'] : 
        z = torch.randn(*ex_tensor.shape)
        M_z = mapper(z.to(args.device)).detach().cpu()
        pca_plt = nonprior_latent_pca_plt(z, ex_tensor, M_z)
        for i in range(z.size(1)) :
            plt_list.append(nonprior_latent_plt(z[:,i],ex_tensor[:,i], M_z[:,i], i))
        
    
    #case learning-prior. plot{Gaussian, M(z), E(x)} E(x) imitate M(z)
    elif args.model_name in ['learning-prior'] : 
        z = torch.randn(*ex_tensor.shape)
        M_z = mapper(z.to(args.device)).detach().cpu()
        pca_plt = learningprior_latent_pca_plt(z, ex_tensor, M_z)
        for i in range(z.size(1)) :
            plt_list.append(learningprior_latent_plt(z[:,i],ex_tensor[:,i], M_z[:,i], i))
        
    
        
    return plt_list, pca_plt


def log_index(wandb, i, log_dict) : 
    wandb.log(log_dict, step=i)

def log_index_with_inception_model(args, wandb, decoder, discriminator, encoder, i, inception_model_score, log_dict,
                                   dataset, distribution, latent_dim, model_name, environment, mapper=None):
    
    start_time = args.start_time
    end_time = time.time()
    run_time = end_time - start_time
    
    plt_list, pca_plt = latent_plot(args, dataset, encoder, mapper, num=2048)
    
    decoder, discriminator, encoder, mapper, inception_model_score = \
        swap_gpu_between_generative_model_and_inception_model(args, decoder, discriminator, encoder, inception_model_score,
                                                              mapper, generative_model_gpu=False)
    inception_model_score.lazy_forward(args, decoder, real_forward=False, mapper=mapper, gen_image_in_gpu=False)
    inception_model_score.calculate_fake_image_statistics()
    metrics, feature_pca_plot, real_pca, fake_pca = \
        inception_model_score.calculate_generative_score(feature_pca_plot=True)
    decoder, discriminator, encoder, mapper, inception_model_score = \
        swap_gpu_between_generative_model_and_inception_model(args, decoder, discriminator, encoder, inception_model_score,
                                                              mapper, generative_model_gpu=True)

    precision, recall, fid, inception_score_real, inception_score_fake, density, coverage = \
        metrics['precision'], metrics['recall'], metrics['fid'], metrics['real_is'], metrics['fake_is'], \
        metrics['density'], metrics['coverage']
    
    log_dict.update({'IsNet feature': wandb.Image(feature_pca_plot),
                   "precision": precision,
                   "recall": recall,
                   "fid": fid,
                   "inception_score_real": inception_score_real,
                   "inception_score_fake": inception_score_fake,
                   "density": density,
                   "coverage": coverage,
                   "latent kde each dim" : [wandb.Image(plt) for plt in plt_list],
                   "latent kde in pca" : wandb.Image(pca_plt),
                   "run time" : run_time
              })

    if decoder.has_mask_layer:
        mask_vector_2d_list = [[each_mask_element] for each_mask_element in decoder.mask_vector.cpu().detach().numpy()]
        mask_table = wandb.Table(data=mask_vector_2d_list, columns=["mask_value"])
        wandb.log({"mask_vector_hist": wandb.plot.histogram(mask_table, "mask_value", title="mask_vector_hist")}, step=i)
    inception_model_score.clear_fake()
    return encoder, decoder, discriminator, mapper, log_dict


def swap_gpu_between_generative_model_and_inception_model(args, decoder, discriminator, encoder, inception_model_score,
                                                          mapper, generative_model_gpu):
    device = args.device
    
    if not generative_model_gpu:
        # offload all GAN model to cpu and onload inception model to gpu
        encoder = encoder.to('cpu')
        decoder = decoder.to('cpu')
        discriminator = discriminator.to('cpu')
        if mapper is not None:
            mapper = mapper.to('cpu')
        inception_model_score.model_to(device)
    else:
        inception_model_score.model_to('cpu')
        encoder = encoder.to(device)
        decoder = decoder.to(device)
        if mapper is not None:
            mapper = mapper.to(device)
        discriminator = discriminator.to(device)
    return decoder, discriminator, encoder, mapper, inception_model_score

def imtensor_to_im(tensor, row=8, col=8, imshow=False, inverse_norm=True) : 
    invTrans = torchvision.transforms.Compose([ torchvision.transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.5, 1/0.5, 1/0.5 ]),
                                torchvision.transforms.Normalize(mean = [ -0.5, -0.5, -0.5 ],
                                                     std = [ 1., 1., 1. ]),
                               ])
    
    assert row*col == tensor.size(0)
    
    if inverse_norm : 
        tensor = invTrans(tensor)
        
    tensor = torch.clip(tensor, min=0, max=1.)
        
    grid_image = torchvision.utils.make_grid(tensor, row).permute(1,2,0)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_image)

    if imshow : 
        plt.show()
    else : 
        return fig
    


def generate_image_and_save_in_wandb(args, wandb, mapper, decoder, each_epoch, model_name, latent_dim, distribution, dataset):
    number_of_image = 64
    sample_tensor = inference_image(args, model_name, mapper, decoder, number_of_image, latent_dim, distribution)
    '''
    folder_name = '%s_%s' % (dataset, model_name)
    os.makedirs('images/%s' % folder_name, exist_ok=True)
    image_name = "images/%s/%d_epoch.png" % (folder_name, each_epoch)
    save_image(sampled_images.data, image_name, nrow=10, normalize=True)
    generated_image_file = Image.open(image_name)
    '''
    sample_fig = imtensor_to_im(sample_tensor, row=8, col=8, imshow=False, inverse_norm=True)
    
    wandb.log({'image': wandb.Image(sample_fig, caption='%s_epochs' % each_epoch)}, step=each_epoch)


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
    if args.model_name in ['mimic', 'non-prior']:
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
    
    device = args.device
    
    if args.load_pretrain_model:
        encoder, decoder, is_pretrained = load_pretrain_autoencoder_model(args, encoder, decoder, args.pretrain_epoch)
    else:
        is_pretrained = False

    if is_pretrained:
        print("Using pretrained autoencoder.")
    else:
        print("There are no pretrained autoencoder, start pretraining.")
        for i in range(0, args.pretrain_epoch):
            loss_r = 0
            for each_batch in tqdm.tqdm(train_loader, desc="pretrain AE[%d/%d]" % (i, args.pretrain_epoch)):
                each_batch = each_batch[0].to(device)
                loss_r = model.update_autoencoder(ae_optimizer, each_batch, encoder, decoder)
            print(f"epoch: {i}, loss: {loss_r}")
        save_pretrain_autoencoder_model(args, decoder, encoder, args.pretrain_epoch)
    return decoder, encoder


def log_and_write_pca(args, wandb, decoder, discriminator, encoder, i, inception_model_score, mapper, log_dict):
    generate_image_and_save_in_wandb(args, wandb, mapper, decoder, i, args.model_name, args.latent_dim, args.distribution,
                                         args.dataset)
    
    if i % args.log_interval == 0 or i == (args.epochs - 1):
        encoder, decoder, discriminator, mapper, log_dict = \
            log_index_with_inception_model(args, wandb, decoder, discriminator, encoder, i,
                                           inception_model_score, log_dict, args.dataset, args.distribution,
                                           args.latent_dim, args.model_name, args.environment, mapper)
    log_index(wandb, i, log_dict)
    
    '''
    if i == (args.epochs - 1):
        write_pca_data(args.model_name, args.dataset, args.image_size, args.distribution, i, real_pca, fake_pca)
        write_feature(args.model_name, args.dataset, args.image_size, args.distribution, i,
                      inception_model_score.real_feature_np,
                      inception_model_score.fake_feature_np)
    '''
    return decoder, discriminator, encoder, mapper


def force_log_and_write_pca(args, wandb, decoder, discriminator, encoder, i, inception_model_score, mapper, log_dict):
    #same as log_and_write_pca but no condition
    generate_image_and_save_in_wandb(args, wandb, mapper, decoder, i, args.model_name, args.latent_dim, args.distribution,
                                         args.dataset)
    encoder, decoder, discriminator, mapper, log_dict = \
        log_index_with_inception_model(args, wandb, decoder, discriminator, encoder, i,
                                       inception_model_score, log_dict, args.dataset, args.distribution,
                                       args.latent_dim, args.model_name, args.environment, mapper)
    
    log_index(wandb, i, log_dict)
    '''
    write_pca_data(args.model_name, args.dataset, args.image_size, args.distribution, i, real_pca, fake_pca)
    write_feature(args.model_name, args.dataset, args.image_size, args.distribution, i,
                      inception_model_score.real_feature_np,
                      inception_model_score.fake_feature_np)
    '''
    return decoder, discriminator, encoder, mapper
    


def timeout(time_limit, start_time) : 
    # if time_limit < run_time, return True
    # if time_limit > run_time, return False
    return int(time_limit) < (time.time() - start_time)

def str2bool(input_string):
    return bool(distutils.util.strtobool(input_string))

