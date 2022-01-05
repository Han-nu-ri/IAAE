import torch
from scipy.stats import entropy
from scipy import linalg
import numpy as np
import prdc
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import data_helper
import prior_factory


class GenerativeModelScore:
    def __init__(self):

        self.inception_model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
        self.inception_model.forward = self._forward
        self.inception_model.eval()

        self.real_predict_softmax = None
        self.real_feature = None

        self.fake_predict_softmax = None
        self.fake_feature = None

        self.lazy = False

        self.hidden_representations_of_true = []

    def _forward(self, x):
        import torchvision

        if x.size(1) != 3:
            x = self.inception_model._transform_input(x)

        resize = torchvision.transforms.Resize((299, 299))
        x = resize(x)

        # N x 3 x 299 x 299
        x = self.inception_model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception_model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception_model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception_model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception_model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception_model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception_model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception_model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception_model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception_model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception_model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception_model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception_model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception_model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception_model.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception_model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception_model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception_model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.inception_model.avgpool(x)
        # N x 2048 x 1 x 1
        feature = x.detach()
        x = self.inception_model.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        x = self.inception_model.fc(x)
        # N x 1000 (num_classes)
        return x, feature

    def predict_to_inception_score(self, predict, splits=1):
        preds = torch.softmax(predict, dim=1).numpy()

        # Now compute the mean kl-div
        split_scores = []

        for k in range(splits):
            part = preds
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))

        return np.mean(split_scores), np.std(split_scores)

    def trainloaderinfo_to_hashedname(self, train_loader):
        import hashlib
        dataset_info = str(train_loader.dataset).split('\n')
        name, datapoints, split = dataset_info[0:3]
        transform = dataset_info[4:]
        except_root_info = [name, datapoints, split] + transform
        name = hashlib.md5(str(except_root_info).encode()).hexdigest() + '.pickle'
        return name

    def feature_to_mu_sig(self, act):
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def model_to(self, device):
        self.inception_model = self.inception_model.to(device)

    def lazy_mode(self, tf):
        self.lazy = tf

    def put_real(self, real_images):
        if not self.lazy:
            self.real_forward(real_images)

    def real_forward(self, real_images):
        real_predict_softmax, real_feature = self.analysis_softmax_and_feature(real_images)
        if self.real_predict_softmax is None:
            self.real_predict_softmax = real_predict_softmax.detach().cpu()
            self.real_feature = real_feature.detach().cpu()
        else:
            self.real_predict_softmax = torch.cat([self.real_predict_softmax, real_predict_softmax.detach().cpu()])
            self.real_feature = torch.cat([self.real_feature, real_feature.detach().cpu()])

    def put_fake(self, fake_images):
        if not self.lazy:
            self.fake_forward(fake_images)

    def fake_forward(self, fake_images):
        fake_predict_softmax, fake_feature = self.analysis_softmax_and_feature(fake_images)
        if self.fake_predict_softmax is None:
            self.fake_predict_softmax = fake_predict_softmax.detach().cpu()
            self.fake_feature = fake_feature.detach().cpu()
        else:
            self.fake_predict_softmax = torch.cat([self.fake_predict_softmax, fake_predict_softmax.detach().cpu()])
            self.fake_feature = torch.cat([self.fake_feature, fake_feature.detach().cpu()])

    def lazy_forward(self, dataset, image_size=32, batch_size=16,
                     decoder=None, distribution=None, latent_dim=None, real_forward=True, device='cpu',
                     model_name='aae', mapper=None, gen_image_in_gpu=False):
        assert self.lazy, "lazy_forward only run in lazy mode. call lazy_mode() first."
        train_loader, _ = data_helper.get_data(dataset, batch_size, image_size)
        if real_forward:
            print("generate real images info")
            for each_batch in tqdm.tqdm(train_loader):
                self.real_forward(each_batch[0].to(device))
        else:
            print("generate fake images info")
            if gen_image_in_gpu : 
                
                fake_images_list = []
                
                decoder = decoder.to(device)
                if isinstance(mapper, torch.nn.Module) : mapper = mapper.to(device)
                
                
                for each_batch in tqdm.tqdm(train_loader, desc='gen_fake'):
                    with torch.no_grad() : 
                        if model_name == "mimic":
                            z = torch.rand(batch_size, latent_dim, device=device) * 2 - 1
                            fake_images = decoder(mapper(z))
                        elif model_name == 'non-prior':
                            z = torch.randn(batch_size, latent_dim, device=device)
                            fake_images = decoder(mapper(z))
                        else:
                            z = torch.FloatTensor(prior_factory.get_sample(distribution, batch_size, latent_dim)).to(device)
                            if decoder.has_mask_layer:
                                z = torch.mul(z, decoder.mask_vector)
                            fake_images = decoder(z)                        
                        fake_images_list.append(fake_images.cpu())
                
                decoder = decoder.to('cpu')
                if isinstance(mapper, torch.nn.Module) : mapper = mapper.to('cpu')
                
                
                fake_predict_softmax_list, fake_feature_list = [], []
                for index in tqdm.tqdm(range(len(fake_images_list)), desc='gen_feature') : 
                    fake_images_gpu = fake_images_list[index].to(device)
                    with torch.no_grad() : 
                        fake_predict_softmax, fake_feature = self.analysis_softmax_and_feature(fake_images)
                        fake_predict_softmax_list.append(fake_predict_softmax.cpu())
                        fake_feature_list.append(fake_feature.cpu())
                        fake_images_list[index] = None
                        
                self.fake_predict_softmax = torch.cat(fake_predict_softmax_list)
                self.fake_feature = torch.cat(fake_feature_list)
  
            else : 
                
                for each_batch in tqdm.tqdm(train_loader):
                    if model_name == "mimic":
                        z = torch.rand(batch_size, latent_dim) * 2 - 1
                        fake_images = decoder(mapper(z)).to(device)
                    elif model_name == 'non-prior':
                        z = torch.randn(batch_size, latent_dim)
                        fake_images = decoder(mapper(z)).to(device)
                    else:
                        z = torch.FloatTensor(prior_factory.get_sample(distribution, batch_size, latent_dim))
                        if decoder.has_mask_layer:
                            z = torch.mul(z, decoder.mask_vector.cpu())
                        fake_images = decoder(z).to(device)
                    self.fake_forward(fake_images)

    def save_real_images_info(self, file_name='real_images_info.pickle'):
        with open(file_name, 'wb') as f:
            pickle.dump((self.real_inception_score, self.real_feature_np, (self.real_mu, self.real_sigma)), f)

    def load_real_images_info(self, file_name='real_images_info.pickle'):
        with open(file_name, 'rb') as f:
            (self.real_inception_score, self.real_feature_np, (self.real_mu, self.real_sigma)) = \
                pickle.load(f)

    def calculate_real_image_statistics(self):
        self.real_inception_score = self.predict_to_inception_score(self.real_predict_softmax)[0]
        self.real_feature_np = self.real_feature.view(-1, 2048).numpy()
        self.real_mu, self.real_sigma = self.feature_to_mu_sig(self.real_feature_np)

    def calculate_fake_image_statistics(self):
        self.fake_inception_score = self.predict_to_inception_score(self.fake_predict_softmax)[0]
        self.fake_feature_np = self.fake_feature.view(-1, 2048).numpy()
        self.fake_mu, self.fake_sigma = self.feature_to_mu_sig(self.fake_feature_np)

    def clear_fake(self):
        self.fake_predict_softmax = None
        self.fake_feature = None
        self.fake_mu, self.fake_sigma = None, None

    def calculate_generative_score(self, feature_pca_plot=False):
        fid = self.calculate_frechet_distance(self.real_mu, self.real_sigma, self.fake_mu, self.fake_sigma)
        real_pick = np.random.permutation(self.real_feature_np)[:10000]
        fake_pick = np.random.permutation(self.fake_feature_np)[:10000]
        metrics = prdc.compute_prdc(real_features=real_pick, fake_features=fake_pick, nearest_k=5)
        metrics['fid'] = fid
        metrics['real_is'] = self.real_inception_score
        metrics['fake_is'] = self.fake_inception_score

        if feature_pca_plot:

            real_feature = torch.tensor(self.real_feature_np)
            fake_feature = torch.tensor(self.fake_feature_np)

            fake_real = torch.cat((real_feature, fake_feature))

            U, S, V = torch.pca_lowrank(fake_real)
            real_pick = torch.tensor(np.random.permutation(self.real_feature_np)[:2048])
            fake_pick = torch.tensor(np.random.permutation(self.fake_feature_np)[:2048])

            real_pca = torch.matmul(real_pick, V[:, :3])
            fake_pca = torch.matmul(fake_pick, V[:, :3])
            
            
            plt.clf()
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(real_pca[:,0], real_pca[:,1], real_pca[:,2], alpha=0.2, label='real', zorder=0)
            ax.scatter(fake_pca[:,0], fake_pca[:,1], fake_pca[:,2], alpha=0.2, label='fake', zorder=10)

            mean_x_fake, mean_y_fake, mean_z_fake = torch.mean(fake_pca[:,0]), torch.mean(fake_pca[:,1]), torch.mean(fake_pca[:,2])
            mean_x_real, mean_y_real, mean_z_real = torch.mean(real_pca[:,0]), torch.mean(real_pca[:,1]), torch.mean(real_pca[:,2])
            diff = ((mean_x_fake-mean_x_real)**2 + (mean_y_fake-mean_y_real)**2 + (mean_z_fake-mean_z_real)**2)**0.5

            ann_x, ann_y, ann_z = (mean_x_fake+mean_x_real)/2, (mean_y_fake+mean_y_real)/2, (mean_z_fake+mean_z_real)/2

            ax.plot((mean_x_real, mean_x_fake), (mean_y_real, mean_y_fake), (mean_z_real, mean_z_fake), color = 'black',  lw = 3, zorder=15, marker='*', linestyle='--')
            ax.text(ann_x, ann_y, ann_z, ' %.1f '%diff, 'y', fontsize=20, zorder=15)

            ax.legend()
            ax.set_title('PCA to 3D feature scatter')
            
            
            return metrics, fig, real_pca, fake_pca
        else:
            return metrics

    def analysis_softmax_and_feature(self, images):
        return self.inception_model(images)

    def freeze_layers(self):
        # Freeze layers
        for child_layer in self.inception_model.children():
            for each_parameters_in_child_layer in child_layer.parameters():
                each_parameters_in_child_layer.requires_grad = False

    def get_hidden_representation(self, x, real=False, epoch=1, batch=None):
        if real is True and epoch > 0:
            return self.hidden_representations_of_true[batch]

        import torchvision

        if x.size(1) != 3:
            x = self.inception_model._transform_input(x)

        resize = torchvision.transforms.Resize((299, 299))
        x = resize(x)

        # N x 3 x 299 x 299
        x = self.inception_model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception_model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception_model.Conv2d_2b_3x3(x)
        # Adaptive average pooling
        hidden_representation = self.inception_model.avgpool(x)

        if real is True and epoch == 0:
            self.hidden_representations_of_true.append(hidden_representation)
        return hidden_representation
