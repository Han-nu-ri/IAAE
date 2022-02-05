import torch
import torch.nn as nn
from torch.autograd import Variable
import itertools
import prior_factory
import tqdm


class Encoder(nn.Module):
    def __init__(self, nz=32, img_size=32, ngpu=1, ndf=64, nc=3):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        kernel = 2 * (img_size // 32)
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            # nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Conv2d(ndf * 8, nz, kernel, 1, 0, bias=False),
            # state size. 1x1x1
        )
        self.nz = nz

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
        else:
            output = self.model(input)
        return output.view(-1, self.nz)


class Decoder(nn.Module):
    def __init__(self, nz=32, img_size=32, ngpu=1, ngf=64, nc=3, has_mask_layer=False):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        kernel = 2 * img_size // 32

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.ConvTranspose2d(nz, ngf * 8, kernel, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            # state size. (nc) x 32 x 32
        )

        self.has_mask_layer = has_mask_layer
        if has_mask_layer:
            self.theta_vector = torch.rand(nz, requires_grad=True)
            self.mask_vector = torch.max(torch.zeros_like(self.theta_vector), 1 - torch.exp(-self.theta_vector))

    def forward(self, input):
        input = input.view(-1, self.nz, 1, 1)
        output = self.model(input)
        return output


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


class Mimic(nn.Module):
    def __init__(self, in_feature, out_feature, hidden_feature, hidden_layer):
        super(Mimic, self).__init__()
        assert in_feature == out_feature
        net_list = [self.single_line(hidden_feature, hidden_layer) for i in range(in_feature)]
        self.net_list = torch.nn.ModuleList(net_list)

    def single_line(self, hidden_feature, hidden_layer):
        linear_list = []
        linear_list.append(nn.Linear(1, hidden_feature))
        linear_list.append(nn.CELU())
        for i in range(hidden_layer):
            linear_list.append(nn.Linear(hidden_feature, hidden_feature))
            linear_list.append(nn.CELU(True))
        linear_list.append(nn.Linear(hidden_feature, 1))
        return torch.nn.ModuleList(linear_list)

    def forward(self, x):
        assert x.size(1) == len(self.net_list)
        predict_list = []
        for each_dim in range(x.size(1)):
            x_each_dim = x[:, each_dim].view(-1, 1)
            net_each_dim = self.net_list[each_dim]
            predict_each_dim = x_each_dim
            for each_layer in net_each_dim:
                predict_each_dim = each_layer(predict_each_dim)
            predict_list.append(predict_each_dim)
        return torch.cat(predict_list, dim=1)


class Mapping(nn.Module):
    def __init__(self, in_out_nz, mapper_inter_nz, mapper_inter_layer):
        super(Mapping, self).__init__()
        linear = nn.ModuleList()
        linear.append(nn.BatchNorm1d(in_out_nz))
        if mapper_inter_layer >= 2:
            linear.append(nn.Linear(in_features=in_out_nz, out_features=mapper_inter_nz))
            linear.append(nn.BatchNorm1d(mapper_inter_nz))
            linear.append(nn.CELU())
            for i in range(mapper_inter_layer-2):
                linear.append(nn.Linear(in_features=mapper_inter_nz, out_features=mapper_inter_nz))
                linear.append(nn.CELU())

            linear.append(nn.Linear(in_features=mapper_inter_nz, out_features=in_out_nz))
        else:
            linear.append(nn.Linear(in_features=in_out_nz, out_features=in_out_nz))
        self.linear = linear
        
    def forward(self, input):
        for layer in self.linear:
            input = layer(input)
        return input

# code from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/003efc4c8819de47ff11b5a0af7ba09aee7f5fc1/models/networks.py#L538
import torch.nn as nn
import functools

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)] 
        # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)      
    

def update_autoencoder(ae_optimizer, each_batch, encoder, decoder, return_encoded_feature=False, return_encoded_feature_gpu=False, flag_retain_graph=True):
    ae_optimizer.zero_grad()
    z_posterior = encoder(each_batch)
    if decoder.has_mask_layer:
        z_posterior = torch.mul(z_posterior, decoder.mask_vector)
    fake_batch = decoder(z_posterior)
    pixel_wise_loss = torch.nn.L1Loss(reduction='mean')
    r_loss = pixel_wise_loss(fake_batch, each_batch)
    if decoder.has_mask_layer:
        r_loss += 0.1 * torch.sum(torch.abs(torch.mul(1 - decoder.mask_vector, decoder.mask_vector)))
    r_loss.backward(retain_graph=flag_retain_graph)
    ae_optimizer.step()
    if return_encoded_feature:
        return {'r_loss': r_loss.item()}, z_posterior.detach().cpu()
    if return_encoded_feature_gpu:
        return {'r_loss': r_loss.item()}, z_posterior.detach()
    return {'r_loss': r_loss.item()}


def update_discriminator(args, d_optimizer, each_batch, encoder, decoder, discriminator):
    d_optimizer.zero_grad()
    batch_size = each_batch.size(0)
    z_prior = Variable(torch.FloatTensor(prior_factory.get_sample(args.distribution, batch_size, args.latent_dim))).to(args.device)
    z_posterior = encoder(each_batch)
    if decoder.has_mask_layer:
        z_prior = torch.mul(z_prior, decoder.mask_vector)
        z_posterior = torch.mul(z_posterior, decoder.mask_vector)
    d_loss = -torch.mean(torch.log(discriminator(z_prior)) + torch.log(1 - discriminator(z_posterior)))
    d_loss.backward(retain_graph=True)
    d_optimizer.step()
    return d_loss.data

def update_discriminator_with_mappedz(args, d_optimizer, each_batch, encoder, decoder, mapper, discriminator):
    d_optimizer.zero_grad()
    batch_size = each_batch.size(0)
    z_prior = Variable(torch.FloatTensor(prior_factory.get_sample(args.distribution, batch_size, args.latent_dim))).to(args.device)
    mappedz = mapper(z_prior)
    z_posterior = encoder(each_batch)
    if decoder.has_mask_layer:
        z_prior = torch.mul(z_prior, decoder.mask_vector)
        z_posterior = torch.mul(z_posterior, decoder.mask_vector)
    d_loss = -torch.mean(torch.log(discriminator(z_prior)) + torch.log(1 - discriminator(mappedz)))
    d_loss.backward(retain_graph=True)
    d_optimizer.step()
    return d_loss.data

def update_generator(g_optimizer, each_batch, encoder, decoder, discriminator):
    g_optimizer.zero_grad()
    z_posterior = encoder(each_batch)
    if decoder.has_mask_layer:
        z_posterior = torch.mul(z_posterior, decoder.mask_vector)
    g_loss = -torch.mean(torch.log(discriminator(z_posterior)))
    g_loss.backward(retain_graph=True)
    g_optimizer.step()
    return g_loss.data



def update_aae(ae_optimizer, args, d_optimizer, decoder, discriminator, each_batch, encoder, g_optimizer, latent_dim):
    log = update_autoencoder(ae_optimizer, each_batch, encoder, decoder)
    d_loss = update_discriminator(args, d_optimizer, each_batch, encoder, decoder, discriminator)
    g_loss = update_generator(g_optimizer, each_batch, encoder, decoder, discriminator)
    log.update({'d_loss': d_loss, 'g_loss': g_loss})
    return log

def update_aae_with_mappedz(args, ae_optimizer, d_optimizer, decoder, discriminator, mapper, each_batch, encoder, g_optimizer):
    log = update_autoencoder(ae_optimizer, each_batch, encoder, decoder)
    d_loss = update_discriminator_with_mappedz(args, d_optimizer, each_batch, encoder, decoder, \
                                               mapper, discriminator)
    g_loss = update_generator(g_optimizer, each_batch, encoder, decoder, discriminator)
    log.update({'d_loss': d_loss, 'g_loss': g_loss})
    return log

def update_ae_and_w_for_aaae(args, each_batch, encoder, decoder, ae_optimizer,
                            discriminator_forpl, dpl_optimizer):
    
    '''update encoder/decoder'''
    
    # Discriminator_w(x) --> minimize(negative)   ??why??
    # distance(decoder(encoder(x)), x) --> minimize
    
    real_image = each_batch.to(args.device)
    Dwx = discriminator_forpl(real_image)
    
    reconstruct_image = decoder(encoder(real_image))
    pixel_wise_loss = torch.nn.L1Loss(reduction='mean')
    distance = pixel_wise_loss(reconstruct_image, real_image)
    
    Dwx_loss = Dwx.mean()
    
    encoder_decoder_loss = Dwx_loss + distance  * args.lambda1
    
    ae_optimizer.zero_grad()
    encoder_decoder_loss.backward()
    ae_optimizer.step()
    
    
    
    '''update discriminator by patch discriminator'''
    Dwx = discriminator_forpl(real_image)
    Dwgx = discriminator_forpl(decoder(encoder(real_image)))
    
    sigmoid = torch.nn.Sigmoid()
    
    w_loss = -(torch.log(sigmoid(Dwx))+ torch.log(1-sigmoid(Dwgx))).mean()
    
    dpl_optimizer.zero_grad()
    w_loss.backward()
    dpl_optimizer.step()
    
    log = {}
    log.update({'update_encoder_decoder/Dwx_loss': Dwx_loss.item(),
                'update_encoder_decoder/d': distance.item(),
                'update_encoder_encoder/encoder_decoder_loss': encoder_decoder_loss.item(),
                
                'update_Dw/Dwx':Dwx.mean().item(),
                'update_Dw/Dwgx':Dwgx.mean().item(),
                'update_Dw/w_loss':w_loss.item() })
    return log

def update_mapper_and_gamma(args, each_batch, encoder, discriminator, mapper, d_optimizer, m_optimizer) :
    
    batch_size = each_batch.size(0)
    real_image = each_batch.to(args.device)
    
    #define Wasserstein distance for generalize
    #code from https://uos-deep-learning.tistory.com/16
    #note that real_data is c, not image!
    def calc_gradient_penalty(args, netD, real_data, c_g):
        # GP strength
        LAMBDA = args.lambda2

        b_size = real_data.size()[0]

        # Calculate interpolation
        #alpha = torch.rand(b_size, 1, 1, 1)  <-- is original code. but we do interpolation latent code, not image
        alpha = torch.rand(b_size, 1)
        alpha = alpha.expand_as(real_data)
        alpha = alpha.to(args.device)

        interpolated = alpha * real_data.data + (1 - alpha) * c_g
        interpolated = interpolated.clone().detach().requires_grad_(True)
        interpolated = interpolated.to(args.device)

        # Calculate probability of interpolated examples
        prob_interpolated = netD(interpolated).mean()

        # Calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda(),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        gradients = gradients.view(b_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

        # Return gradient penalty
        return LAMBDA * ((gradients_norm - 1) ** 2).mean()
    
    for repeat in range(args.aaae_k) : 
        # first, update D gamma
        d_optimizer.zero_grad()
        noise = torch.randn(batch_size, args.latent_dim, device=args.device)
        c_g = mapper(noise)
        c = encoder(real_image)
        
        Drc = discriminator(c)
        Drcg = discriminator(c_g)
    
        reg_value = calc_gradient_penalty(args, discriminator, c, c_g)
        
        Dgamma_loss = (-Drc + Drcg + reg_value).mean()
        Dgamma_loss.backward()
        d_optimizer.step()
        
        c_g = mapper(noise)
        Drcg_new = discriminator(c_g)
        mapper_loss = -Drcg_new.mean()
        m_optimizer.zero_grad()
        mapper_loss.backward()
        m_optimizer.step()
        
   
    return {'update_mapper_gamma/Drc' : Drc.mean().item(),
            'update_mapper_gamma/Drcg' : Drcg.mean().item(),
            'update_mapper_gamma/reg_value' : reg_value.item(),
            'update_mapper_gamma/Dgamma_loss' : Dgamma_loss.item(),
            'update_mapper_gamma/Drcg_after_update_gamma' : Drcg_new.mean().item(),
            'update_mapper_gamma/mapper_loss' : mapper_loss.item()}


def update_mapper_with_discriminator_forpl(args, dpl_optimizer, decoder_optimizer, m_optimizer, discriminator_forpl, decoder, mapper, each_batch) :
    #discriminator_forpl은 x를 True로, decoder(mapper(noise))를 False로 구별한다
    #mapper와 decoder는 discriminator를 속이려고 한다
    
    batch_size = each_batch.size(0)
    each_batch = each_batch.view(batch_size, -1)
    noise = torch.randn(batch_size, args.latent_dim, device=args.device)
    mapped_noise = mapper(noise)
    fake_img = decoder(mapped_noise).view(batch_size,-1) # discriminator sturcture cannot control 2D image but only flatten tensor
    
    #train discriminator. make D(x) -->Positive and D(decoder(mapped_noise))-->Negative
    d_loss = -torch.mean(torch.log(discriminator_forpl(each_batch)) + torch.log(1 - discriminator_forpl(fake_img)))
    dpl_optimizer.zero_grad()
    d_loss.backward()
    dpl_optimizer.step()
    
    #train mapper and decoder. make D(decoder(mapped_noise))-->Positive
    mapped_noise = mapper(noise)
    fake_img = decoder(mapped_noise).view(batch_size,-1)
    m_loss = -torch.mean(torch.log(discriminator_forpl(fake_img)))
    m_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    m_loss.backward()
    m_optimizer.step()
    decoder_optimizer.step()
    
    return {'D loss for prior' : d_loss}


def get_nonprior_model_and_optimizer(args):
    mapper = Mapping(args.latent_dim, args.mapper_inter_nz, args.mapper_inter_layer).to(args.device)
    m_optimizer = torch.optim.Adam(mapper.parameters(), lr=1e-4)
    return mapper, m_optimizer

def get_learning_prior_model_and_optimizer(args):
    mapper = Mapping(args.latent_dim, args.mapper_inter_nz, args.mapper_inter_layer).to(args.device)
    m_optimizer = torch.optim.Adam(mapper.parameters(), lr=1e-4)
    discriminator_forpl = Discriminator(args.image_size * args.image_size * 3).to(args.device)
    dpl_optimizer = torch.optim.Adam(discriminator_forpl.parameters(), lr=1e-4)
    return mapper, m_optimizer, discriminator_forpl, dpl_optimizer

def get_aaae_model_and_optimizer(args):
    mapper = Mapping(args.latent_dim, args.mapper_inter_nz, args.mapper_inter_layer).to(args.device)
    m_optimizer = torch.optim.Adam(mapper.parameters(), lr=1e-4)
    discriminator_forpl = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d).to(args.device)
    dpl_optimizer = torch.optim.Adam(discriminator_forpl.parameters(), lr=1e-4)
    return mapper, m_optimizer, discriminator_forpl, dpl_optimizer

def get_aae_model_and_optimizer(args):
    encoder = Encoder(args.latent_dim, args.image_size).to(args.device)
    decoder = Decoder(args.latent_dim, args.image_size, has_mask_layer=args.has_mask_layer).to(args.device)
    mapper = None
    discriminator = Discriminator(args.latent_dim).to(args.device)
    ae_optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=1e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-4)
    return ae_optimizer, d_optimizer, decoder, discriminator, encoder, g_optimizer, mapper


def encoded_feature_to_dl(encoded_feature_tensor, batch_size):
    #각 차원의 feature를 각 차원별로 sort한 tensor가 필요함
    sorted_encoded_feature_tensor = torch.empty(encoded_feature_tensor.shape)
    for each_dim in range(sorted_encoded_feature_tensor.size(1)):
        sorted_encoded_feature_tensor[:, each_dim] = torch.sort(encoded_feature_tensor[:, each_dim])[0]

    # linspace로 입력할 -1~1사이의 균일한 값이 필요함
    uniform_input = torch.empty(sorted_encoded_feature_tensor.shape)
    for each_dim in range(sorted_encoded_feature_tensor.size(1)):
        uniform_input[:, each_dim] = torch.linspace(-1, 1, sorted_encoded_feature_tensor.size(0))

    feature_tensor_ds = torch.utils.data.TensorDataset(uniform_input, sorted_encoded_feature_tensor)
    feature_tensor_dloader = torch.utils.data.DataLoader(feature_tensor_ds, batch_size=batch_size, shuffle=True)
    return feature_tensor_dloader


def update_mimic(m_optimizer, uniform_input, sorted_encoded_feature, mapper):
    mse = torch.nn.MSELoss(reduction='sum')
    predict = mapper(uniform_input)
    loss = mse(predict, sorted_encoded_feature)
    m_optimizer.zero_grad()
    loss.backward()
    m_optimizer.step()
    return loss.item()


def update_posterior_part(args, mapper, discriminator, m_optimizer, d_optimizer, encoded_feature) : 
    #discriminator는 encoded feature를 True로, mapper(noise)를 False로 구별한다
    #mapper는 discriminator를 속이려고 한다
    
    batch_size = encoded_feature.size(0)
    noise = torch.randn(batch_size, args.latent_dim, device=args.device)
    mapped_noise = mapper(noise)
    
    #train discriminator. make D(encoded_feature)-->Positive and D(mapped_noise)-->Negative
    D_Ex = discriminator(encoded_feature)
    D_Mz = discriminator(mapped_noise)
    d_loss = -torch.mean(torch.log(D_Ex) + torch.log(1 - D_Mz))
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    log = {'before update D/D_Ex': D_Ex.mean().item(),
           'before update D/D_Mz': D_Mz.mean().item()}
    
    #train mapper. make D(mapped_noise)-->Positive
    noise = torch.randn(batch_size, args.latent_dim, device=args.device)
    D_Mz = discriminator(mapper(noise))
    m_loss = -torch.mean(torch.log(D_Mz))
    m_optimizer.zero_grad()
    m_loss.backward()
    m_optimizer.step()

    log.update({'after update D/D_Mz': D_Mz.mean().item(),
                'd_loss': d_loss.item(),
                'm_loss': m_loss.item()})

    return log
    


def train_mapper(args, encoder, mapper, encoded_feature_list):
    m_optimizer = torch.optim.Adam(mapper.parameters(), lr=args.lr, weight_decay=1e-3)
    encoder.eval()
    feature_tensor_dloader = encoded_feature_to_dl(torch.cat(encoded_feature_list), args.batch_size)
    for each_batch, label_feature in tqdm.tqdm(feature_tensor_dloader):
        uniform_input = each_batch.to(args.device)
        sorted_encoded_feature = label_feature.to(args.device)
        m_loss = update_mimic(m_optimizer, uniform_input, sorted_encoded_feature, mapper)
    return {'m_loss' : m_loss}
