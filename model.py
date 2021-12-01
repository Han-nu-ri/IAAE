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
    def __init__(self, nz=32, img_size=32, ngpu=1, ngf=64, nc=3):
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


def update_autoencoder(ae_optimizer, each_batch, encoder, decoder, return_encoded_feature=False):
    ae_optimizer.zero_grad()
    z_posterior = encoder(each_batch)
    fake_batch = decoder(z_posterior)
    pixel_wise_loss = torch.nn.L1Loss(reduction='sum')
    r_loss = pixel_wise_loss(fake_batch, each_batch)
    r_loss.backward()
    ae_optimizer.step()
    if return_encoded_feature:
        return r_loss, z_posterior.detach().cpu()
    return r_loss


def update_discriminator(d_optimizer, each_batch, encoder, discriminator, latent_dim, distribution):
    d_optimizer.zero_grad()
    batch_size = each_batch.size(0)
    z_prior = Variable(torch.FloatTensor(prior_factory.get_sample(distribution, batch_size, latent_dim))).cuda()
    z_posterior = encoder(each_batch)
    d_loss = -torch.mean(torch.log(discriminator(z_prior)) + torch.log(1 - discriminator(z_posterior)))
    d_loss.backward()
    d_optimizer.step()
    return d_loss.data


def update_generator(g_optimizer, each_batch, encoder, discriminator):
    g_optimizer.zero_grad()
    z_posterior = encoder(each_batch)
    g_loss = -torch.mean(torch.log(discriminator(z_posterior)))
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data


def update_aae(ae_optimizer, args, d_optimizer, decoder, discriminator, each_batch, encoder, g_optimizer, latent_dim):
    r_loss = update_autoencoder(ae_optimizer, each_batch, encoder, decoder)
    d_loss = update_discriminator(d_optimizer, each_batch, encoder, discriminator, latent_dim,
                                  args.distribution)
    g_loss = update_generator(g_optimizer, each_batch, encoder, discriminator)
    return d_loss, g_loss, r_loss


def get_aae_model_and_optimizer(latent_dim, image_size):
    encoder = Encoder(latent_dim, image_size).cuda()
    decoder = Decoder(latent_dim, image_size).cuda()
    mapper = None
    discriminator = Discriminator(latent_dim).cuda()
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


def train_mapper(encoder, mapper, device, lr, batch_size, encoded_feature_list):
    m_optimizer = torch.optim.Adam(mapper.parameters(), lr=lr, weight_decay=1e-3)
    encoder.eval()
    feature_tensor_dloader = encoded_feature_to_dl(torch.cat(encoded_feature_list), batch_size)
    for each_batch, label_feature in tqdm.tqdm(feature_tensor_dloader):
        uniform_input = each_batch.to(device)
        sorted_encoded_feature = label_feature.to(device)
        m_loss = update_mimic(m_optimizer, uniform_input, sorted_encoded_feature, mapper)
    return m_loss
