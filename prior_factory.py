import numpy as np


def standard_normal(batch_size, latent_dim):
    normal_sampled = np.random.normal(0, 1, (batch_size, latent_dim))
    return normal_sampled


def uniform(batch_size, latent_dim):
    uniform_sampled = np.random.uniform(0, 1, (batch_size, latent_dim))
    return uniform_sampled


def gamma(batch_size, latent_dim, k=9, theta=0.5):
    gamma_sampled = np.random.gamma(k, theta, (batch_size, latent_dim))
    return gamma_sampled


def beta(batch_size, latent_dim, alpha=2, beta=2):
    beta_sampled = np.random.beta(alpha, beta, (batch_size, latent_dim))
    return beta_sampled


def chi(batch_size, latent_dim, k=2):
    chi_sampled = np.random.chisquare(k, (batch_size, latent_dim))
    return chi_sampled


def dirichlet(batch_size, alpha=(1, 2, 2, 1, 2, 2, 1, 2, 2, 1)):
    dirichlet_sampled = np.random.dirichlet(alpha, batch_size)
    return dirichlet_sampled


def laplace(batch_size, latent_dim, mu=0, b=2):
    laplace_sampled = np.random.laplace(mu, b, (batch_size, latent_dim))
    return laplace_sampled


def get_sample(distribution, batch_size, latent_dim, args=None):
    if distribution == 'standard_normal':
        return standard_normal(batch_size, latent_dim)
    elif distribution == 'uniform':
        return uniform(batch_size, latent_dim)
    elif distribution == 'gamma':
        return gamma(batch_size, latent_dim)
    elif distribution == 'beta':
        return beta(batch_size, latent_dim)
    elif distribution == 'chi':
        return chi(batch_size, latent_dim)
    elif distribution == 'dirichlet':
        return dirichlet(batch_size)
    elif distribution == 'laplace':
        return laplace(batch_size, latent_dim)
    else:
        return Exception('distribution input argument error')