import hashlib
import os
import tensorflow as tf
import torch

import data_helper
import wandb
from tensorflow import keras
import matplotlib.pyplot as plt
import generative_model_score
from PIL import Image
import matplotlib
matplotlib.use('Agg')


def sampling(args):
    mean, logsigma = args
    epsilon = keras.backend.random_normal(shape=keras.backend.shape(mean))
    return mean + tf.exp(logsigma / 2) * epsilon


def encoder(image_size, depth, kernel_size, latent_depth):
    input_E = keras.layers.Input(shape=(image_size, image_size, 3))

    X = keras.layers.Conv2D(filters=depth * 2, kernel_size=kernel_size, strides=2, padding='same')(input_E)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2D(filters=depth * 4, kernel_size=kernel_size, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2D(filters=depth * 8, kernel_size=kernel_size, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2D(filters=depth * 8, kernel_size=kernel_size, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(latent_depth)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    mean = keras.layers.Dense(latent_depth, activation="tanh")(X)
    logsigma = keras.layers.Dense(latent_depth, activation="tanh")(X)
    latent = keras.layers.Lambda(sampling, output_shape=(latent_depth,))([mean, logsigma])

    kl_loss = 1 + logsigma - keras.backend.square(mean) - keras.backend.exp(logsigma)
    kl_loss = keras.backend.mean(kl_loss, axis=-1)
    kl_loss *= -0.5

    return keras.models.Model(input_E, [latent, kl_loss])


def generator(image_size, depth, kernel_size, latent_depth):
    input_G = keras.layers.Input(shape=(latent_depth,))

    X = keras.layers.Dense(image_size // 16 * image_size // 16 * depth * 16)(input_G)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    X = keras.layers.Reshape((image_size // 16, image_size // 16, depth * 16))(X)

    X = keras.layers.Conv2DTranspose(filters=depth * 8, kernel_size=kernel_size, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2DTranspose(filters=depth * 8, kernel_size=kernel_size, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2DTranspose(filters=depth * 4, kernel_size=kernel_size, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2DTranspose(filters=depth, kernel_size=kernel_size, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2D(filters=3, kernel_size=kernel_size, padding='same')(X)
    X = keras.layers.Activation('sigmoid')(X)

    return keras.models.Model(input_G, X)


def discriminator(image_size, depth, kernel_size):
    input_D = keras.layers.Input(shape=(image_size, image_size, 3))

    X = keras.layers.Conv2D(filters=depth, kernel_size=kernel_size, strides=2, padding='same')(input_D)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2D(filters=depth * 4, kernel_size=kernel_size, strides=2, padding='same')(input_D)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)
    X = keras.layers.BatchNormalization()(X)

    X = keras.layers.Conv2D(filters=depth * 8, kernel_size=kernel_size, strides=2, padding='same')(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Conv2D(filters=depth * 8, kernel_size=kernel_size, padding='same')(X)
    inner_output = keras.layers.Flatten()(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    X = keras.layers.Flatten()(X)
    X = keras.layers.Dense(depth * 8)(X)
    X = keras.layers.BatchNormalization()(X)
    X = keras.layers.LeakyReLU(alpha=0.2)(X)

    output = keras.layers.Dense(1)(X)

    return keras.models.Model(input_D, [output, inner_output])


@tf.function
def train_step_vaegan(x, batch_size, latent_depth, E, G, D, E_opt, G_opt, D_opt):
    latent_r = tf.random.normal((batch_size, latent_depth))
    with tf.GradientTape(persistent=True) as tape:
        latent, kl_loss = E(x)
        fake = G(latent)
        dis_fake, dis_inner_fake = D(fake)
        dis_fake_r, _ = D(G(latent_r))
        dis_true, dis_inner_true = D(x)

        vae_inner = dis_inner_fake - dis_inner_true
        vae_inner = vae_inner * vae_inner

        mean, var = tf.nn.moments(E(x)[0], axes=0)
        var_to_one = var - 1

        normal_loss = tf.reduce_mean(mean * mean) + tf.reduce_mean(var_to_one * var_to_one)

        kl_loss = tf.reduce_mean(kl_loss)
        vae_diff_loss = tf.reduce_mean(vae_inner)
        f_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake), dis_fake))
        r_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.zeros_like(dis_fake_r), dis_fake_r))
        t_dis_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.ones_like(dis_true), dis_true))
        gan_loss = (0.5 * t_dis_loss + 0.25 * f_dis_loss + 0.25 * r_dis_loss)
        vae_loss = tf.reduce_mean(tf.abs(x - fake))
        normal_coef = 0.1
        kl_coef = 0.01
        E_loss = vae_diff_loss + kl_coef * kl_loss + normal_coef * normal_loss
        inner_loss_coef = 1
        G_loss = inner_loss_coef * vae_diff_loss - gan_loss
        D_loss = gan_loss

    E_grad = tape.gradient(E_loss, E.trainable_variables)
    G_grad = tape.gradient(G_loss, G.trainable_variables)
    D_grad = tape.gradient(D_loss, D.trainable_variables)
    del tape
    E_opt.apply_gradients(zip(E_grad, E.trainable_variables))
    G_opt.apply_gradients(zip(G_grad, G.trainable_variables))
    D_opt.apply_gradients(zip(D_grad, D.trainable_variables))

    return E_loss, D_loss, G_loss


def sample_image(E, G, data):
    latent, _ = E(data)
    fake = G(latent)
    return fake


def save_images(each_epoch, images):
    images = images.numpy()
    folder_name = 'generated_images/vaegan'
    os.makedirs(folder_name, exist_ok=True)
    plt.figure(figsize=(5, 5))
    for i in range(images.shape[0]):
        plt.subplot(5, 5, i + 1)
        plt.imshow((images[i, :, :, :] * 255).astype('uint8'))
        plt.axis('off')
    generated_image_file = folder_name + '/image_at_epoch_' + str(each_epoch + 1) + '.png'
    plt.savefig(generated_image_file)
    plt.close()
    generated_image_file = Image.open(generated_image_file)
    wandb.log({'image': wandb.Image(generated_image_file, caption='%s_epochs' % i)}, step=i)


def convert_tf_tensor_into_torch_tensor(tf_tensor):
    return torch.tensor(tf_tensor.numpy())


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


def get_model_and_optimizer(image_size, depth, kernel_size, latent_depth, learning_rate):
    E = encoder(image_size, depth, kernel_size, latent_depth)
    G = generator(image_size, depth, kernel_size, latent_depth)
    D = discriminator(image_size, depth, kernel_size)
    E_opt = keras.optimizers.Adam(lr=learning_rate)
    G_opt = keras.optimizers.Adam(lr=learning_rate)
    D_opt = keras.optimizers.Adam(lr=learning_rate)
    return E, G, D, E_opt, G_opt, D_opt


def log_index_with_inception_model(inception_model_score, e_loss, d_loss, g_loss, i):
    inception_model_score.lazy_forward(batch_size=32, device='cuda', fake_forward=True)
    inception_model_score.calculate_fake_image_statistics()
    metrics = inception_model_score.calculate_generative_score()
    inception_model_score.clear_fake()
    precision, recall, fid, inception_score_real, inception_score_fake, density, coverage = \
        metrics['precision'], metrics['recall'], metrics['fid'], metrics['real_is'], metrics['fake_is'], metrics[
            'density'], metrics['coverage']
    wandb.log({"r_loss": e_loss,
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


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    image_size = 16
    depth = 16
    latent_depth = 64
    kernel_size = 5
    learning_rate = 1e-4
    batch_size = 32
    epochs = 100
    save_image_interval = loss_calculation_interval = 5
    train_loader, _ = data_helper.get_ffhq_thumbnails_tensorflow(batch_size, image_size)

    wandb.login()
    wandb.init(project="AAE",
               config={
                   "batch_size": batch_size,
                   "epochs": epochs,
                   "img_size": image_size,
                   "save_image_interval": save_image_interval,
                   "loss_calculation_interval": loss_calculation_interval,
                   "latent_dim": latent_depth,
               })
    inception_model_score = load_inception_model(train_loader)

    E, G, D, E_opt, G_opt, D_opt = get_model_and_optimizer(image_size, depth, kernel_size, latent_depth, learning_rate)

    for i in range(0, epochs):
        for each_batch in train_loader:
            e_loss, d_loss, g_loss = train_step_vaegan(each_batch, batch_size, latent_depth, E, G, D, E_opt, G_opt, D_opt)
            if i % loss_calculation_interval == 0:
                sampled_images = sample_image(E, G, each_batch)
                sampled_images_torch = convert_tf_tensor_into_torch_tensor(sampled_images)
                inception_model_score.put_fake(sampled_images_torch)

        if i % save_image_interval == 0:
            save_images(i, sampled_images[0:25])

        if i % loss_calculation_interval == 0:
            log_index_with_inception_model(inception_model_score, e_loss, d_loss, g_loss, i)
        wandb.finish()


if __name__ == "__main__":
    main()