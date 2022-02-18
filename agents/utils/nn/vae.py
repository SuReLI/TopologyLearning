import numpy as np
import torch
from torch import optim, sigmoid, relu
from torch.nn import Module, Sigmoid, Linear
from torch.optim import Optimizer


def sample_normal(means, log_stds):
    """
    Sample a value from a batch of means and log(std)
    :param means: mean from the encoder's latent space
    :param log_stds: log variance from the encoder's latent space
    """
    stds = torch.exp(0.5 * log_stds)  # standard deviation
    eps = torch.randn_like(stds)  # `randn_like` as we need the same size
    sample = means + (eps * stds)  # sampling as if coming from the input space
    return sample


class VAE(Module):

    def __init__(self, device, input_size,
                 encoder_hidden_layer_1_size,
                 nb_latent_variables,
                 decoder_hidden_layer_1_size,
                 learning_rate=0.0001,
                 optimizer_class=optim.Adam,
                 criterion=torch.nn.BCELoss(reduction='sum'),
                 beta=1,
                 prior_mean=0,
                 prior_std=1):
        """
        A class that define a variational auto-encoder: https://arxiv.org/abs/1312.6114

         - encoder_hidden_layer_1_size:
         - nb_latent_variables:
         - decoder_hidden_layer_1_size:
         - learning_rate:
         - optimizer_class:
         - criterion:
         - beta: ratio between BCE and DKL loss in VAE global loss
         loss = -beta * KLD + BCE
        """

        super().__init__()
        assert issubclass(optimizer_class, Optimizer)
        self.learning_rate = learning_rate
        self.nb_latent_variables = nb_latent_variables
        self.input_size = input_size
        self.criterion = criterion
        self.device = device
        self.to(self.device)
        self.encoder_hidden_layer_1_size = encoder_hidden_layer_1_size
        self.decoder_hidden_layer_1_size = decoder_hidden_layer_1_size
        self.prior_mean = prior_mean
        self.prior_std = prior_std

        # encoder
        self.encoder_layer_1 = torch.nn.Linear(in_features=self.input_size,
                                               out_features=self.encoder_hidden_layer_1_size)
        self.encoder_layer_2 = torch.nn.Linear(in_features=self.encoder_hidden_layer_1_size,
                                               out_features=self.nb_latent_variables * 2)

        # decoder
        self.decoder_layer_1 = torch.nn.Linear(in_features=self.nb_latent_variables,
                                               out_features=self.decoder_hidden_layer_1_size)
        self.decoder_layer_1_activation = torch.nn.ReLU()
        self.decoder_layer_2 = torch.nn.Linear(in_features=self.decoder_hidden_layer_1_size,
                                               out_features=self.input_size)
        self.decoder_layer_2_activation = torch.nn.Sigmoid()
        self.optimizer = optimizer_class(params=self.parameters(), lr=learning_rate)
        self.bce_loss_memory = []
        self.kld_loss_memory = []
        self.total_loss_memory = []
        self.beta = beta

        # Initialize the standard gaussian as our prior, we will fit our latent space to it
        prior_means = torch.zeros((self.nb_latent_variables,))
        prior_stds = torch.ones((self.nb_latent_variables,))
        self.prior = torch.distributions.Normal(prior_means, prior_stds)

        # Initialise means and stds for our latent space generator, that can be used to generate samples that are
        # close to the average latent space
        self.generator_means = prior_means
        self.generator_stds = prior_stds
        self.latent_generator_lr = 0.001

    def forward(self, inputs_data, learn=True):
        """
        Encode and decode inputs_data.
        If learn = True, we use encoding and decoding result of given inputs data to train the VAE.
        """

        if isinstance(inputs_data, list):
            inputs_data = torch.Tensor(inputs_data).to(self.device, dtype=torch.float32)
        if isinstance(inputs_data, np.ndarray):
            inputs_data = torch.from_numpy(inputs_data).to(self.device, dtype=torch.float32)

        # Encode
        latent_variables, means, stds = self.get_features(inputs_data, learn)
        average_mean = torch.mean(means, axis=0)
        average_std = torch.mean(stds, axis=0)

        # Decode
        reconstruction = self.get_reconstruction(latent_variables, learn)

        # Learn
        if learn:
            self.optimizer.zero_grad()
            bce_loss = self.criterion(reconstruction, inputs_data)
            kld_loss = - self.beta * torch.sum(1 + torch.log(stds) - means.pow(2) - stds)
            vae_loss = bce_loss + kld_loss

            self.total_loss_memory.append(vae_loss.item())
            self.bce_loss_memory.append(bce_loss.item())
            self.kld_loss_memory.append(kld_loss.item())
            vae_loss.backward()
            self.optimizer.step()

            # Train our generator attributes
            self.generator_means += self.latent_generator_lr * (average_mean - self.generator_means)
            self.generator_stds += self.latent_generator_lr * (average_std - self.generator_stds)

        return latent_variables, reconstruction

    def generate_latent(self) -> torch.Tensor:
        """
        Use our generator means and stds to generate samples that are expected to be close to our average latent space
        """
        # distribution = torch.distributions.Normal(self.generator_means, self.generator_stds)
        means = torch.zeros((self.nb_latent_variables,))
        stds = torch.full((self.nb_latent_variables,), 1.2)
        distribution = torch.distributions.Normal(means, stds)
        return distribution.sample()

    def get_features(self, inputs_data, learn=False):
        """
        Return latent variables vector associate with the given input.
        return : features, means, stds
        """
        if isinstance(inputs_data, list):
            inputs_data = torch.Tensor(inputs_data).to(self.device, dtype=torch.float32)
        if isinstance(inputs_data, np.ndarray):
            inputs_data = torch.from_numpy(inputs_data).to(self.device, dtype=torch.float32)

        if len(inputs_data.shape) < 2:
            # Then inputs_data isn't a batch but a single data
            learn = False  # we should learn on batch only for more stabilisation
            batch = False
        else:
            batch = True

        with torch.set_grad_enabled(learn):
            latent_output = self.encoder_layer_1(inputs_data)
            latent_output = self.encoder_layer_2(latent_output)

            if batch:
                means = latent_output[:, :self.nb_latent_variables]
                stds = 1e-6 + torch.nn.Softplus()(latent_output[:, self.nb_latent_variables:])
                feature = means + stds * self.prior.sample((inputs_data.shape[0],))
            else:
                means = latent_output[:self.nb_latent_variables]
                stds = 1e-6 + torch.nn.Softplus()(latent_output[self.nb_latent_variables:])
                feature = means + stds * self.prior.sample()
        return feature, means, stds

    def get_reconstruction(self, features, learn=False):

        if isinstance(features, list):
            features = torch.Tensor(features).to(self.device, dtype=torch.float32)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).to(self.device, dtype=torch.float32)

        with torch.set_grad_enabled(learn):
            reconstruction = self.decoder_layer_1(features)  # Layer 1
            reconstruction = self.decoder_layer_1_activation(reconstruction)  # Activation
            reconstruction = self.decoder_layer_2(reconstruction)  # Layer 2
            reconstruction = self.decoder_layer_2_activation(reconstruction)  # Activation
        return reconstruction
