import warnings

import numpy as np
import pandas as pd
import torch
from packaging import version
from torch import optim
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer_dp import DataTransformerDP
from ctgan.synthesizers.base import BaseSynthesizer

###### Privacy ###### 
import opacus
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules



PRIVACY_DISCRIMINATOR = True
PRIVACY_QUANTUM   = True
GRADIENT_PENALTY = False


ALPHAS            = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] + list(range(5, 64)) + [128, 256, 512])
NOISE_MULTIPLIER  = 1.1
MAX_GRAD_NORM     = 1.0
TARGET_DELTA      = -1
BUDGET            = 3

#Discrete
NOISY_OCCURENCES_D  = True
NOISE_D_STD         = 2
NOISE_D_MEAN        = 10
L_THRESHOLD_D       = 10

#Continuous
NOISY_OCCURENCES_C  = True
NOISE_C_STD         = 2
NOISE_C_MEAN        = 10
L_THRESHOLD_C       = 7000
COMPONENTS_C        = 3
SYNTHETIC_STD       = 9
###### Privacy ###### 

'''
# custom for calcuate grad_sample for multiple loss.backward()
def _custom_create_or_extend_grad_sample(param: torch.Tensor, grad_sample: torch.Tensor, batch_dim: int) -> None:
    """
    Create a 'grad_sample' attribute in the given parameter, or accumulate it
    if the 'grad_sample' attribute already exists.
    This custom code will not work when using optimizer.virtual_step()
    """
    if hasattr(param, "grad_sample"):
        print ("adding gradient")
        param.grad_sample = param.grad_sample + grad_sample
        # param.grad_sample = torch.cat((param.grad_sample, grad_sample), batch_dim)
    else:
        print ("new gradient")
        param.grad_sample = grad_sample

#opacus.supported_layers_grad_samplers._create_or_extend_grad_sample = (_custom_create_or_extend_grad_sample)
'''

# custom weights initialization called on netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class Discriminator(Module):

    def __init__(self, input_dim, discriminator_dim, pack=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pack
        self.pack = pack
        self.packdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)
        #self.requires_grad_(requires_grad=False)

    
    def calc_gradient_penalty(self, real_data, fake_data, device, pac=10, lambda_=10):
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))
        

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        
        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        
        gradients = gradients.view(-1, pac * real_data.size(1))
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty =  lambda_ * ((gradients_norm - 1) ** 2).mean()
        
        return gradient_penalty
        
       
    
    def forward(self, input):
        assert input.size()[0] % self.pack == 0
        return self.seq(input.view(-1, self.packdim))


class Residual(Module):

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input):
        out = self.fc(input)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input], dim=1)


class Generator(Module):

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input):
        data = self.seq(input)
        return data



'''   
def entangle_n_collapse_superpositions(data, mapping, condvec): #Warning: it needs to entangle only the condvec feature/values
    for feature in data.head():
        #print(mapping[feature])
        for index, value in data[feature].iteritems():
            tmp_value = int(round(value)) ###Todo: this will work only with categorical datasets that represent the values as ints
            
            #print("superpos: ", mapping[feature][tmp_value])                
            values, freqs = zip(*mapping[feature][tmp_value])
            val = np.random.choice(values, p=freqs) #sample value from the superposition
            print("before", data[feature][index])
            data.loc[feature,index] = val
            print("after", data[feature][index])
    return data
'''

class dpCTGANSynthesizer(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, discriminator_lr=2e-4,
                 discriminator_decay=0, batch_size=500, discriminator_steps=1, log_frequency=True,
                 verbose=False, epochs=300):

        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = 1
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trained_epochs = 0

        ###### Privacy ######
        self.alphas = ALPHAS
        self.noise_multiplier = NOISE_MULTIPLIER
        self.max_grad_norm = MAX_GRAD_NORM
        self.target_delta = TARGET_DELTA
        self.budget = BUDGET
        ###### Privacy ###### 
    
    def _init_dp_engine(self, model, batch_size, sample_size):
        privacy_engine = PrivacyEngine(
            model,
            batch_size = batch_size,
            sample_size = sample_size,
            alphas=self.alphas,
            noise_multiplier=self.noise_multiplier,
            #clip_per_layer=True,
            max_grad_norm=self.max_grad_norm,
        )

        return privacy_engine

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits:
                […, num_features] unnormalized log probabilities
            tau:
                non-negative scalar temperature
            hard:
                if True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                a dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard,
                                                        eps=eps, dim=dim)
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    assert 0

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != "softmax":
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError('Invalid columns found: {}'.format(invalid_columns))

    def fit(self, train_data, discrete_columns=tuple(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )

        self._transformer = DataTransformerDP(
                    privacy_quantum=PRIVACY_QUANTUM, 
                    noisy_occurences_d=NOISY_OCCURENCES_D, 
                    noise_d_mean=NOISE_D_MEAN,
                    noise_d_std=NOISE_D_STD,
                    l_threshold_d=L_THRESHOLD_D,
                    noisy_occurences_c=NOISY_OCCURENCES_C, 
                    noise_c_mean=NOISE_C_MEAN,
                    noise_c_std=NOISE_C_STD,
                    l_threshold_c=L_THRESHOLD_C,
                    components_c=COMPONENTS_C,
                    synthetic_std=SYNTHETIC_STD
                    )
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)


        self._data_sampler = DataSampler(
            train_data,
            self._transformer.output_info_list,
            self._log_frequency)

        data_dim = self._transformer.output_dimensions

        self._generator = Generator(
            self._embedding_dim + self._data_sampler.dim_cond_vec(),
            self._generator_dim,
            data_dim
        ).to(self._device)

        self._discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim
        ).to(self._device)

        self._optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        self._optimizerD = optim.Adam(
            self._discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        ###### Privacy ###### 
        if PRIVACY_DISCRIMINATOR:
            if self.target_delta==-1:
                self.target_delta = 1/len(train_data)

            #self._discriminator = convert_batchnorm_modules(self._discriminator).to(self.device)
            #self._discriminator.apply(weights_init)
            self.engine = self._init_dp_engine(self._discriminator, self._batch_size, len(train_data))
            self.engine.attach(self._optimizerD)
            #opacus.autograd_grad_sample.disable_hooks()
        ###### Privacy ###### 

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in range(epochs):
            self.trained_epochs += 1
            for id_ in range(steps_per_epoch):

                for n in range(self._discriminator_steps):
                    
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size) #This is used to sample real data, yes privacy !!! It returns discrete_column_ids, "discrete"!!!
                    #cond, mask, discrete_column_id, category_id_in_col
                    #c1,    m1,  col,                opt = condvec
                    #condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real = self._data_sampler.sample_data(self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)
                        fakez = torch.cat([fakez, c1], dim=1)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        if PRIVACY_QUANTUM:
                            real = self._data_sampler.sample_data_quantum(self._batch_size, col[perm], opt[perm], self._transformer.value_maps_id)
                        else:    
                            real = self._data_sampler.sample_data(self._batch_size, col[perm], opt[perm])
                        c2 = c1[perm]

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)
                    
                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fake



                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)


                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))
                    
                    if GRADIENT_PENALTY:
                        pen = self._discriminator.calc_gradient_penalty(real_cat, fake_cat, self._device)

                    self._optimizerD.zero_grad()
                    if GRADIENT_PENALTY:
                        pen.backward(retain_graph=True)
                    loss_d.backward()                    
                    self._optimizerD.step()


                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size) # This is not used to sample real data, no privacy

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)
                

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = self._discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                loss_g = -torch.mean(y_fake) + cross_entropy

                self._optimizerG.zero_grad()
                loss_g.backward()
                self._optimizerG.step()
                
            if self._verbose:
                print(f"Epoch {i+1}, Loss G: {loss_g.detach().cpu(): .4f},"
                      f"Loss D: {loss_d.detach().cpu(): .4f}",
                      flush=True)

    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:            
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        self._device = device
        if hasattr(self, '_generator'):
            self._generator.to(self._device)
        if hasattr(self, '_discriminator'):
            self._discriminator.to(self._device)
