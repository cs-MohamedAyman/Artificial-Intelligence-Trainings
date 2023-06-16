# Wasserstein GAN with Gradient Penalty (WGAN-GP)

### Goals
In this notebook, you're going to build a Wasserstein GAN with Gradient Penalty (WGAN-GP) that solves some of the stability issues with the GANs that you have been using up until this point. Specifically, you'll use a special kind of loss function known as the W-loss, where W stands for Wasserstein, and gradient penalties to prevent mode collapse.

*Fun Fact: Wasserstein is named after a mathematician at Penn State, Leonid Vaseršteĭn. You'll see it abbreviated to W (e.g. WGAN, W-loss, W-distance).*

### Learning Objectives
1.   Get hands-on experience building a more stable GAN: Wasserstein GAN with Gradient Penalty (WGAN-GP).
2.   Train the more advanced WGAN-GP model.



## Generator and Critic

You will begin by importing some useful packages, defining visualization functions, building the generator, and building the critic. Since the changes for WGAN-GP are done to the loss function during training, you can simply reuse your previous GAN code for the generator and critic class. Remember that in WGAN-GP, you no longer use a discriminator that classifies fake and real as 0 and 1 but rather a critic that scores images with real numbers.

#### Packages and Visualizations


```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()

def make_grad_hook():
    '''
    Function to keep track of gradients for visualization purposes,
    which fills the grads list when using model.apply(grad_hook).
    '''
    grads = []
    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)
    return grads, grad_hook
```

#### Generator and Noise


```python
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        z_dim: the dimension of the noise vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(z_dim, hidden_dim * 4),
            self.make_gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.make_gen_block(hidden_dim * 2, hidden_dim),
            self.make_gen_block(hidden_dim, im_chan, kernel_size=4, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a generator block of DCGAN;
        a transposed convolution, a batchnorm (except in the final layer), and an activation.
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.Tanh(),
            )

    def forward(self, noise):
        '''
        Function for completing a forward pass of the generator: Given a noise tensor,
        returns generated images.
        Parameters:
            noise: a noise tensor with dimensions (n_samples, z_dim)
        '''
        x = noise.view(len(noise), self.z_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, z_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
      n_samples: the number of samples to generate, a scalar
      z_dim: the dimension of the noise vector, a scalar
      device: the device type
    '''
    return torch.randn(n_samples, z_dim, device=device)
```

#### Critic


```python
class Critic(nn.Module):
    '''
    Critic Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)
```

## Training Initializations
Now you can start putting it all together.
As usual, you will start by setting the parameters:
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   z_dim: the dimension of the noise vector
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   beta_1, beta_2: the momentum terms
  *   c_lambda: weight of the gradient penalty
  *   crit_repeats: number of times to update the critic per generator update - there are more details about this in the *Putting It All Together* section
  *   device: the device type

You will also load and transform the MNIST dataset to tensors.





```python
n_epochs = 100
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
device = 'cuda'

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=False, transform=transform),
    batch_size=batch_size,
    shuffle=True)
```

Then, you can initialize your generator, critic, and optimizers.


```python
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit = Critic().to(device)
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
crit = crit.apply(weights_init)

```

## Gradient Penalty
Calculating the gradient penalty can be broken into two functions: (1) compute the gradient with respect to the images and (2) compute the gradient penalty given the gradient.

You can start by getting the gradient. The gradient is computed by first creating a mixed image. This is done by weighing the fake and real image using epsilon and then adding them together. Once you have the intermediate image, you can get the critic's output on the image. Finally, you compute the gradient of the critic score's on the mixed images (output) with respect to the pixels of the mixed images (input). You will need to fill in the code to get the gradient wherever you see *None*. There is a test function in the next block for you to test your solution.


```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gradient
def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=mixed_images,
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient

```


```python
# UNIT TEST
# DO NOT MODIFY THIS
def test_get_gradient(image_shape):
    real = torch.randn(*image_shape, device=device) + 1
    fake = torch.randn(*image_shape, device=device) - 1
    epsilon_shape = [1 for _ in image_shape]
    epsilon_shape[0] = image_shape[0]
    epsilon = torch.rand(epsilon_shape, device=device).requires_grad_()
    gradient = get_gradient(crit, real, fake, epsilon)
    assert tuple(gradient.shape) == image_shape
    assert gradient.max() > 0
    assert gradient.min() < 0
    return gradient

gradient = test_get_gradient((256, 1, 28, 28))
print("Success!")
```

    Success!


The second function you need to complete is to compute the gradient penalty given the gradient. First, you calculate the magnitude of each image's gradient. The magnitude of a gradient is also called the norm. Then, you calculate the penalty by squaring the distance between each magnitude and the ideal norm of 1 and taking the mean of all the squared distances.

Again, you will need to fill in the code wherever you see *None*. There are hints below that you can view if you need help and there is a test function in the next block for you to test your solution.

<details>

<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">gradient_penalty</font></code></b>
</font>
</summary>


1.   Make sure you take the mean at the end.
2.   Note that the magnitude of each gradient has already been calculated for you.

</details>



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: gradient_penalty
def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    #### START CODE HERE ####
    penalty = torch.mean((gradient_norm - 1)**2)
    #### END CODE HERE ####
    return penalty
```


```python
# UNIT TEST
def test_gradient_penalty(image_shape):
    bad_gradient = torch.zeros(*image_shape)
    bad_gradient_penalty = gradient_penalty(bad_gradient)
    assert torch.isclose(bad_gradient_penalty, torch.tensor(1.))

    image_size = torch.prod(torch.Tensor(image_shape[1:]))
    good_gradient = torch.ones(*image_shape) / torch.sqrt(image_size)
    good_gradient_penalty = gradient_penalty(good_gradient)
    assert torch.isclose(good_gradient_penalty, torch.tensor(0.))

    random_gradient = test_get_gradient(image_shape)
    random_gradient_penalty = gradient_penalty(random_gradient)
    assert torch.abs(random_gradient_penalty - 1) < 0.1

test_gradient_penalty((256, 1, 28, 28))
print("Success!")
```

    Success!


## Losses
Next, you need to calculate the loss for the generator and the critic.

For the generator, the loss is calculated by maximizing the critic's prediction on the generator's fake images. The argument has the scores for all fake images in the batch, but you will use the mean of them.

There are optional hints below and a test function in the next block for you to test your solution.

<details><summary><font size="3" color="green"><b>Optional hints for <code><font size="4">get_gen_loss</font></code></b></font></summary>

1. This can be written in one line.
2. This is the negative of the mean of the critic's scores.

</details>


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_gen_loss
def get_gen_loss(crit_fake_pred):
    '''
    Return the loss of a generator given the critic's scores of the generator's fake images.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
    Returns:
        gen_loss: a scalar loss value for the current batch of the generator
    '''
    #### START CODE HERE ####
    gen_loss = -1. * torch.mean(crit_fake_pred)
    #### END CODE HERE ####
    return gen_loss
```


```python
# UNIT TEST
assert torch.isclose(
    get_gen_loss(torch.tensor(1.)), torch.tensor(-1.0)
)

assert torch.isclose(
    get_gen_loss(torch.rand(10000)), torch.tensor(-0.5), 0.05
)

print("Success!")
```

    Success!


For the critic, the loss is calculated by maximizing the distance between the critic's predictions on the real images and the predictions on the fake images while also adding a gradient penalty. The gradient penalty is weighed according to lambda. The arguments are the scores for all the images in the batch, and you will use the mean of them.

There are hints below if you get stuck and a test function in the next block for you to test your solution.

<details><summary><font size="3" color="green"><b>Optional hints for <code><font size="4">get_crit_loss</font></code></b></font></summary>

1. The higher the mean fake score, the higher the critic's loss is.
2. What does this suggest about the mean real score?
3. The higher the gradient penalty, the higher the critic's loss is, proportional to lambda.


</details>



```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_crit_loss
def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    #### START CODE HERE ####
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    #### END CODE HERE ####
    return crit_loss
```


```python
# UNIT TEST
assert torch.isclose(
    get_crit_loss(torch.tensor(1.), torch.tensor(2.), torch.tensor(3.), 0.1),
    torch.tensor(-0.7)
)
assert torch.isclose(
    get_crit_loss(torch.tensor(20.), torch.tensor(-20.), torch.tensor(2.), 10),
    torch.tensor(60.)
)

print("Success!")
```

    Success!


## Putting It All Together
Before you put everything together, there are a few things to note.
1.   Even on GPU, the **training will run more slowly** than previous labs because the gradient penalty requires you to compute the gradient of a gradient -- this means potentially a few minutes per epoch! For best results, run this for as long as you can while on GPU.
2.   One important difference from earlier versions is that you will **update the critic multiple times** every time you update the generator This helps prevent the generator from overpowering the critic. Sometimes, you might see the reverse, with the generator updated more times than the critic. This depends on architectural (e.g. the depth and width of the network) and algorithmic choices (e.g. which loss you're using).
3.   WGAN-GP isn't necessarily meant to improve overall performance of a GAN, but just **increases stability** and avoids mode collapse. In general, a WGAN will be able to train in a much more stable way than the vanilla DCGAN from last assignment, though it will generally run a bit slower. You should also be able to train your model for more epochs without it collapsing.


<!-- Once again, be warned that this runs very slowly on a CPU. One way to run this more quickly is to download the .ipynb and upload it to Google Drive, then open it with Google Colab and make the runtime type GPU and replace
`device = "cpu"`
with
`device = "cuda"`
and make sure that your `get_noise` function uses the right device.  -->

Here is a snapshot of what your WGAN-GP outputs should resemble:
![MNIST Digits Progression](MNIST_WGAN_Progression.png)


```python
import matplotlib.pyplot as plt

cur_step = 0
generator_losses = []
critic_losses = []
for epoch in range(n_epochs):
    # Dataloader returns the batches
    for real, _ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)

        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            ### Update critic ###
            crit_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device=device)
            fake = gen(fake_noise)
            crit_fake_pred = crit(fake.detach())
            crit_real_pred = crit(real)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(crit, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss.item() / crit_repeats
            # Update gradients
            crit_loss.backward(retain_graph=True)
            # Update optimizer
            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]

        ### Update generator ###
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device=device)
        fake_2 = gen(fake_noise_2)
        crit_fake_pred = crit(fake_2)

        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()

        # Update the weights
        gen_opt.step()

        # Keep track of the average generator loss
        generator_losses += [gen_loss.item()]

        ### Visualization code ###
        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
            show_tensor_images(fake)
            show_tensor_images(real)
            step_bins = 20
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Critic Loss"
            )
            plt.legend()
            plt.show()

        cur_step += 1

```


    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 1, step 500: Generator loss: -2.6130034635863266, critic loss: -93.57838140429553




![png](output_26_4.png)





![png](output_26_5.png)





![png](output_26_6.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 2, step 1000: Generator loss: -24.81751082983613, critic loss: -148.869506141281




![png](output_26_10.png)





![png](output_26_11.png)





![png](output_26_12.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 3, step 1500: Generator loss: -33.87178573513031, critic loss: -23.101583268356336




![png](output_26_16.png)





![png](output_26_17.png)





![png](output_26_18.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 4, step 2000: Generator loss: -38.46174158191681, critic loss: -21.28685361576083




![png](output_26_22.png)





![png](output_26_23.png)





![png](output_26_24.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 5, step 2500: Generator loss: -38.551915594697, critic loss: -9.283173213195798




![png](output_26_28.png)





![png](output_26_29.png)





![png](output_26_30.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 6, step 3000: Generator loss: -38.36437850952149, critic loss: -15.136300289821627




![png](output_26_34.png)





![png](output_26_35.png)





![png](output_26_36.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 7, step 3500: Generator loss: -40.52105819702148, critic loss: -10.409297717666627




![png](output_26_40.png)





![png](output_26_41.png)





![png](output_26_42.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 8, step 4000: Generator loss: -43.842125370025634, critic loss: -17.763680433177953




![png](output_26_46.png)





![png](output_26_47.png)





![png](output_26_48.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 9, step 4500: Generator loss: -37.25323214864731, critic loss: -10.629762846803658




![png](output_26_52.png)





![png](output_26_53.png)





![png](output_26_54.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 10, step 5000: Generator loss: -32.31490753364563, critic loss: -9.369103980588903




![png](output_26_58.png)





![png](output_26_59.png)





![png](output_26_60.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 11, step 5500: Generator loss: -30.510587425231932, critic loss: -2.346437727451326




![png](output_26_64.png)





![png](output_26_65.png)





![png](output_26_66.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 12, step 6000: Generator loss: -31.698964164733887, critic loss: -4.818620150613787




![png](output_26_70.png)





![png](output_26_71.png)





![png](output_26_72.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 13, step 6500: Generator loss: -33.77822774887085, critic loss: -12.204162207221987




![png](output_26_76.png)





![png](output_26_77.png)





![png](output_26_78.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 14, step 7000: Generator loss: -30.134064867019653, critic loss: -4.993988493013379




![png](output_26_82.png)





![png](output_26_83.png)





![png](output_26_84.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 15, step 7500: Generator loss: -30.54196647679806, critic loss: -8.114517228388786




![png](output_26_88.png)





![png](output_26_89.png)





![png](output_26_90.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 17, step 8000: Generator loss: -26.154088873267174, critic loss: -14.941630329108232




![png](output_26_96.png)





![png](output_26_97.png)





![png](output_26_98.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 18, step 8500: Generator loss: -21.27524675679207, critic loss: -10.377117546272276




![png](output_26_102.png)





![png](output_26_103.png)





![png](output_26_104.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 19, step 9000: Generator loss: -14.834061573207379, critic loss: -11.4978941814661




![png](output_26_108.png)





![png](output_26_109.png)





![png](output_26_110.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 20, step 9500: Generator loss: -5.011375313282013, critic loss: 0.6161161472082131




![png](output_26_114.png)





![png](output_26_115.png)





![png](output_26_116.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 21, step 10000: Generator loss: -5.242827995538711, critic loss: -2.8212464205265033




![png](output_26_120.png)





![png](output_26_121.png)





![png](output_26_122.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 22, step 10500: Generator loss: -1.9092173187695443, critic loss: -4.838149548983578




![png](output_26_126.png)





![png](output_26_127.png)





![png](output_26_128.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 23, step 11000: Generator loss: 0.4144488729909062, critic loss: -5.758865286350251




![png](output_26_132.png)





![png](output_26_133.png)





![png](output_26_134.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 24, step 11500: Generator loss: 1.7391010952964425, critic loss: -6.757206315636642




![png](output_26_138.png)





![png](output_26_139.png)





![png](output_26_140.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 25, step 12000: Generator loss: 1.7600376224964858, critic loss: -7.560103627181061




![png](output_26_144.png)





![png](output_26_145.png)





![png](output_26_146.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 26, step 12500: Generator loss: 1.9848615301176906, critic loss: -8.342054908180232




![png](output_26_150.png)





![png](output_26_151.png)





![png](output_26_152.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 27, step 13000: Generator loss: 1.8516939841508866, critic loss: -9.667190536117547




![png](output_26_156.png)





![png](output_26_157.png)





![png](output_26_158.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 28, step 13500: Generator loss: 3.230499232798815, critic loss: -10.920460025811186




![png](output_26_162.png)





![png](output_26_163.png)





![png](output_26_164.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 29, step 14000: Generator loss: 4.341595393188298, critic loss: -10.113627188491833




![png](output_26_168.png)





![png](output_26_169.png)





![png](output_26_170.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 30, step 14500: Generator loss: 4.006132373906672, critic loss: -9.241648130273816




![png](output_26_174.png)





![png](output_26_175.png)





![png](output_26_176.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 31, step 15000: Generator loss: 2.946048934161663, critic loss: -10.871459705495825




![png](output_26_180.png)





![png](output_26_181.png)





![png](output_26_182.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 33, step 15500: Generator loss: 3.218309499874711, critic loss: -11.676032731544984




![png](output_26_188.png)





![png](output_26_189.png)





![png](output_26_190.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 34, step 16000: Generator loss: 4.01928523170948, critic loss: -11.546917405152323




![png](output_26_194.png)





![png](output_26_195.png)





![png](output_26_196.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 35, step 16500: Generator loss: 3.2826939994692803, critic loss: -12.412927752327914




![png](output_26_200.png)





![png](output_26_201.png)





![png](output_26_202.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 36, step 17000: Generator loss: 4.181035980209709, critic loss: -12.798222673940652




![png](output_26_206.png)





![png](output_26_207.png)





![png](output_26_208.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 37, step 17500: Generator loss: 3.689686143159866, critic loss: -12.131436739623547




![png](output_26_212.png)





![png](output_26_213.png)





![png](output_26_214.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 38, step 18000: Generator loss: 3.7106358278095724, critic loss: -12.794200841724868




![png](output_26_218.png)





![png](output_26_219.png)





![png](output_26_220.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 39, step 18500: Generator loss: 2.4775173953175544, critic loss: -12.067752436482893




![png](output_26_224.png)





![png](output_26_225.png)





![png](output_26_226.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 40, step 19000: Generator loss: 2.8633497329354287, critic loss: -12.252317002373927




![png](output_26_230.png)





![png](output_26_231.png)





![png](output_26_232.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 41, step 19500: Generator loss: 2.4251849276423454, critic loss: -12.936019743216038




![png](output_26_236.png)





![png](output_26_237.png)





![png](output_26_238.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 42, step 20000: Generator loss: 5.274558512434363, critic loss: -12.483650515997411




![png](output_26_242.png)





![png](output_26_243.png)





![png](output_26_244.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 43, step 20500: Generator loss: 3.332445617720485, critic loss: -13.379663030433665




![png](output_26_248.png)





![png](output_26_249.png)





![png](output_26_250.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 44, step 21000: Generator loss: 5.68652634063363, critic loss: -12.318959063363065




![png](output_26_254.png)





![png](output_26_255.png)





![png](output_26_256.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 45, step 21500: Generator loss: 4.883447543740273, critic loss: -14.50521084616184




![png](output_26_260.png)





![png](output_26_261.png)





![png](output_26_262.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 46, step 22000: Generator loss: 7.0264445532262325, critic loss: -12.985385465860372




![png](output_26_266.png)





![png](output_26_267.png)





![png](output_26_268.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 47, step 22500: Generator loss: 5.2431119225919245, critic loss: -13.466373706233501




![png](output_26_272.png)





![png](output_26_273.png)





![png](output_26_274.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 49, step 23000: Generator loss: 5.380567958205939, critic loss: -9.114056124413008




![png](output_26_280.png)





![png](output_26_281.png)





![png](output_26_282.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 50, step 23500: Generator loss: 3.8788348366767167, critic loss: -12.102092664718636




![png](output_26_286.png)





![png](output_26_287.png)





![png](output_26_288.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 51, step 24000: Generator loss: 1.2081416932344438, critic loss: -13.93244610551594




![png](output_26_292.png)





![png](output_26_293.png)





![png](output_26_294.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 52, step 24500: Generator loss: 2.5779134930372236, critic loss: -13.226770010185238




![png](output_26_298.png)





![png](output_26_299.png)





![png](output_26_300.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 53, step 25000: Generator loss: 2.988876987569034, critic loss: -14.510088012635695




![png](output_26_304.png)





![png](output_26_305.png)





![png](output_26_306.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 54, step 25500: Generator loss: 2.4377902239859104, critic loss: -11.616658616924294




![png](output_26_310.png)





![png](output_26_311.png)





![png](output_26_312.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 55, step 26000: Generator loss: 3.0088468269109727, critic loss: -14.523087581980233




![png](output_26_316.png)





![png](output_26_317.png)





![png](output_26_318.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 56, step 26500: Generator loss: 3.9679329726099968, critic loss: -12.483734093630305




![png](output_26_322.png)





![png](output_26_323.png)





![png](output_26_324.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 57, step 27000: Generator loss: 6.855015438437462, critic loss: -15.476485918915278




![png](output_26_328.png)





![png](output_26_329.png)





![png](output_26_330.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 58, step 27500: Generator loss: 5.559839993536472, critic loss: -15.944693973350528




![png](output_26_334.png)





![png](output_26_335.png)





![png](output_26_336.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 59, step 28000: Generator loss: 6.251448056101799, critic loss: -12.265172090435046




![png](output_26_340.png)





![png](output_26_341.png)





![png](output_26_342.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 60, step 28500: Generator loss: 7.8407208215892314, critic loss: -11.22569173244239




![png](output_26_346.png)





![png](output_26_347.png)





![png](output_26_348.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 61, step 29000: Generator loss: 5.193619565486908, critic loss: -17.26747007243634




![png](output_26_352.png)





![png](output_26_353.png)





![png](output_26_354.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 62, step 29500: Generator loss: 5.095620941579342, critic loss: -13.171797966730589




![png](output_26_358.png)





![png](output_26_359.png)





![png](output_26_360.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 63, step 30000: Generator loss: 3.258391621917486, critic loss: -14.459536708199973




![png](output_26_364.png)





![png](output_26_365.png)





![png](output_26_366.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 65, step 30500: Generator loss: 6.293651960797608, critic loss: -12.009962124705313




![png](output_26_372.png)





![png](output_26_373.png)





![png](output_26_374.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 66, step 31000: Generator loss: 6.60381964468956, critic loss: -12.239786578977098




![png](output_26_378.png)





![png](output_26_379.png)





![png](output_26_380.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 67, step 31500: Generator loss: 5.9919731840491295, critic loss: -12.965568694937238




![png](output_26_384.png)





![png](output_26_385.png)





![png](output_26_386.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 68, step 32000: Generator loss: 10.275166946932673, critic loss: -11.530831430780903




![png](output_26_390.png)





![png](output_26_391.png)





![png](output_26_392.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 69, step 32500: Generator loss: 6.5229389488399026, critic loss: -16.17527731132508




![png](output_26_396.png)





![png](output_26_397.png)





![png](output_26_398.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 70, step 33000: Generator loss: 13.190424212068319, critic loss: -14.517456501209738




![png](output_26_402.png)





![png](output_26_403.png)





![png](output_26_404.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 71, step 33500: Generator loss: 18.933644334346056, critic loss: -14.667508805680278




![png](output_26_408.png)





![png](output_26_409.png)





![png](output_26_410.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 72, step 34000: Generator loss: 15.12660926643014, critic loss: -15.58177264270783




![png](output_26_414.png)





![png](output_26_415.png)





![png](output_26_416.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 73, step 34500: Generator loss: 15.656991827964783, critic loss: -12.909885948705675




![png](output_26_420.png)





![png](output_26_421.png)





![png](output_26_422.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 74, step 35000: Generator loss: 17.182914264559745, critic loss: -12.89598947244882




![png](output_26_426.png)





![png](output_26_427.png)





![png](output_26_428.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 75, step 35500: Generator loss: 17.75225321403146, critic loss: -13.474317762231829




![png](output_26_432.png)





![png](output_26_433.png)





![png](output_26_434.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 76, step 36000: Generator loss: 20.594631458103656, critic loss: -14.267572848010056




![png](output_26_438.png)





![png](output_26_439.png)





![png](output_26_440.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 77, step 36500: Generator loss: 22.828764033436777, critic loss: -12.55741561290623




![png](output_26_444.png)





![png](output_26_445.png)





![png](output_26_446.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 78, step 37000: Generator loss: 22.348887607753277, critic loss: -15.764453994810584




![png](output_26_450.png)





![png](output_26_451.png)





![png](output_26_452.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 79, step 37500: Generator loss: 17.741696625083684, critic loss: -18.498055571079274




![png](output_26_456.png)





![png](output_26_457.png)





![png](output_26_458.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 81, step 38000: Generator loss: 22.84428488099575, critic loss: -15.676653561496739




![png](output_26_464.png)





![png](output_26_465.png)





![png](output_26_466.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 82, step 38500: Generator loss: 25.32885788500309, critic loss: -14.056283354270455




![png](output_26_470.png)





![png](output_26_471.png)





![png](output_26_472.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 83, step 39000: Generator loss: 23.259546346127987, critic loss: -13.70831554048061




![png](output_26_476.png)





![png](output_26_477.png)





![png](output_26_478.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 84, step 39500: Generator loss: 23.781078409284355, critic loss: -14.663695296466335




![png](output_26_482.png)





![png](output_26_483.png)





![png](output_26_484.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 85, step 40000: Generator loss: 23.271473619520663, critic loss: -11.79705920108556




![png](output_26_488.png)





![png](output_26_489.png)





![png](output_26_490.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 86, step 40500: Generator loss: 21.809282642424108, critic loss: -13.305809428942208




![png](output_26_494.png)





![png](output_26_495.png)





![png](output_26_496.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 87, step 41000: Generator loss: 21.930424605727197, critic loss: -13.55650947036744




![png](output_26_500.png)





![png](output_26_501.png)





![png](output_26_502.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 88, step 41500: Generator loss: 23.244686062216758, critic loss: -18.6932151334405




![png](output_26_506.png)





![png](output_26_507.png)





![png](output_26_508.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 89, step 42000: Generator loss: 27.527889772593976, critic loss: -15.09604394518137




![png](output_26_512.png)





![png](output_26_513.png)





![png](output_26_514.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 90, step 42500: Generator loss: 25.162855397850276, critic loss: -18.276383932447445




![png](output_26_518.png)





![png](output_26_519.png)





![png](output_26_520.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 91, step 43000: Generator loss: 29.147730276092886, critic loss: -14.994430229175093




![png](output_26_524.png)





![png](output_26_525.png)





![png](output_26_526.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 92, step 43500: Generator loss: 29.658255671739578, critic loss: -13.740705809640883




![png](output_26_530.png)





![png](output_26_531.png)





![png](output_26_532.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 93, step 44000: Generator loss: 27.232941491007804, critic loss: -14.354459015452871




![png](output_26_536.png)





![png](output_26_537.png)





![png](output_26_538.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 94, step 44500: Generator loss: 28.197334048628807, critic loss: -14.591090174204123




![png](output_26_542.png)





![png](output_26_543.png)





![png](output_26_544.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 95, step 45000: Generator loss: 26.89384837180376, critic loss: -15.47949554339648




![png](output_26_548.png)





![png](output_26_549.png)





![png](output_26_550.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 97, step 45500: Generator loss: 27.29001526758075, critic loss: -18.45944005686046




![png](output_26_556.png)





![png](output_26_557.png)





![png](output_26_558.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 98, step 46000: Generator loss: 33.41176569509506, critic loss: -15.320763889157782




![png](output_26_562.png)





![png](output_26_563.png)





![png](output_26_564.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 99, step 46500: Generator loss: 38.510583362340924, critic loss: -15.36393708914519




![png](output_26_568.png)





![png](output_26_569.png)





![png](output_26_570.png)







```python

```
