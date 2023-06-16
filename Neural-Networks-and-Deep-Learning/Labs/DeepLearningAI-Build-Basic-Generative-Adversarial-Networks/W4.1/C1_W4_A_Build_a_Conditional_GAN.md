# Build a Conditional GAN

### Goals
In this notebook, you're going to make a conditional GAN in order to generate hand-written images of digits, conditioned on the digit to be generated (the class vector). This will let you choose what digit you want to generate.

You'll then do some exploration of the generated images to visualize what the noise and class vectors mean.

### Learning Objectives
1.   Learn the technical difference between a conditional and unconditional GAN.
2.   Understand the distinction between the class and noise vector in a conditional GAN.



## Getting Started

For this assignment, you will be using the MNIST dataset again, but there's nothing stopping you from applying this generator code to produce images of animals conditioned on the species or pictures of faces conditioned on facial characteristics.

Note that this assignment requires no changes to the architectures of the generator or discriminator, only changes to the data passed to both. The generator will no longer take `z_dim` as an argument, but  `input_dim` instead, since you need to pass in both the noise and class vectors. In addition to good variable naming, this also means that you can use the generator and discriminator code you have previously written with different parameters.

You will begin by importing the necessary libraries and building the generator and discriminator.

#### Packages and Visualization


```python
import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) # Set for our testing purposes, please do not change!

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28), nrow=5, show=True):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    if show:
        plt.show()
```

#### Generator and Noise


```python
class Generator(nn.Module):
    '''
    Generator Class
    Values:
        input_dim: the dimension of the input vector, a scalar
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, input_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 4),
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
            noise: a noise tensor with dimensions (n_samples, input_dim)
        '''
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, input_dim, device='cpu'):
    '''
    Function for creating noise vectors: Given the dimensions (n_samples, input_dim)
    creates a tensor of that shape filled with random numbers from the normal distribution.
    Parameters:
        n_samples: the number of samples to generate, a scalar
        input_dim: the dimension of the input vector, a scalar
        device: the device type
    '''
    return torch.randn(n_samples, input_dim, device=device)
```

#### Discriminator


```python
class Discriminator(nn.Module):
    '''
    Discriminator Class
    Values:
      im_chan: the number of channels in the images, fitted for the dataset used, a scalar
            (MNIST is black-and-white, so 1 channel is your default)
      hidden_dim: the inner dimension, a scalar
    '''
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.make_disc_block(im_chan, hidden_dim),
            self.make_disc_block(hidden_dim, hidden_dim * 2),
            self.make_disc_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_disc_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a discriminator block of the DCGAN;
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
        Function for completing a forward pass of the discriminator: Given an image tensor,
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        disc_pred = self.disc(image)
        return disc_pred.view(len(disc_pred), -1)
```

## Class Input

In conditional GANs, the input vector for the generator will also need to include the class information. The class is represented using a one-hot encoded vector where its length is the number of classes and each index represents a class. The vector is all 0's and a 1 on the chosen class. Given the labels of multiple images (e.g. from a batch) and number of classes, please create one-hot vectors for each label. There is a class within the PyTorch functional library that can help you.

<details>

<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">get_one_hot_labels</font></code></b>
</font>
</summary>

1.   This code can be done in one line.
2.   The documentation for [F.one_hot](https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.one_hot) may be helpful.

</details>



```python
# UNQ_C1 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_one_hot_labels

import torch.nn.functional as F
def get_one_hot_labels(labels, n_classes):
    '''
    Function for creating one-hot vectors for the labels, returns a tensor of shape (?, num_classes).
    Parameters:
        labels: tensor of labels from the dataloader, size (?)
        n_classes: the total number of classes in the dataset, an integer scalar
    '''
    #### START CODE HERE ####
    return F.one_hot(labels,n_classes)
    #### END CODE HERE ####
```


```python
assert (
    get_one_hot_labels(
        labels=torch.Tensor([[0, 2, 1]]).long(),
        n_classes=3
    ).tolist() ==
    [[
      [1, 0, 0],
      [0, 0, 1],
      [0, 1, 0]
    ]]
)
# Check that the device of get_one_hot_labels matches the input device
if torch.cuda.is_available():
    assert str(get_one_hot_labels(torch.Tensor([[0]]).long().cuda(), 1).device).startswith("cuda")

print("Success!")
```

    Success!


Next, you need to be able to concatenate the one-hot class vector to the noise vector before giving it to the generator. You will also need to do this when adding the class channels to the discriminator.

To do this, you will need to write a function that combines two vectors. Remember that you need to ensure that the vectors are the same type: floats. Again, you can look to the PyTorch library for help.
<details>
<summary>
<font size="3" color="green">
<b>Optional hints for <code><font size="4">combine_vectors</font></code></b>
</font>
</summary>

1.   This code can also be written in one line.
2.   The documentation for [torch.cat](https://pytorch.org/docs/master/generated/torch.cat.html) may be helpful.
3.   Specifically, you might want to look at what the `dim` argument of `torch.cat` does.

</details>



```python
# UNQ_C2 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: combine_vectors
def combine_vectors(x, y):
    '''
    Function for combining two vectors with shapes (n_samples, ?) and (n_samples, ?).
    Parameters:
      x: (n_samples, ?) the first vector.
        In this assignment, this will be the noise vector of shape (n_samples, z_dim),
        but you shouldn't need to know the second dimension's size.
      y: (n_samples, ?) the second vector.
        Once again, in this assignment this will be the one-hot class vector
        with the shape (n_samples, n_classes), but you shouldn't assume this in your code.
    '''
    # Note: Make sure this function outputs a float no matter what inputs it receives
    #### START CODE HERE ####
    combined = torch.cat((x.float(),y.float()), 1)
    #### END CODE HERE ####
    return combined
```


```python
combined = combine_vectors(torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6], [7, 8]]))
if torch.cuda.is_available():
    # Check that it doesn't break with cuda
    cuda_check = combine_vectors(torch.tensor([[1, 2], [3, 4]]).cuda(), torch.tensor([[5, 6], [7, 8]]).cuda())
    assert str(cuda_check.device).startswith("cuda")
# Check exact order of elements
assert torch.all(combined == torch.tensor([[1, 2, 5, 6], [3, 4, 7, 8]]))
# Tests that items are of float type
assert (type(combined[0][0].item()) == float)
# Check shapes
combined = combine_vectors(torch.randn(1, 4, 5), torch.randn(1, 8, 5));
assert tuple(combined.shape) == (1, 12, 5)
assert tuple(combine_vectors(torch.randn(1, 10, 12).long(), torch.randn(1, 20, 12).long()).shape) == (1, 30, 12)
# Check that the float transformation doesn't happen after the inputs are concatenated
assert tuple(combine_vectors(torch.randn(1, 10, 12).long(), torch.randn(1, 20, 12)).shape) == (1, 30, 12)
print("Success!")
```

    Success!


## Training
Now you can start to put it all together!
First, you will define some new parameters:

*   mnist_shape: the number of pixels in each MNIST image, which has dimensions 28 x 28 and one channel (because it's black-and-white) so 1 x 28 x 28
*   n_classes: the number of classes in MNIST (10, since there are the digits from 0 to 9)


```python
mnist_shape = (1, 28, 28)
n_classes = 10
```

And you also include the same parameters from previous assignments:

  *   criterion: the loss function
  *   n_epochs: the number of times you iterate through the entire dataset when training
  *   z_dim: the dimension of the noise vector
  *   display_step: how often to display/visualize the images
  *   batch_size: the number of images per forward/backward pass
  *   lr: the learning rate
  *   device: the device type



```python
criterion = nn.BCEWithLogitsLoss()
n_epochs = 200
z_dim = 64
display_step = 500
batch_size = 128
lr = 0.0002
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

Then, you can initialize your generator, discriminator, and optimizers. To do this, you will need to update the input dimensions for both models. For the generator, you will need to calculate the size of the input vector; recall that for conditional GANs, the generator's input is the noise vector concatenated with the class vector. For the discriminator, you need to add a channel for every class.


```python
# UNQ_C3 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED FUNCTION: get_input_dimensions
def get_input_dimensions(z_dim, mnist_shape, n_classes):
    '''
    Function for getting the size of the conditional input dimensions
    from z_dim, the image shape, and number of classes.
    Parameters:
        z_dim: the dimension of the noise vector, a scalar
        mnist_shape: the shape of each MNIST image as (C, W, H), which is (1, 28, 28)
        n_classes: the total number of classes in the dataset, an integer scalar
                (10 for MNIST)
    Returns:
        generator_input_dim: the input dimensionality of the conditional generator,
                          which takes the noise and class vectors
        discriminator_im_chan: the number of input channels to the discriminator
                            (e.g. C x 28 x 28 for MNIST)
    '''
    #### START CODE HERE ####
    generator_input_dim = z_dim + n_classes
    discriminator_im_chan = mnist_shape[0] + n_classes
    #### END CODE HERE ####
    return generator_input_dim, discriminator_im_chan
```


```python
def test_input_dims():
    gen_dim, disc_dim = get_input_dimensions(23, (12, 23, 52), 9)
    assert gen_dim == 32
    assert disc_dim == 21
test_input_dims()
print("Success!")
```

    Success!



```python
generator_input_dim, discriminator_im_chan = get_input_dimensions(z_dim, mnist_shape, n_classes)

gen = Generator(input_dim=generator_input_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator(im_chan=discriminator_im_chan).to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)
```

Now to train, you would like both your generator and your discriminator to know what class of image should be generated. There are a few locations where you will need to implement code.

For example, if you're generating a picture of the number "1", you would need to:

1.   Tell that to the generator, so that it knows it should be generating a "1"
2.   Tell that to the discriminator, so that it knows it should be looking at a "1". If the discriminator is told it should be looking at a 1 but sees something that's clearly an 8, it can guess that it's probably fake

There are no explicit unit tests here -- if this block of code runs and you don't change any of the other variables, then you've done it correctly!


```python
# UNQ_C4 (UNIQUE CELL IDENTIFIER, DO NOT EDIT)
# GRADED CELL
cur_step = 0
generator_losses = []
discriminator_losses = []

#UNIT TEST NOTE: Initializations needed for grading
noise_and_labels = False
fake = False

fake_image_and_labels = False
real_image_and_labels = False
disc_fake_pred = False
disc_real_pred = False

for epoch in range(n_epochs):
    # Dataloader returns the batches and the labels
    for real, labels in tqdm(dataloader):
        cur_batch_size = len(real)
        # Flatten the batch of real images from the dataset
        real = real.to(device)

        one_hot_labels = get_one_hot_labels(labels.to(device), n_classes)
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = image_one_hot_labels.repeat(1, 1, mnist_shape[1], mnist_shape[2])

        ### Update discriminator ###
        # Zero out the discriminator gradients
        disc_opt.zero_grad()
        # Get noise corresponding to the current batch_size
        fake_noise = get_noise(cur_batch_size, z_dim, device=device)

        # Now you can get the images from the generator
        # Steps: 1) Combine the noise vectors and the one-hot labels for the generator
        #        2) Generate the conditioned fake images

        #### START CODE HERE ####
        noise_and_labels = combine_vectors(fake_noise, one_hot_labels)
        fake = gen(noise_and_labels)
        #### END CODE HERE ####

        # Make sure that enough images were generated
        assert len(fake) == len(real)
        # Check that correct tensors were combined
        assert tuple(noise_and_labels.shape) == (cur_batch_size, fake_noise.shape[1] + one_hot_labels.shape[1])
        # It comes from the correct generator
        assert tuple(fake.shape) == (len(real), 1, 28, 28)

        # Now you can get the predictions from the discriminator
        # Steps: 1) Create the input for the discriminator
        #           a) Combine the fake images with image_one_hot_labels,
        #              remember to detach the generator (.detach()) so you do not backpropagate through it
        #           b) Combine the real images with image_one_hot_labels
        #        2) Get the discriminator's prediction on the fakes as disc_fake_pred
        #        3) Get the discriminator's prediction on the reals as disc_real_pred

        #### START CODE HERE ####
        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        real_image_and_labels = combine_vectors(real, image_one_hot_labels)
        disc_fake_pred = disc(fake_image_and_labels.detach())
        disc_real_pred = disc(real_image_and_labels)
        #### END CODE HERE ####

        # Make sure shapes are correct
        assert tuple(fake_image_and_labels.shape) == (len(real), fake.detach().shape[1] + image_one_hot_labels.shape[1], 28 ,28)
        assert tuple(real_image_and_labels.shape) == (len(real), real.shape[1] + image_one_hot_labels.shape[1], 28 ,28)
        # Make sure that enough predictions were made
        assert len(disc_real_pred) == len(real)
        # Make sure that the inputs are different
        assert torch.any(fake_image_and_labels != real_image_and_labels)
        # Shapes must match
        assert tuple(fake_image_and_labels.shape) == tuple(real_image_and_labels.shape)
        assert tuple(disc_fake_pred.shape) == tuple(disc_real_pred.shape)


        disc_fake_loss = criterion(disc_fake_pred, torch.zeros_like(disc_fake_pred))
        disc_real_loss = criterion(disc_real_pred, torch.ones_like(disc_real_pred))
        disc_loss = (disc_fake_loss + disc_real_loss) / 2
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        # Keep track of the average discriminator loss
        discriminator_losses += [disc_loss.item()]

        ### Update generator ###
        # Zero out the generator gradients
        gen_opt.zero_grad()

        fake_image_and_labels = combine_vectors(fake, image_one_hot_labels)
        # This will error if you didn't concatenate your labels to your image correctly
        disc_fake_pred = disc(fake_image_and_labels)
        gen_loss = criterion(disc_fake_pred, torch.ones_like(disc_fake_pred))
        gen_loss.backward()
        gen_opt.step()

        # Keep track of the generator losses
        generator_losses += [gen_loss.item()]
        #

        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            disc_mean = sum(discriminator_losses[-display_step:]) / display_step
            print(f"Epoch {epoch}, step {cur_step}: Generator loss: {gen_mean}, discriminator loss: {disc_mean}")
            show_tensor_images(fake)
            show_tensor_images(real)
            step_bins = 20
            x_axis = sorted([i * step_bins for i in range(len(generator_losses) // step_bins)] * step_bins)
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins),
                torch.Tensor(discriminator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Discriminator Loss"
            )
            plt.legend()
            plt.show()
        elif cur_step == 0:
            print("Congratulations! If you've gotten here, it's working. Please let this train until you're happy with how the generated numbers look, and then go on to the exploration!")
        cur_step += 1
```


    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Congratulations! If you've gotten here, it's working. Please let this train until you're happy with how the generated numbers look, and then go on to the exploration!




    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 1, step 500: Generator loss: 2.2994460270404815, discriminator loss: 0.25802686666324737




![png](output_24_4.png)





![png](output_24_5.png)





![png](output_24_6.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 2, step 1000: Generator loss: 4.133580854415894, discriminator loss: 0.05157274711877108




![png](output_24_10.png)





![png](output_24_11.png)





![png](output_24_12.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 3, step 1500: Generator loss: 4.527823904037476, discriminator loss: 0.0704661600086838




![png](output_24_16.png)





![png](output_24_17.png)





![png](output_24_18.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 4, step 2000: Generator loss: 5.218474022865295, discriminator loss: 0.01968503071460873




![png](output_24_22.png)





![png](output_24_23.png)





![png](output_24_24.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 5, step 2500: Generator loss: 3.8541908452510834, discriminator loss: 0.09571333551965654




![png](output_24_28.png)





![png](output_24_29.png)





![png](output_24_30.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 6, step 3000: Generator loss: 2.775699615240097, discriminator loss: 0.2197765261977911




![png](output_24_34.png)





![png](output_24_35.png)





![png](output_24_36.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 7, step 3500: Generator loss: 2.216576353549957, discriminator loss: 0.30455979496240615




![png](output_24_40.png)





![png](output_24_41.png)





![png](output_24_42.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 8, step 4000: Generator loss: 2.1370259330272674, discriminator loss: 0.30433118137717247




![png](output_24_46.png)





![png](output_24_47.png)





![png](output_24_48.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 9, step 4500: Generator loss: 2.207469543218613, discriminator loss: 0.32984550965577364




![png](output_24_52.png)





![png](output_24_53.png)





![png](output_24_54.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 10, step 5000: Generator loss: 2.1172194998264313, discriminator loss: 0.31326664006710053




![png](output_24_58.png)





![png](output_24_59.png)





![png](output_24_60.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 11, step 5500: Generator loss: 1.950115122795105, discriminator loss: 0.3513113224208355




![png](output_24_64.png)





![png](output_24_65.png)





![png](output_24_66.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 12, step 6000: Generator loss: 1.83089737200737, discriminator loss: 0.375219158321619




![png](output_24_70.png)





![png](output_24_71.png)





![png](output_24_72.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 13, step 6500: Generator loss: 1.69481006026268, discriminator loss: 0.41080950063467025




![png](output_24_76.png)





![png](output_24_77.png)





![png](output_24_78.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 14, step 7000: Generator loss: 1.67418967628479, discriminator loss: 0.44732009023427965




![png](output_24_82.png)





![png](output_24_83.png)





![png](output_24_84.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 15, step 7500: Generator loss: 1.5759689384698867, discriminator loss: 0.47812069046497346




![png](output_24_88.png)





![png](output_24_89.png)





![png](output_24_90.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 17, step 8000: Generator loss: 1.4187868062257767, discriminator loss: 0.5012637298703194




![png](output_24_96.png)





![png](output_24_97.png)





![png](output_24_98.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 18, step 8500: Generator loss: 1.305615213394165, discriminator loss: 0.5090425471067429




![png](output_24_102.png)





![png](output_24_103.png)





![png](output_24_104.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 19, step 9000: Generator loss: 1.2828618836402894, discriminator loss: 0.5233917621970177




![png](output_24_108.png)





![png](output_24_109.png)





![png](output_24_110.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 20, step 9500: Generator loss: 1.2711520748138427, discriminator loss: 0.5383985496163368




![png](output_24_114.png)





![png](output_24_115.png)





![png](output_24_116.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 21, step 10000: Generator loss: 1.2465871752500535, discriminator loss: 0.5483337042331695




![png](output_24_120.png)





![png](output_24_121.png)





![png](output_24_122.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 22, step 10500: Generator loss: 1.207080193042755, discriminator loss: 0.5544485647082329




![png](output_24_126.png)





![png](output_24_127.png)





![png](output_24_128.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 23, step 11000: Generator loss: 1.1446410044431687, discriminator loss: 0.5506661142110825




![png](output_24_132.png)





![png](output_24_133.png)





![png](output_24_134.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 24, step 11500: Generator loss: 1.2000531482696533, discriminator loss: 0.5671555352807045




![png](output_24_138.png)





![png](output_24_139.png)





![png](output_24_140.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 25, step 12000: Generator loss: 1.0748716387748718, discriminator loss: 0.5708335009813309




![png](output_24_144.png)





![png](output_24_145.png)





![png](output_24_146.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 26, step 12500: Generator loss: 1.1174158352613448, discriminator loss: 0.5847431383728982




![png](output_24_150.png)





![png](output_24_151.png)





![png](output_24_152.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 27, step 13000: Generator loss: 1.112717655301094, discriminator loss: 0.5799836780428886




![png](output_24_156.png)





![png](output_24_157.png)





![png](output_24_158.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 28, step 13500: Generator loss: 1.0725123938322068, discriminator loss: 0.5744892561435699




![png](output_24_162.png)





![png](output_24_163.png)





![png](output_24_164.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 29, step 14000: Generator loss: 1.1407982506752015, discriminator loss: 0.5859443846344948




![png](output_24_168.png)





![png](output_24_169.png)





![png](output_24_170.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 30, step 14500: Generator loss: 1.1023805133104325, discriminator loss: 0.5946435178518296




![png](output_24_174.png)





![png](output_24_175.png)





![png](output_24_176.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 31, step 15000: Generator loss: 1.0495222618579865, discriminator loss: 0.6034062060713768




![png](output_24_180.png)





![png](output_24_181.png)





![png](output_24_182.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 33, step 15500: Generator loss: 1.0478539620637894, discriminator loss: 0.5844691667556763




![png](output_24_188.png)





![png](output_24_189.png)





![png](output_24_190.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 34, step 16000: Generator loss: 1.0449236941337585, discriminator loss: 0.5895821680426597




![png](output_24_194.png)





![png](output_24_195.png)





![png](output_24_196.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 35, step 16500: Generator loss: 1.0930972822904588, discriminator loss: 0.582294180214405




![png](output_24_200.png)





![png](output_24_201.png)





![png](output_24_202.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 36, step 17000: Generator loss: 1.037439875125885, discriminator loss: 0.5889619362354278




![png](output_24_206.png)





![png](output_24_207.png)





![png](output_24_208.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 37, step 17500: Generator loss: 1.0347317955493927, discriminator loss: 0.5812775977253913




![png](output_24_212.png)





![png](output_24_213.png)





![png](output_24_214.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 38, step 18000: Generator loss: 1.0417944066524505, discriminator loss: 0.5819809479117394




![png](output_24_218.png)





![png](output_24_219.png)





![png](output_24_220.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 39, step 18500: Generator loss: 1.0844580264091492, discriminator loss: 0.5811848262548447




![png](output_24_224.png)





![png](output_24_225.png)





![png](output_24_226.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 40, step 19000: Generator loss: 1.0687696969509124, discriminator loss: 0.5846740243434906




![png](output_24_230.png)





![png](output_24_231.png)





![png](output_24_232.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 41, step 19500: Generator loss: 1.0468221977949141, discriminator loss: 0.5912315812706948




![png](output_24_236.png)





![png](output_24_237.png)





![png](output_24_238.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 42, step 20000: Generator loss: 1.0052846026420594, discriminator loss: 0.5822294471859932




![png](output_24_242.png)





![png](output_24_243.png)





![png](output_24_244.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 43, step 20500: Generator loss: 1.047353527188301, discriminator loss: 0.5890760014653206




![png](output_24_248.png)





![png](output_24_249.png)





![png](output_24_250.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 44, step 21000: Generator loss: 1.0245046194791794, discriminator loss: 0.586601186811924




![png](output_24_254.png)





![png](output_24_255.png)





![png](output_24_256.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 45, step 21500: Generator loss: 1.0408730741739274, discriminator loss: 0.5865002237558364




![png](output_24_260.png)





![png](output_24_261.png)





![png](output_24_262.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 46, step 22000: Generator loss: 1.0136560165882111, discriminator loss: 0.5953187294602394




![png](output_24_266.png)





![png](output_24_267.png)





![png](output_24_268.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 47, step 22500: Generator loss: 1.0204750542640686, discriminator loss: 0.5868579177856446




![png](output_24_272.png)





![png](output_24_273.png)





![png](output_24_274.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 49, step 23000: Generator loss: 1.0054126621484756, discriminator loss: 0.5874018022418022




![png](output_24_280.png)





![png](output_24_281.png)





![png](output_24_282.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 50, step 23500: Generator loss: 1.030528330564499, discriminator loss: 0.5886289908289909




![png](output_24_286.png)





![png](output_24_287.png)





![png](output_24_288.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 51, step 24000: Generator loss: 1.0082680051326751, discriminator loss: 0.5886369405984878




![png](output_24_292.png)





![png](output_24_293.png)





![png](output_24_294.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 52, step 24500: Generator loss: 0.9958629623651505, discriminator loss: 0.5966381486058235




![png](output_24_298.png)





![png](output_24_299.png)





![png](output_24_300.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 53, step 25000: Generator loss: 1.0317959920167923, discriminator loss: 0.587431320130825




![png](output_24_304.png)





![png](output_24_305.png)





![png](output_24_306.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 54, step 25500: Generator loss: 1.005542025089264, discriminator loss: 0.5861775594353675




![png](output_24_310.png)





![png](output_24_311.png)





![png](output_24_312.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 55, step 26000: Generator loss: 1.0076510442495346, discriminator loss: 0.5930678802132606




![png](output_24_316.png)





![png](output_24_317.png)





![png](output_24_318.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 56, step 26500: Generator loss: 0.9942079210281372, discriminator loss: 0.5904385181069374




![png](output_24_322.png)





![png](output_24_323.png)





![png](output_24_324.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 57, step 27000: Generator loss: 0.9958783601522445, discriminator loss: 0.5932279360294342




![png](output_24_328.png)





![png](output_24_329.png)





![png](output_24_330.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 58, step 27500: Generator loss: 0.9974129213094711, discriminator loss: 0.5908913087844848




![png](output_24_334.png)





![png](output_24_335.png)





![png](output_24_336.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 59, step 28000: Generator loss: 0.9851499705314636, discriminator loss: 0.5908181931972504




![png](output_24_340.png)





![png](output_24_341.png)





![png](output_24_342.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 60, step 28500: Generator loss: 0.979336688041687, discriminator loss: 0.5973297680020332




![png](output_24_346.png)





![png](output_24_347.png)





![png](output_24_348.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 61, step 29000: Generator loss: 1.023082589149475, discriminator loss: 0.5964895519018173




![png](output_24_352.png)





![png](output_24_353.png)





![png](output_24_354.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 62, step 29500: Generator loss: 0.9958865077495574, discriminator loss: 0.5935217607021331




![png](output_24_358.png)





![png](output_24_359.png)





![png](output_24_360.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 63, step 30000: Generator loss: 0.9841636501550675, discriminator loss: 0.5933473367094994




![png](output_24_364.png)





![png](output_24_365.png)





![png](output_24_366.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 65, step 30500: Generator loss: 0.9764117330312729, discriminator loss: 0.5957786435484886




![png](output_24_372.png)





![png](output_24_373.png)





![png](output_24_374.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 66, step 31000: Generator loss: 1.0209563899040222, discriminator loss: 0.5927806071639061




![png](output_24_378.png)





![png](output_24_379.png)





![png](output_24_380.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 67, step 31500: Generator loss: 1.0094713985919952, discriminator loss: 0.6008330743908882




![png](output_24_384.png)





![png](output_24_385.png)





![png](output_24_386.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 68, step 32000: Generator loss: 1.0177247301340102, discriminator loss: 0.6040955215096474




![png](output_24_390.png)





![png](output_24_391.png)





![png](output_24_392.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 69, step 32500: Generator loss: 1.0077689690589904, discriminator loss: 0.5968445336818695




![png](output_24_396.png)





![png](output_24_397.png)





![png](output_24_398.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 70, step 33000: Generator loss: 0.9736541006565094, discriminator loss: 0.5955089228749275




![png](output_24_402.png)





![png](output_24_403.png)





![png](output_24_404.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 71, step 33500: Generator loss: 0.9935735427141189, discriminator loss: 0.5949350751042366




![png](output_24_408.png)





![png](output_24_409.png)





![png](output_24_410.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 72, step 34000: Generator loss: 0.9953819218873977, discriminator loss: 0.6004861964583397




![png](output_24_414.png)





![png](output_24_415.png)





![png](output_24_416.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 73, step 34500: Generator loss: 0.974625633239746, discriminator loss: 0.6046462606191635




![png](output_24_420.png)





![png](output_24_421.png)





![png](output_24_422.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 74, step 35000: Generator loss: 0.9890502947568893, discriminator loss: 0.5977474223971367




![png](output_24_426.png)





![png](output_24_427.png)





![png](output_24_428.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 75, step 35500: Generator loss: 0.9912460250854492, discriminator loss: 0.600559443116188




![png](output_24_432.png)





![png](output_24_433.png)





![png](output_24_434.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 76, step 36000: Generator loss: 0.9953379040956497, discriminator loss: 0.5983214326500893




![png](output_24_438.png)





![png](output_24_439.png)





![png](output_24_440.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 77, step 36500: Generator loss: 0.9721168154478073, discriminator loss: 0.6085075702667236




![png](output_24_444.png)





![png](output_24_445.png)





![png](output_24_446.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 78, step 37000: Generator loss: 0.9797779836654663, discriminator loss: 0.6021550856232644




![png](output_24_450.png)





![png](output_24_451.png)





![png](output_24_452.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 79, step 37500: Generator loss: 0.9722304557561874, discriminator loss: 0.6052372695207596




![png](output_24_456.png)





![png](output_24_457.png)





![png](output_24_458.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 81, step 38000: Generator loss: 0.9635982868671418, discriminator loss: 0.6063666025996208




![png](output_24_464.png)





![png](output_24_465.png)





![png](output_24_466.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 82, step 38500: Generator loss: 0.9790893284082413, discriminator loss: 0.6004205818772316




![png](output_24_470.png)





![png](output_24_471.png)





![png](output_24_472.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 83, step 39000: Generator loss: 0.9857608903646469, discriminator loss: 0.598531287908554




![png](output_24_476.png)





![png](output_24_477.png)





![png](output_24_478.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 84, step 39500: Generator loss: 0.9807024494409561, discriminator loss: 0.5998865463137627




![png](output_24_482.png)





![png](output_24_483.png)





![png](output_24_484.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 85, step 40000: Generator loss: 0.9751378606557846, discriminator loss: 0.6068361273407936




![png](output_24_488.png)





![png](output_24_489.png)





![png](output_24_490.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 86, step 40500: Generator loss: 0.9733406958580018, discriminator loss: 0.5987349880933761




![png](output_24_494.png)





![png](output_24_495.png)





![png](output_24_496.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 87, step 41000: Generator loss: 0.9807240604162216, discriminator loss: 0.6067619522213936




![png](output_24_500.png)





![png](output_24_501.png)





![png](output_24_502.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 88, step 41500: Generator loss: 0.9791790872812272, discriminator loss: 0.6043006680011749




![png](output_24_506.png)





![png](output_24_507.png)





![png](output_24_508.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 89, step 42000: Generator loss: 0.976837577342987, discriminator loss: 0.6059988568425179




![png](output_24_512.png)





![png](output_24_513.png)





![png](output_24_514.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 90, step 42500: Generator loss: 0.9823344362974167, discriminator loss: 0.6064379267692566




![png](output_24_518.png)





![png](output_24_519.png)





![png](output_24_520.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 91, step 43000: Generator loss: 0.9678240333795547, discriminator loss: 0.6021820861697197




![png](output_24_524.png)





![png](output_24_525.png)





![png](output_24_526.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 92, step 43500: Generator loss: 0.9621212687492371, discriminator loss: 0.6033081632256508




![png](output_24_530.png)





![png](output_24_531.png)





![png](output_24_532.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 93, step 44000: Generator loss: 0.988887352347374, discriminator loss: 0.6004443855881691




![png](output_24_536.png)





![png](output_24_537.png)





![png](output_24_538.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 94, step 44500: Generator loss: 0.9598115750551224, discriminator loss: 0.6023827365636826




![png](output_24_542.png)





![png](output_24_543.png)





![png](output_24_544.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 95, step 45000: Generator loss: 0.9674626207351684, discriminator loss: 0.6025593253970146




![png](output_24_548.png)





![png](output_24_549.png)





![png](output_24_550.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 97, step 45500: Generator loss: 0.9771217782497406, discriminator loss: 0.60115096783638




![png](output_24_556.png)





![png](output_24_557.png)





![png](output_24_558.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 98, step 46000: Generator loss: 1.0053023726940156, discriminator loss: 0.599763991177082




![png](output_24_562.png)





![png](output_24_563.png)





![png](output_24_564.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 99, step 46500: Generator loss: 0.9896198904514313, discriminator loss: 0.6026074609160423




![png](output_24_568.png)





![png](output_24_569.png)





![png](output_24_570.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 100, step 47000: Generator loss: 0.9828343307971954, discriminator loss: 0.6008687398433685




![png](output_24_574.png)





![png](output_24_575.png)





![png](output_24_576.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 101, step 47500: Generator loss: 0.9692636550664901, discriminator loss: 0.6042671158313752




![png](output_24_580.png)





![png](output_24_581.png)





![png](output_24_582.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 102, step 48000: Generator loss: 0.9830075682401657, discriminator loss: 0.6025417411327362




![png](output_24_586.png)





![png](output_24_587.png)





![png](output_24_588.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 103, step 48500: Generator loss: 0.9854782959222793, discriminator loss: 0.5995112216472626




![png](output_24_592.png)





![png](output_24_593.png)





![png](output_24_594.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 104, step 49000: Generator loss: 0.9588533289432526, discriminator loss: 0.6008244843482972




![png](output_24_598.png)





![png](output_24_599.png)





![png](output_24_600.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 105, step 49500: Generator loss: 1.0199856545925141, discriminator loss: 0.5981171219348907




![png](output_24_604.png)





![png](output_24_605.png)





![png](output_24_606.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 106, step 50000: Generator loss: 0.9824487428665161, discriminator loss: 0.6034878003597259




![png](output_24_610.png)





![png](output_24_611.png)





![png](output_24_612.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 107, step 50500: Generator loss: 0.9811592879295349, discriminator loss: 0.5948119945526124




![png](output_24_616.png)





![png](output_24_617.png)





![png](output_24_618.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 108, step 51000: Generator loss: 0.9880111052989959, discriminator loss: 0.6039895302057267




![png](output_24_622.png)





![png](output_24_623.png)





![png](output_24_624.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 109, step 51500: Generator loss: 0.9837779339551925, discriminator loss: 0.5921505220532417




![png](output_24_628.png)





![png](output_24_629.png)





![png](output_24_630.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 110, step 52000: Generator loss: 0.9822017234563828, discriminator loss: 0.5995462906956672




![png](output_24_634.png)





![png](output_24_635.png)





![png](output_24_636.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 111, step 52500: Generator loss: 0.9796540808677673, discriminator loss: 0.5976293622851372




![png](output_24_640.png)





![png](output_24_641.png)





![png](output_24_642.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 113, step 53000: Generator loss: 0.9906062974929809, discriminator loss: 0.5986420919895172




![png](output_24_648.png)





![png](output_24_649.png)





![png](output_24_650.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 114, step 53500: Generator loss: 0.9778676257133484, discriminator loss: 0.5983552414178849




![png](output_24_654.png)





![png](output_24_655.png)





![png](output_24_656.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 115, step 54000: Generator loss: 0.971722414135933, discriminator loss: 0.5977405114769936




![png](output_24_660.png)





![png](output_24_661.png)





![png](output_24_662.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 116, step 54500: Generator loss: 0.9890982238054276, discriminator loss: 0.598822805762291




![png](output_24_666.png)





![png](output_24_667.png)





![png](output_24_668.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 117, step 55000: Generator loss: 0.9802026765346528, discriminator loss: 0.5978608839511871




![png](output_24_672.png)





![png](output_24_673.png)





![png](output_24_674.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 118, step 55500: Generator loss: 0.9655655288696289, discriminator loss: 0.6067186052799225




![png](output_24_678.png)





![png](output_24_679.png)





![png](output_24_680.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 119, step 56000: Generator loss: 0.993438374042511, discriminator loss: 0.6027222691774369




![png](output_24_684.png)





![png](output_24_685.png)





![png](output_24_686.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 120, step 56500: Generator loss: 0.9607727353572846, discriminator loss: 0.6031061400175095




![png](output_24_690.png)





![png](output_24_691.png)





![png](output_24_692.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 121, step 57000: Generator loss: 0.9820762330293655, discriminator loss: 0.5963677777647972




![png](output_24_696.png)





![png](output_24_697.png)





![png](output_24_698.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 122, step 57500: Generator loss: 0.9982331368923187, discriminator loss: 0.6025833143591881




![png](output_24_702.png)





![png](output_24_703.png)





![png](output_24_704.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 123, step 58000: Generator loss: 0.9746223330497742, discriminator loss: 0.605237594127655




![png](output_24_708.png)





![png](output_24_709.png)





![png](output_24_710.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 124, step 58500: Generator loss: 0.9534994895458222, discriminator loss: 0.6024277060031891




![png](output_24_714.png)





![png](output_24_715.png)





![png](output_24_716.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 125, step 59000: Generator loss: 0.9843374140262604, discriminator loss: 0.6024842596054077




![png](output_24_720.png)





![png](output_24_721.png)





![png](output_24_722.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 126, step 59500: Generator loss: 0.968258891582489, discriminator loss: 0.6001843035817146




![png](output_24_726.png)





![png](output_24_727.png)





![png](output_24_728.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 127, step 60000: Generator loss: 0.9780463007688522, discriminator loss: 0.5974084703326226




![png](output_24_732.png)





![png](output_24_733.png)





![png](output_24_734.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 128, step 60500: Generator loss: 0.9780380479097366, discriminator loss: 0.6042231923937798




![png](output_24_738.png)





![png](output_24_739.png)





![png](output_24_740.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 130, step 61000: Generator loss: 0.976155357837677, discriminator loss: 0.5997502791285515




![png](output_24_746.png)





![png](output_24_747.png)





![png](output_24_748.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 131, step 61500: Generator loss: 0.9718122878074646, discriminator loss: 0.596609869658947




![png](output_24_752.png)





![png](output_24_753.png)





![png](output_24_754.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 132, step 62000: Generator loss: 0.9622417948246003, discriminator loss: 0.6042600727081299




![png](output_24_758.png)





![png](output_24_759.png)





![png](output_24_760.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 133, step 62500: Generator loss: 0.9778058724403381, discriminator loss: 0.605848188817501




![png](output_24_764.png)





![png](output_24_765.png)





![png](output_24_766.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 134, step 63000: Generator loss: 0.9621023033857345, discriminator loss: 0.6038125606775284




![png](output_24_770.png)





![png](output_24_771.png)





![png](output_24_772.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 135, step 63500: Generator loss: 0.975193552851677, discriminator loss: 0.6011783007979393




![png](output_24_776.png)





![png](output_24_777.png)





![png](output_24_778.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 136, step 64000: Generator loss: 0.9780105739831925, discriminator loss: 0.6042954063415528




![png](output_24_782.png)





![png](output_24_783.png)





![png](output_24_784.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 137, step 64500: Generator loss: 0.9797543345689773, discriminator loss: 0.6041819321513175




![png](output_24_788.png)





![png](output_24_789.png)





![png](output_24_790.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 138, step 65000: Generator loss: 0.9714867478609085, discriminator loss: 0.6045372453927994




![png](output_24_794.png)





![png](output_24_795.png)





![png](output_24_796.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 139, step 65500: Generator loss: 0.9686758607625962, discriminator loss: 0.6052218024134636




![png](output_24_800.png)





![png](output_24_801.png)





![png](output_24_802.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 140, step 66000: Generator loss: 0.9674166550636292, discriminator loss: 0.6047065742611885




![png](output_24_806.png)





![png](output_24_807.png)





![png](output_24_808.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 141, step 66500: Generator loss: 0.982537739276886, discriminator loss: 0.6088487620353699




![png](output_24_812.png)





![png](output_24_813.png)





![png](output_24_814.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 142, step 67000: Generator loss: 0.9715240691900253, discriminator loss: 0.607586745917797




![png](output_24_818.png)





![png](output_24_819.png)





![png](output_24_820.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 143, step 67500: Generator loss: 0.9790239825248718, discriminator loss: 0.6083207066059113




![png](output_24_824.png)





![png](output_24_825.png)





![png](output_24_826.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 144, step 68000: Generator loss: 0.9779148453474045, discriminator loss: 0.6007446509003639




![png](output_24_830.png)





![png](output_24_831.png)





![png](output_24_832.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 146, step 68500: Generator loss: 0.9903695073127746, discriminator loss: 0.6116653069853782




![png](output_24_838.png)





![png](output_24_839.png)





![png](output_24_840.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 147, step 69000: Generator loss: 0.9639525074958801, discriminator loss: 0.6074223895072937




![png](output_24_844.png)





![png](output_24_845.png)





![png](output_24_846.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 148, step 69500: Generator loss: 0.9652282826900482, discriminator loss: 0.6063721247911453




![png](output_24_850.png)





![png](output_24_851.png)





![png](output_24_852.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 149, step 70000: Generator loss: 0.973575701713562, discriminator loss: 0.6052504378557205




![png](output_24_856.png)





![png](output_24_857.png)





![png](output_24_858.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 150, step 70500: Generator loss: 0.9622790558338166, discriminator loss: 0.6088725597858429




![png](output_24_862.png)





![png](output_24_863.png)





![png](output_24_864.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 151, step 71000: Generator loss: 0.9665740678310394, discriminator loss: 0.6042987343072891




![png](output_24_868.png)





![png](output_24_869.png)





![png](output_24_870.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 152, step 71500: Generator loss: 0.9672456266880035, discriminator loss: 0.6124499165415764




![png](output_24_874.png)





![png](output_24_875.png)





![png](output_24_876.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 153, step 72000: Generator loss: 0.9645790061950683, discriminator loss: 0.6066190376281738




![png](output_24_880.png)





![png](output_24_881.png)





![png](output_24_882.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 154, step 72500: Generator loss: 0.9760936006307602, discriminator loss: 0.6071418480873108




![png](output_24_886.png)





![png](output_24_887.png)





![png](output_24_888.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 155, step 73000: Generator loss: 0.966168284535408, discriminator loss: 0.60827181828022




![png](output_24_892.png)





![png](output_24_893.png)





![png](output_24_894.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 156, step 73500: Generator loss: 0.9696594289541245, discriminator loss: 0.6040887070894241




![png](output_24_898.png)





![png](output_24_899.png)





![png](output_24_900.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 157, step 74000: Generator loss: 0.983679612994194, discriminator loss: 0.6096131769418717




![png](output_24_904.png)





![png](output_24_905.png)





![png](output_24_906.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 158, step 74500: Generator loss: 0.9651159902811051, discriminator loss: 0.6118641901016235




![png](output_24_910.png)





![png](output_24_911.png)





![png](output_24_912.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 159, step 75000: Generator loss: 0.9699658069610596, discriminator loss: 0.6197167947292328




![png](output_24_916.png)





![png](output_24_917.png)





![png](output_24_918.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 160, step 75500: Generator loss: 0.9481092828512192, discriminator loss: 0.6120364649891853




![png](output_24_922.png)





![png](output_24_923.png)





![png](output_24_924.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 162, step 76000: Generator loss: 0.9592023229598999, discriminator loss: 0.6111783970594407




![png](output_24_930.png)





![png](output_24_931.png)





![png](output_24_932.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 163, step 76500: Generator loss: 0.954637156367302, discriminator loss: 0.6138023303747178




![png](output_24_936.png)





![png](output_24_937.png)





![png](output_24_938.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 164, step 77000: Generator loss: 0.9571660722494125, discriminator loss: 0.6136248801350593




![png](output_24_942.png)





![png](output_24_943.png)





![png](output_24_944.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 165, step 77500: Generator loss: 0.9583341913223267, discriminator loss: 0.6200984737873078




![png](output_24_948.png)





![png](output_24_949.png)





![png](output_24_950.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 166, step 78000: Generator loss: 0.9685019857883453, discriminator loss: 0.6169410107731819




![png](output_24_954.png)





![png](output_24_955.png)





![png](output_24_956.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 167, step 78500: Generator loss: 0.9513339095115662, discriminator loss: 0.6117570389509202




![png](output_24_960.png)





![png](output_24_961.png)





![png](output_24_962.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 168, step 79000: Generator loss: 0.9620497760772705, discriminator loss: 0.6154631490111351




![png](output_24_966.png)





![png](output_24_967.png)





![png](output_24_968.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 169, step 79500: Generator loss: 0.9453037875890732, discriminator loss: 0.6178486551046372




![png](output_24_972.png)





![png](output_24_973.png)





![png](output_24_974.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 170, step 80000: Generator loss: 0.9407494051456451, discriminator loss: 0.6188076483607292




![png](output_24_978.png)





![png](output_24_979.png)





![png](output_24_980.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 171, step 80500: Generator loss: 0.9397314624786377, discriminator loss: 0.6189351926445961




![png](output_24_984.png)





![png](output_24_985.png)





![png](output_24_986.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 172, step 81000: Generator loss: 0.9442030874490738, discriminator loss: 0.6152944527864456




![png](output_24_990.png)





![png](output_24_991.png)





![png](output_24_992.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 173, step 81500: Generator loss: 0.9587772606611252, discriminator loss: 0.6180650225877762




![png](output_24_996.png)





![png](output_24_997.png)





![png](output_24_998.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 174, step 82000: Generator loss: 0.9607018042802811, discriminator loss: 0.6135607680678368




![png](output_24_1002.png)





![png](output_24_1003.png)





![png](output_24_1004.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 175, step 82500: Generator loss: 0.9314943372011185, discriminator loss: 0.618624664068222




![png](output_24_1008.png)





![png](output_24_1009.png)





![png](output_24_1010.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 176, step 83000: Generator loss: 0.931631964802742, discriminator loss: 0.6191835305690765




![png](output_24_1014.png)





![png](output_24_1015.png)





![png](output_24_1016.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 178, step 83500: Generator loss: 0.9484588551521301, discriminator loss: 0.6189018679857254




![png](output_24_1022.png)





![png](output_24_1023.png)





![png](output_24_1024.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 179, step 84000: Generator loss: 0.9557680226564408, discriminator loss: 0.6210512470602989




![png](output_24_1028.png)





![png](output_24_1029.png)





![png](output_24_1030.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 180, step 84500: Generator loss: 0.9387476215362549, discriminator loss: 0.6224695174694062




![png](output_24_1034.png)





![png](output_24_1035.png)





![png](output_24_1036.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 181, step 85000: Generator loss: 0.9352725939750671, discriminator loss: 0.6201993046402932




![png](output_24_1040.png)





![png](output_24_1041.png)





![png](output_24_1042.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 182, step 85500: Generator loss: 0.9456083300113678, discriminator loss: 0.6191201339960098




![png](output_24_1046.png)





![png](output_24_1047.png)





![png](output_24_1048.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 183, step 86000: Generator loss: 0.9358500038385391, discriminator loss: 0.6235596992373467




![png](output_24_1052.png)





![png](output_24_1053.png)





![png](output_24_1054.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 184, step 86500: Generator loss: 0.9476438941955566, discriminator loss: 0.6223020035624504




![png](output_24_1058.png)





![png](output_24_1059.png)





![png](output_24_1060.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 185, step 87000: Generator loss: 0.9355169577598572, discriminator loss: 0.6217075709104538




![png](output_24_1064.png)





![png](output_24_1065.png)





![png](output_24_1066.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 186, step 87500: Generator loss: 0.9437664337158204, discriminator loss: 0.6229386448860168




![png](output_24_1070.png)





![png](output_24_1071.png)





![png](output_24_1072.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 187, step 88000: Generator loss: 0.9427596902847291, discriminator loss: 0.6250917895436287




![png](output_24_1076.png)





![png](output_24_1077.png)





![png](output_24_1078.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 188, step 88500: Generator loss: 0.9312309228181839, discriminator loss: 0.6242808635234833




![png](output_24_1082.png)





![png](output_24_1083.png)





![png](output_24_1084.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 189, step 89000: Generator loss: 0.9327132829427719, discriminator loss: 0.6250814302563668




![png](output_24_1088.png)





![png](output_24_1089.png)





![png](output_24_1090.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 190, step 89500: Generator loss: 0.9384497290849686, discriminator loss: 0.6250820506811142




![png](output_24_1094.png)





![png](output_24_1095.png)





![png](output_24_1096.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 191, step 90000: Generator loss: 0.9272738991975784, discriminator loss: 0.6254813120365142




![png](output_24_1100.png)





![png](output_24_1101.png)





![png](output_24_1102.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 192, step 90500: Generator loss: 0.9394500435590744, discriminator loss: 0.625072253704071




![png](output_24_1106.png)





![png](output_24_1107.png)





![png](output_24_1108.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))






    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 194, step 91000: Generator loss: 0.9479806874990463, discriminator loss: 0.6233925930261612




![png](output_24_1114.png)





![png](output_24_1115.png)





![png](output_24_1116.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 195, step 91500: Generator loss: 0.9384836674928665, discriminator loss: 0.6281133351922035




![png](output_24_1120.png)





![png](output_24_1121.png)





![png](output_24_1122.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 196, step 92000: Generator loss: 0.9260509090423584, discriminator loss: 0.6260301347970962




![png](output_24_1126.png)





![png](output_24_1127.png)





![png](output_24_1128.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 197, step 92500: Generator loss: 0.9363479852676392, discriminator loss: 0.6266585158109665




![png](output_24_1132.png)





![png](output_24_1133.png)





![png](output_24_1134.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 198, step 93000: Generator loss: 0.9431276168823243, discriminator loss: 0.6262624267339706




![png](output_24_1138.png)





![png](output_24_1139.png)





![png](output_24_1140.png)







    HBox(children=(FloatProgress(value=0.0, max=469.0), HTML(value='')))


    Epoch 199, step 93500: Generator loss: 0.929187188744545, discriminator loss: 0.6237477086186409




![png](output_24_1144.png)





![png](output_24_1145.png)





![png](output_24_1146.png)






## Exploration
You can do a bit of exploration now!


```python
# Before you explore, you should put the generator
# in eval mode, both in general and so that batch norm
# doesn't cause you issues and is using its eval statistics
gen = gen.eval()
```

#### Changing the Class Vector
You can generate some numbers with your new model! You can add interpolation as well to make it more interesting.

So starting from a image, you will produce intermediate images that look more and more like the ending image until you get to the final image. Your're basically morphing one image into another. You can choose what these two images will be using your conditional GAN.


```python
import math

### Change me! ###
n_interpolation = 9 # Choose the interpolation: how many intermediate images you want + 2 (for the start and end image)
interpolation_noise = get_noise(1, z_dim, device=device).repeat(n_interpolation, 1)

def interpolate_class(first_number, second_number):
    first_label = get_one_hot_labels(torch.Tensor([first_number]).long(), n_classes)
    second_label = get_one_hot_labels(torch.Tensor([second_number]).long(), n_classes)

    # Calculate the interpolation vector between the two labels
    percent_second_label = torch.linspace(0, 1, n_interpolation)[:, None]
    interpolation_labels = first_label * (1 - percent_second_label) + second_label * percent_second_label

    # Combine the noise and the labels
    noise_and_labels = combine_vectors(interpolation_noise, interpolation_labels.to(device))
    fake = gen(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation, nrow=int(math.sqrt(n_interpolation)), show=False)

### Change me! ###
start_plot_number = 1 # Choose the start digit
### Change me! ###
end_plot_number = 5 # Choose the end digit

plt.figure(figsize=(8, 8))
interpolate_class(start_plot_number, end_plot_number)
_ = plt.axis('off')

### Uncomment the following lines of code if you would like to visualize a set of pairwise class
### interpolations for a collection of different numbers, all in a single grid of interpolations.
### You'll also see another visualization like this in the next code block!
# plot_numbers = [2, 3, 4, 5, 7]
# n_numbers = len(plot_numbers)
# plt.figure(figsize=(8, 8))
# for i, first_plot_number in enumerate(plot_numbers):
#     for j, second_plot_number in enumerate(plot_numbers):
#         plt.subplot(n_numbers, n_numbers, i * n_numbers + j + 1)
#         interpolate_class(first_plot_number, second_plot_number)
#         plt.axis('off')
# plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.1, wspace=0)
# plt.show()
# plt.close()
```



![png](output_28_0.png)



#### Changing the Noise Vector
Now, what happens if you hold the class constant, but instead you change the noise vector? You can also interpolate the noise vector and generate an image at each step.


```python
n_interpolation = 9 # How many intermediate images you want + 2 (for the start and end image)

# This time you're interpolating between the noise instead of the labels
interpolation_label = get_one_hot_labels(torch.Tensor([5]).long(), n_classes).repeat(n_interpolation, 1).float()

def interpolate_noise(first_noise, second_noise):
    # This time you're interpolating between the noise instead of the labels
    percent_first_noise = torch.linspace(0, 1, n_interpolation)[:, None].to(device)
    interpolation_noise = first_noise * percent_first_noise + second_noise * (1 - percent_first_noise)

    # Combine the noise and the labels again
    noise_and_labels = combine_vectors(interpolation_noise, interpolation_label.to(device))
    fake = gen(noise_and_labels)
    show_tensor_images(fake, num_images=n_interpolation, nrow=int(math.sqrt(n_interpolation)), show=False)

# Generate noise vectors to interpolate between
### Change me! ###
n_noise = 5 # Choose the number of noise examples in the grid
plot_noises = [get_noise(1, z_dim, device=device) for i in range(n_noise)]
plt.figure(figsize=(8, 8))
for i, first_plot_noise in enumerate(plot_noises):
    for j, second_plot_noise in enumerate(plot_noises):
        plt.subplot(n_noise, n_noise, i * n_noise + j + 1)
        interpolate_noise(first_plot_noise, second_plot_noise)
        plt.axis('off')
plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0.1, wspace=0)
plt.show()
plt.close()
```



![png](output_30_0.png)


