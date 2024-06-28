# Diffusion_project


## DDPM are a class of unconditional generative models. 

### In this project, we have built a diffusion model to generate CIFAR10 images.


In order to generate the images, 

Install the necessary dependencies using 

```bash
pip install requirements.txt
```

Additionally, git lfs must be installed, if running locally, from [Git LFS](https://git-lfs.com/).

Run the following code! 

```bash
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import os
import torch


pipeline = DiffusionPipeline.from_pretrained("balakrish181/cifar10")

images = pipeline(
batch_size=16,
generator=torch.Generator(device='cpu').manual_seed(1), 
).images

image_grid = make_image_grid(images, rows=4, cols=4)

image_grid.save("sample_image.png")
```

This code snippet uses the DiffusionPipeline class from the diffusers library to generate 16 CIFAR10 images using a pre-trained diffusion model. The generated images are then saved as a grid in a single image named "sample_image.png".



### Sample Generated Images 

![Generated image]('https://github.com/balakrish181/diffusion_project/blob/main/cifar_gen_iamge.png')

