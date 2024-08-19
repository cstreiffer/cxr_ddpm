import torch
from torch import nn
from diffusers import UNet2DModel, UNet2DConditionModel, DDPMScheduler
import os
import glob

class StateEmbedding(nn.Module):
    def __init__(self, n_state, out_channels):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(n_state, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.linear(x)

class ClassConditionedUnet(nn.Module):
  def __init__(self, model, context_size, n_channels=0):
    super().__init__()

    # Get the time embedding
    self.time_embed_dim = n_channels*4
    
    # The embedding layer will map the class label to a vector of size class_emb_size
    self.state_emb = StateEmbedding(context_size, self.time_embed_dim)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = model

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, context, return_dict=False):
    # Class conditioning
    context = self.state_emb(context) # Map to embedding dinemsion

    # Feed this to the unet alongside the timestep and return the prediction
    return self.model(x, t, class_labels=context, return_dict=return_dict)

class InputConditionedUnet(nn.Module):
  def __init__(self, model):
    super().__init__()

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = model

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, context, return_dict=False):
    # Shape of x:
    bs, ch, w, h = x.shape
    
    # Class conditioning in right shape to add as additional input channels
    context = context.view(bs, -1, 1, 1).expand(bs, -1, w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)
    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, context), 1) # (bs, 6, 28, 28)

    # Feed this to the unet alongside the timestep and return the prediction
    return self.model(net_input, t, return_dict=return_dict)

class BasicUnet(nn.Module):
  def __init__(self, model):
    super().__init__()

    # Store the model
    self.model = model

  def forward(self, x, t, context, return_dict=False):
    return self.model(net_input, t, return_dict=return_dict)

def load_basic_diffusion_model(image_size, context_size):
  model = UNet2DModel(
      sample_size=image_size,  # the target image resolution
      in_channels=1,  # the number of input channels, 3 for RGB images
      out_channels=1,  # the number of output channels
      layers_per_block=2,  # how many ResNet layers to use per UNet block
      block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
      down_block_types=(
          "DownBlock2D",  # a regular ResNet downsampling block
          "DownBlock2D",
          "DownBlock2D",
          "DownBlock2D",
          "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
          "DownBlock2D",
      ),
      up_block_types=(
          "UpBlock2D",  # a regular ResNet upsampling block
          "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
      ),
  )
  return BasicUnet(model)

def load_class_diffusion_model(image_size, context_size):
  model = UNet2DModel(
      sample_size=image_size,  # the target image resolution
      in_channels=1,  # the number of input channels, 3 for RGB images
      out_channels=1,  # the number of output channels
      layers_per_block=2,  # how many ResNet layers to use per UNet block
      block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
      down_block_types=(
          "DownBlock2D",  # a regular ResNet downsampling block
          "DownBlock2D",
          "DownBlock2D",
          "DownBlock2D",
          "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
          "DownBlock2D",
      ),
      up_block_types=(
          "UpBlock2D",  # a regular ResNet upsampling block
          "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
      ),
      class_embed_type="identity"
  )
  return ClassConditionedUnet(model, context_size, n_channels=128)

def load_class_diffusion_model_large(image_size, context_size):
  model = UNet2DModel(
      sample_size=image_size,  # the target image resolution
      in_channels=1,  # the number of input channels, 3 for RGB images
      out_channels=1,  # the number of output channels
      layers_per_block=2,
      block_out_channels=(128, 256, 256, 512, 512, 1024),
      down_block_types=(
          "DownBlock2D",
          "DownBlock2D",
          "DownBlock2D",
          "AttnDownBlock2D",
          "AttnDownBlock2D",
          "DownBlock2D",
      ),
      up_block_types=(
          "UpBlock2D",
          "AttnUpBlock2D",
          "AttnUpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
      ),
      class_embed_type="identity"
  )
  return ClassConditionedUnet(model, context_size, n_channels=128)

def load_input_diffusion_model_large(image_size, context_size):
  model = UNet2DModel(
      sample_size=image_size,
      in_channels=1 + context_size,  # Adjusted based on input features
      out_channels=1,
      layers_per_block=2,
      block_out_channels=(128, 256, 256, 512, 512, 1024),
      down_block_types=(
          "DownBlock2D",
          "DownBlock2D",
          "DownBlock2D",
          "AttnDownBlock2D",
          "AttnDownBlock2D",
          "DownBlock2D",
      ),
      up_block_types=(
          "UpBlock2D",
          "AttnUpBlock2D",
          "AttnUpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
      )
  )
  return InputConditionedUnet(model)

def load_input_diffusion_model(image_size, context_size):
  model = UNet2DModel(
      sample_size=image_size,  # the target image resolution
      in_channels=1+context_size,  # the number of input channels, 3 for RGB images
      out_channels=1,  # the number of output channels
      layers_per_block=2,  # how many ResNet layers to use per UNet block
      block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
      down_block_types=(
          "DownBlock2D",  # a regular ResNet downsampling block
          "DownBlock2D",
          "DownBlock2D",
          "DownBlock2D",
          "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
          "DownBlock2D",
      ),
      up_block_types=(
          "UpBlock2D",  # a regular ResNet upsampling block
          "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
          "UpBlock2D",
      )
  )
  return InputConditionedUnet(model)

# Load the model
class_mapping = {
  "basic_diffusion":       load_basic_diffusion_model,
  "input_diffusion":       load_input_diffusion_model,
  "input_diffusion_large": load_input_diffusion_model_large,
  "class_diffusion":       load_class_diffusion_model,
  "class_diffusion_large": load_class_diffusion_model_large
}
def load_model(name, image_size, context_size):
  return class_mapping[name](image_size, context_size)

# Load the model state
def get_device():
  return "cuda" if torch.cuda.is_available() else "cpu"

def load_model_state(model, path, optimizer=None):
  checkpoint = torch.load(path, map_location=torch.device(get_device()))

  model.load_state_dict(checkpoint['model_state_dict'])
  if optimizer:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def load_pipeline(model_path, epoch=None):
  # 0. Load stats/args
  stats_path = os.path.join(model_path, "models_pth", "diffusion_checkpoint_stats.pth")
  model_stats = torch.load(stats_path)

  # 1. Create the scheduler
  noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

  # 2. Create the model
  model_name = model_stats['args'].model_name
  image_size = model_stats['args'].dataset_settings['downsample_size']
  context_size = model_stats['args'].num_feats
  model = load_model(model_name, image_size, context_size)

  # 3. Find the correct model
  files = glob.glob(os.path.join(model_path, "models_pth", "diffusion_checkpoint_epoch*"))
  if epoch is None:
    epoch = max([int(os.path.basename(n).split('_')[3]) for n in files])

  # 4. Get the correct model
  file = list(filter(lambda x: int(os.path.basename(x).split('_')[3]) == epoch, files))[0]
  print(f"Loaded {file}")
  
  # Now load
  load_model_state(model, file)

  # Now return
  return model, noise_scheduler, model_stats

# Model Other - Not used
# class_embed_type="projection"
# num_class_embeds=4
# projection_class_embeddings_input_dim=128
# class_embeddings_concat=True
# def load_cross_attention_model(image_size, context_size, emb_size):
#   model = UNet2DConditionModel(
#     sample_size=image_size,         # the target image resolution, as set above
#     in_channels=1,            # Additional input channels for class cond.
#     out_channels=1,           # the number of output channels
#     layers_per_block=2,       # how many ResNet layers to use per UNet block
#     block_out_channels=(32, 64, 128),
#     down_block_types=(
#       "CrossAttnDownBlock2D",
#       "CrossAttnDownBlock2D",
#       "DownBlock2D"
#     ),
#     mid_block_type="UNetMidBlock2DCrossAttn",
#     up_block_types=(
#         "UpBlock2D",
#         "CrossAttnUpBlock2D",
#         "CrossAttnUpBlock2D",
#       ),
#     time_embedding_type='positional', # Or fourier
#     dropout=0.1,
#     cross_attention_dim=emb_size,
#     encoder_hid_dim_type="text_proj",
#     encoder_hid_dim=context_size
#   )
#   return model