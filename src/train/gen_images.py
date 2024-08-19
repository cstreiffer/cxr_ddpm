import torch
from torchvision.utils import save_image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os
from datetime import datetime

from models import get_device, load_pipeline

def generate_image_batch(model, noise_scheduler, x, context):
    model.eval()

    # Sampling loop
    for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

        # Get model pred
        with torch.no_grad():
            t = t.to(get_device())
            residual = model(x, t, context=context, return_dict=False)  # Again, note that we pass in our labels y

        # Update sample with step
        x = noise_scheduler.step(residual[0], t, x).prev_sample

    # Prepare images for saving
    generated_images = x.detach().cpu().clip(-1, 1)

    # Now return
    return generated_images

def save_images(images, output_path, file_names=None, format="png"):
  ret = []
  for i,img in enumerate(images):
    # Get the path
    if file_names is not None:
      file_path = os.path.join(output_path, file_names[i])
    else:
      file_path = os.path.join(output_path, f"image-{i:05}.{format}")

    # Now save
    save_image(img, file_path, format=format)
    ret.append(file_path)
  return ret

# Random sampling and generation run
def sample_image_noise(bs, image_size):
  return torch.randn((bs, 1, image_size, image_size))

def sample_context(
  bs,
  age_d=np.random.normal,
  sex_d=np.random.choice([0, 1]),
  ivsd_d=np.random.normal, 
  lvpwd_d=np.random.normal,
  lvidd_d=np.random.normal
):
  s = [sex_d() for i in range(bs)]
  return [[
      age_d(),
      s[i],
      1 if s[i] == 0 else 0,
      ivsd_d(),
      lvpwd_d(),
      lvidd_d()
  ] for i in range(bs)]

def recover_context(samples, age_stats, cont_feat_stats):
  ret = []
  for s in samples:
    ret.append([
        (s[0] * np.sqrt(age_stats[1])) + age_stats[0],
        s[1],
        s[2],
        (s[3] * np.sqrt(cont_feat_stats['ivsd'][1])) + cont_feat_stats['ivsd'][0],
        (s[4] * np.sqrt(cont_feat_stats['lvpwd'][1])) + cont_feat_stats['lvpwd'][0],
        (s[5] * np.sqrt(cont_feat_stats['lvidd'][1])) + cont_feat_stats['lvidd'][0]
    ])
  return ret

def gen_dict(file_paths, context_batch, context_labels, image_ids=None):
  formatted_data = []
  for i,f in enumerate(file_paths):
    d = {context_labels[j]:context_batch[i][j] for j in range(len(context_labels))}
    d['file_path'] = f
    d['image_id'] = os.path.basename(f) if image_ids is None else image_ids[i]
    formatted_data.append(d)
  return formatted_data

def gen(
  model_path,
  output_path,
  num_batches,
  bs,
  sample_fn=None
):
  # Create the output path
  if not os.path.exists(output_path):
    os.makedirs(output_path)

  # Format the date for the run
  run_date = datetime.now().strftime("%Y-%m-%d")

  # Load the pipeline
  model, scheduler, stats = load_pipeline(model_path)

  # Now place on device
  model = model.to(get_device())

  # Create output paths
  os.makedirs(output_path, exist_ok=True)
  image_path = os.path.join(output_path, 'images')
  # Create the output path
  if not os.path.exists(image_path):
    os.makedirs(image_path)
    
  csv_path   = os.path.join(output_path, 'gen_metadata.csv')
  tot_data = []

  for i in range(num_batches):
    print(f"{datetime.now()} - Running for batch: {i:03d}")

    # Sample from feature distribution
    context = None
    if sample_fn is not None:
      context = sample_fn(bs)
    else:
      context = sample_context(bs)

    # Generate random noise
    image_noise = sample_image_noise(bs, stats['args'].dataset_settings['downsample_size'])

    # Now place
    context = torch.Tensor(context).to(get_device())
    image_noise = image_noise.to(get_device())

    # 2. Generate images
    gen_images = generate_image_batch(
        model,
        scheduler,
        image_noise,
        context
    )

    # 3. Save image
    file_names = [f'image-{i*bs+j:05}-{run_date}.png' for j in range(bs)]

    file_paths = save_images(
        gen_images,
        image_path,
        file_names=file_names
    )

    # 4. Save recovered output features
    context_denorm = recover_context(
        context.detach().cpu().numpy().tolist(),
        stats['args'].eval_dataset_settings['age_stats'],
        stats['args'].eval_dataset_settings['cont_feat_stats']
    )

    # Now generate the dict and save
    data = gen_dict(
        file_paths,
        context_denorm,
        stats['args'].dataset_settings['label_cols']
    )
    tot_data.extend(data)

    # Now store
    pd.DataFrame(tot_data).to_csv(csv_path, index=False)

  return pd.DataFrame(tot_data)