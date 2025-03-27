# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import logging
import os
import time
from torchvision.utils import save_image
import numpy as np
import numpy.ma as ma
from tqdm import tqdm
import wandb
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import functional as F
from typing import List, Tuple
import math 
from lib.utils.utils import *
from lib.utils.points_from_den import local_maximum_points
from lib.eval.eval_loc_count import eval_loc_MLE_point, eval_loc_F1_boxes
import matplotlib.pyplot as plt

def denormalize(normalized_image):
        # Step 1: Define hardcoded mean and std (RGB)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # Shape: (1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)    # Shape: (1, 1, 3)

    # Step 2: Move tensors to CPU and convert to NumPy
    with torch.no_grad():
        image_cpu = normalized_image.cpu().numpy()  # Shape: (3, H, W)
        

    # Step 3: Transpose image to (H, W, C)
    denorm_image = np.transpose(image_cpu, (1, 2, 0))  # Shape: (H, W, 3)

    # Step 4: Denormalize the image
    denorm_image = denorm_image * std + mean          # Reverse normalization
    denorm_image = denorm_image * 255.0              # Scale to [0, 255]

    # Reverse channel order if needed (RGB to BGR)

    #denorm_image = denorm_image[:, :, ::-1]       # Shape: (H, W, 3)

    # Clip values to [0, 255] and convert to uint8
    denorm_image = np.clip(denorm_image, 0, 255).astype(np.uint8)
    return denorm_image

def save_torch_image(image,name='prova.png'):
    im = denormalize(image.cpu().squeeze(0))
    pil_image = Image.fromarray(im)
    pil_image.save(os.path.join('./debug/', name))

def denormalize_and_blend_torch(normalized_image, density_map, output_path, reverse_channel=True, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """
    Denormalizes the image with hardcoded mean and std, blends it with a density map heatmap, and saves the result.

    Parameters:
    - normalized_image (torch.Tensor): The normalized image tensor with shape (3, H, W) on CUDA.
    - density_map (torch.Tensor): The density map tensor with shape (1, H, W) on CUDA.
    - output_path (str): Path where the blended image will be saved.
    - reverse_channel (bool): If True, reverses the channel order (e.g., RGB to BGR).
    - alpha (float): Transparency factor for blending (0.0 transparent through 1.0 opaque).
    - colormap (int): OpenCV colormap to apply to the density map.

    Returns:
    - None: The function saves the blended image to the specified path.
    """

    # Step 1: Define hardcoded mean and std (RGB)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)  # Shape: (1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)    # Shape: (1, 1, 3)

    # Step 2: Move tensors to CPU and convert to NumPy
    with torch.no_grad():
        image_cpu = normalized_image.cpu().numpy()  # Shape: (3, H, W)
        density_cpu = density_map.cpu().numpy()    # Shape: (1, H, W)

    # Step 3: Transpose image to (H, W, C)
    denorm_image = np.transpose(image_cpu, (1, 2, 0))  # Shape: (H, W, 3)

    # Step 4: Denormalize the image
    denorm_image = denorm_image * std + mean          # Reverse normalization
    denorm_image = denorm_image * 255.0              # Scale to [0, 255]

    # Reverse channel order if needed (RGB to BGR)
    if reverse_channel:
        denorm_image = denorm_image[:, :, ::-1]       # Shape: (H, W, 3)

    # Clip values to [0, 255] and convert to uint8
    denorm_image = np.clip(denorm_image, 0, 255).astype(np.uint8)

    # Step 5: Process density map
    density_map_np = density_cpu.squeeze(0)           # Shape: (H, W)

    # Normalize density map to [0, 255]
    density_normalized = cv2.normalize(density_map_np, None, 0, 255, cv2.NORM_MINMAX)
    density_normalized = density_normalized.astype(np.uint8)

    # Apply colormap to create heatmap
    heatmap = cv2.applyColorMap(density_normalized, colormap)  # Shape: (H, W, 3)

    # Step 6: Blend heatmap with denormalized image
    blended_image = cv2.addWeighted(denorm_image, alpha, heatmap, 1 - alpha, 0)

    # Step 7: Save the blended image using OpenCV
    success = cv2.imwrite(output_path, blended_image)
    if success:
        print(f"Blended image successfully saved at: {output_path}")
    else:
        print(f"Failed to save blended image at: {output_path}")

def save_blended_image(blended_image, output_path, use_matplotlib=False):
    """
    Saves the blended image to the specified file path.

    Parameters:
    - blended_image (np.ndarray): The image to save.
    - output_path (str): Path where the image will be saved.
    - use_matplotlib (bool): If True, saves using Matplotlib (useful for RGB images).
                              If False, saves using OpenCV (BGR format).
    """
    if use_matplotlib:
        import matplotlib.pyplot as plt
        # Convert BGR to RGB for Matplotlib
        blended_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
        plt.imsave(output_path, blended_rgb)
    else:
        # Save using OpenCV
        cv2.imwrite(output_path, blended_image)

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that 
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return inp
    with torch.no_grad():
        reduced_inp = inp
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp
def allreduce_tensor(inp):
    """
    Reduce the loss from all processes so that
    process with rank 0 has the averaged results.
    """
    world_size = get_world_size()
    if world_size < 2:
        return  None
    dist.all_reduce(inp,op=dist.ReduceOp.SUM)

def train(config, epoch, num_epoch, epoch_iters, num_iters,
         trainloader, optimizer,scheduler, model, writer_dict, device,img_vis_dir,mean,std,task_KPI,train_dataset):
    
    # Training
    model.train()
    batch_time = AverageMeter()
    avg_loss = AverageMeter()
    tic = time.time()

    world_size = get_world_size()
    model.zero_grad()
    accumulation_steps = 1
    print(f'Using {accumulation_steps} accumulation_steps')
    for i_iter, batch in enumerate(tqdm(trainloader)):
        images, label, size, name_idx, human_num = batch
        images = images.to(device)
        for i in range(len(label)):
            label[i] = label[i].to(device)

        result = model(images, label, 'train')
        losses=result['losses']
        pre_den=result['pre_den']['1']
        gt_den = result['gt_den']['1']

        for i in range(len(name_idx[0])):
            _name  = name_idx[0][i]
            if _name not in train_dataset.resize_memory_pool.keys():
                p_h= int(np.ceil(size[i][0]/config.train.route_size[0]))
                p_w = int(np.ceil(size[i][1]/config.train.route_size[1]))
                train_dataset.resize_memory_pool.update({_name:{"avg_size":np.ones((p_h,p_w)),
                                                      "load_num":  np.zeros((p_h,p_w)),
                                                       'size':np.array(size)}})

        loss = losses.mean() / accumulation_steps
        loss.backward()
        if (i_iter + 1) % accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()

        task_KPI.add({
                'acc1': {'gt': result['acc1']['gt'], 'error': result['acc1']['error']},
                'x4': {'gt':result['x4']['gt'], 'error':result['x4']['error']},
              'x8': {'gt': result['x8']['gt'], 'error': result['x8']['error']},
              'x16': {'gt': result['x16']['gt'], 'error': result['x16']['error']},
              'x32': {'gt': result['x32']['gt'], 'error': result['x32']['error']}

                      })
        with torch.no_grad():
            KPI = task_KPI.query()
            reduced_loss = reduce_tensor(loss)
        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()
        # update average loss
        avg_loss.update(reduced_loss.item())
        #
        scheduler.step_update(epoch * epoch_iters + i_iter)
        lr = optimizer.param_groups[0]['lr']
        gt_cnt, pred_cnt = label[0].sum().item() , pre_den.sum().item()
        if config.log:
            wandb.log({"MAE": abs(gt_cnt-pred_cnt), "loss": loss.item(), "lr":lr*1e5})
    
    if (i_iter + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
        

def validate(config, testloader, model, writer_dict, device,
             num_patches,img_vis_dir,mean,std):
    
    rank = get_rank()
    world_size = get_world_size()
    model.eval()
    avg_loss = AverageMeter()
    cnt_errors = {'mae': AverageMeter(), 'ppmae': AverageMeter(), 'mse': AverageMeter(),
                  'nae': AverageMeter(),'acc1':AverageMeter()}
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(testloader)):
            image, label, _, name, human_num = batch
            image = image.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)
            # result = model(image, label, 'val')
            result = patch_forward(model, image, label, num_patches,'val')
            losses=result['losses']
            pre_den=result['pre_den']['1']
            gt_den = result['gt_den']['1']
            gt_count, pred_cnt = label[0].sum(), pre_den.sum()
            s_mae = torch.abs(gt_count - pred_cnt)
            s_ppmae = s_mae/human_num.to(s_mae.device)
            s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
            allreduce_tensor(s_mae)
            allreduce_tensor(s_mse)
            allreduce_tensor(s_ppmae)
            reduced_loss = reduce_tensor(losses)
            avg_loss.update(reduced_loss.item())
            cnt_errors['mae'].update(s_mae.item())
            cnt_errors['mse'].update(s_mse.item())
            cnt_errors['ppmae'].update(s_ppmae.item())
            s_nae = (torch.abs(gt_count - pred_cnt) / (gt_count+1e-10))
            allreduce_tensor(s_nae)
            cnt_errors['nae'].update(s_nae.item())
    print_loss = avg_loss.average()/world_size
    mae = cnt_errors['mae'].avg / world_size
    ppmae = cnt_errors['ppmae'].avg / world_size
    mse = np.sqrt(cnt_errors['mse'].avg / world_size)
    nae = cnt_errors['nae'].avg / world_size
    if rank == 0:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', print_loss, global_steps)
        writer.add_scalar('valid_mae', mae, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1
    return print_loss, mae, mse, nae, ppmae


def patch_forward(model, img, dot_map, num_patches,mode):
    # crop the img and gt_map with a max stride on x and y axis
    # size: HW: __C_NWPU.TRAIN_SIZE
    # stack them with a the batchsize: __C_NWPU.TRAIN_BATCH_SIZE
    crop_imgs = []
    crop_dots, crop_masks = {},{}

    crop_dots['1'],crop_dots['2'],crop_dots['4'],crop_dots['8'] = [],[],[],[]
    crop_masks['1'],crop_masks['2'],crop_masks['4'],crop_masks['8'] = [], [], [],[]
    b, c, h, w = img.shape
    rh, rw = 768, 1024

    # support for multi-scale patch forward
    for i in range(0, h, rh):
        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
        for j in range(0, w, rw):
            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

            crop_imgs.append(img[:, :, gis:gie, gjs:gje])
            for res_i in range(len(dot_map)):
                gis_,gie_ = gis//2**res_i, gie//2**res_i
                gjs_,gje_ = gjs//2**res_i, gje//2**res_i
                crop_dots[str(2**res_i)].append(dot_map[res_i][:, gis_:gie_, gjs_:gje_])
                mask = torch.zeros_like(dot_map[res_i]).cpu()
                mask[:, gis_:gie_, gjs_:gje_].fill_(1.0)
                crop_masks[str(2**res_i)].append(mask)

    crop_imgs = torch.cat(crop_imgs, dim=0)
    for k,v in crop_dots.items():
        crop_dots[k] =  torch.cat(v, dim=0)
    for k,v in crop_masks.items():
        crop_masks[k] =  torch.cat(v, dim=0)

    # forward may need repeatng
    crop_losses = []
    crop_preds = {}
    crop_labels = {}
    crop_labels['1'],crop_labels['2'],crop_labels['4'],crop_labels['8'] = [],[],[],[]
    crop_preds['1'],crop_preds['2'],crop_preds['4'],crop_preds['8'] = [], [], [],[]
    nz, bz = crop_imgs.size(0), num_patches
    keys_pre = None

    for i in range(0, nz, bz):
        gs, gt = i, min(nz, i + bz)
        result = model(crop_imgs[gs:gt], [crop_dots[k][gs:gt] for k in crop_dots.keys() ],
                                          mode)
        crop_pred = result['pre_den']
        crop_label =  result['gt_den']

        keys_pre = result['pre_den'].keys()
        for k in keys_pre:
            crop_preds[k].append(crop_pred[k].cpu())
            crop_labels[k].append(crop_label[k].cpu())

        crop_losses.append(result['losses'].mean())

    for k in keys_pre:
        crop_preds[k] =  torch.cat(crop_preds[k], dim=0)
        crop_labels[k] =  torch.cat(crop_labels[k], dim=0)


    # splice them to the original size

    result = {'pre_den': {},'gt_den':{}}

    for res_i, k in enumerate(keys_pre):

        pred_map = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
        labels = torch.zeros_like(dot_map[res_i]).unsqueeze(0).cpu().float()
        idx =0
        for i in range(0, h, rh):
            gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
            for j in range(0, w, rw):
                gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

                gis_,gie_ = gis//2**res_i, gie//2**res_i
                gjs_,gje_ = gjs//2**res_i, gje//2**res_i

                pred_map[:,:, gis_:gie_, gjs_:gje_] += crop_preds[k][idx]
                labels[:,:, gis_:gie_, gjs_:gje_] += crop_labels[k][idx]
                idx += 1
        # import pdb
        # pdb.set_trace()
        # for the overlapping area, compute average value
        mask = crop_masks[k].sum(dim=0).unsqueeze(0).unsqueeze(0)
        pred_map = (pred_map / mask)
        labels = (labels / mask)
        result['pre_den'].update({k: pred_map} )
        result['gt_den'].update({k: labels} )
        result.update({'losses': crop_losses[0]} )
    return result

def extract_patches(
    image: torch.Tensor,
    patch_size: int = 512,
    overlap: float = 0.25
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    """
    Splits the image into overlapping patches.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch, channels, height, width).
        patch_size (int): Size of each square patch.
        overlap (float): Fraction of overlap between patches (e.g., 0.25 for 25% overlap).

    Returns:
        patches (List[torch.Tensor]): List of image patches.
        coords (List[Tuple[int, int]]): List of (h_start, w_start) coordinates for each patch.
    """
    batch, channels, height, width = image.shape
    stride = int(patch_size * (1 - overlap))
    
    # Calculate the number of patches along height and width
    num_patches_h = math.ceil((height - patch_size) / stride) + 1
    num_patches_w = math.ceil((width - patch_size) / stride) + 1
    
    # Calculate required padding
    pad_h = max((num_patches_h - 1) * stride + patch_size - height, 0)
    pad_w = max((num_patches_w - 1) * stride + patch_size - width, 0)
    
    # Apply padding to the image
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
    
    patches = []
    coords = []
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + patch_size
            w_end = w_start + patch_size
            
            patch = image[:, :, h_start:h_end, w_start:w_end]
            patches.append(patch)
            coords.append((h_start, w_start))
    
    return patches, coords

def extract_patches(
    image: torch.Tensor,
    patch_size: int = 512
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]], Tuple[int, int]]:
    """
    Splits the image into non-overlapping patches.

    Args:
        image (torch.Tensor): Input image tensor of shape (batch, channels, height, width).
        patch_size (int): Size of each square patch.

    Returns:
        patches (List[torch.Tensor]): List of image patches.
        coords (List[Tuple[int, int]]): List of (h_start, w_start) coordinates for each patch.
        original_size (Tuple[int, int]): Original image size as (height, width).
    """
    batch, channels, height, width = image.shape
    stride = patch_size  # No overlap

    # Calculate the number of patches along height and width
    num_patches_h = math.ceil(height / patch_size)
    num_patches_w = math.ceil(width / patch_size)

    # Calculate required padding
    pad_h = max(num_patches_h * patch_size - height, 0)
    pad_w = max(num_patches_w * patch_size - width, 0)

    # Apply padding to the image
    if pad_h > 0 or pad_w > 0:
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')

    patches = []
    coords = []

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            h_start = i * stride
            w_start = j * stride
            h_end = h_start + patch_size
            w_end = w_start + patch_size

            patch = image[:, :, h_start:h_end, w_start:w_end]
            patches.append(patch)
            coords.append((h_start, w_start))

    original_size = (height, width)
    return patches, coords, original_size

def reconstruct_image(
    patches: List[torch.Tensor],
    coords: List[Tuple[int, int]],
    original_size: Tuple[int, int],
    patch_size: int = 512
) -> torch.Tensor:
    """
    Reconstructs the full image from processed patches.

    Args:
        patches (List[torch.Tensor]): List of processed image patches.
        coords (List[Tuple[int, int]]): List of (h_start, w_start) coordinates for each patch.
        original_size (Tuple[int, int]): Original image size as (height, width).
        patch_size (int): Size of each square patch.

    Returns:
        reconstructed (torch.Tensor): Reconstructed image tensor of shape (batch, channels, height, width).
    """
    height, width = original_size
    batch, channels, _, _ = patches[0].shape

    # Calculate the padded size
    num_patches_h = math.ceil(height / patch_size)
    num_patches_w = math.ceil(width / patch_size)
    padded_height = num_patches_h * patch_size
    padded_width = num_patches_w * patch_size

    # Initialize the reconstruction tensor
    device = patches[0].device
    reconstructed = torch.zeros((batch, channels, padded_height, padded_width), device=device)

    for patch, (h_start, w_start) in zip(patches, coords):
        h_end = h_start + patch_size
        w_end = w_start + patch_size

        reconstructed[:, :, h_start:h_end, w_start:w_end] = patch

    # Crop to the original size
    reconstructed = reconstructed[:, :, :height, :width]

    return reconstructed

def my_patch_forward(
    model: torch.nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    patch_size: int = 512,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    batch_size: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Splits the image into non-overlapping patches, processes each patch through the model,
    and reconstructs the full image from the processed patches.

    Args:
        model (torch.nn.Module): The neural network model to process each patch.
        image (torch.Tensor): Input image tensor of shape (batch, channels, height, width).
        label (torch.Tensor): Corresponding labels for the image (unused in patching).
        patch_size (int): Size of each square patch.
        device (torch.device): Device to perform computations on.
        batch_size (int): Number of patches to process in a single batch.

    Returns:
        reconstructed_output (torch.Tensor): Reconstructed output image from the model.
        label (torch.Tensor): Original labels (unchanged).
    """
    model.eval()
    image = image.to(device)
    model = model.to(device)

    # Extract patches
    patches, coords, original_size = extract_patches(image, patch_size)

    # Process patches in batches
    processed_patches = []
    num_patches = len(patches)
    batch_size = num_patches
    with torch.no_grad():
        for i in range(0, num_patches, batch_size):
            batch_patches = torch.cat(patches[i:i+batch_size], dim=0)  # Shape: (batch_size, C, H, W)
            outputs = model(batch_patches, labels=batch_patches, mode='patch')['pre_den']['1']  # Assuming model outputs same spatial dimensions
            #outputs = batch_patches
            processed_patches.extend(outputs.chunk(outputs.size(0), dim=0))

    # Reconstruct the full image from processed patches
    reconstructed_output = reconstruct_image(processed_patches, coords, original_size, patch_size)


    return reconstructed_output.cpu(), label

def test_cc(config, test_dataset, testloader, model
            ,mean, std, sv_dir='', sv_pred=False,logger=None):

    model.eval()
    print(f"num parameters: {sum(p.numel() for p in model.parameters())}")
    save_count_txt = ''
    device = torch.cuda.current_device()
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter(), 'ppmae': AverageMeter()}
    cnt = 0
    
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, _, name, human_num = batch
            image = image.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)
            if cnt < 100:
                start_batch = torch.cuda.Event(enable_timing=True)
                end_batch = torch.cuda.Event(enable_timing=True)
                start_batch.record()
            result = model(image, label, 'val')
            losses=result['losses']
            pre_den=result['pre_den']['1']
            gt_count, pred_cnt = label[0].sum().item(), pre_den.sum().item() 
            save_count_txt+='{} {}\n'.format(name[0], pred_cnt)
            msg = '{} {}' .format(gt_count,pred_cnt)
            logger.info(msg)
            s_mae = abs(gt_count - pred_cnt)
            s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
            cnt_errors['mae'].update(s_mae)
            cnt_errors['mse'].update(s_mse)
            if gt_count != 0:
                s_nae = (abs(gt_count - pred_cnt) / gt_count)
                cnt_errors['nae'].update(s_nae)
                ppmae = s_mae/human_num.item() if human_num.item() > 0 else 0
                cnt_errors['ppmae'].update(ppmae)
            image = image[0]
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                mae = cnt_errors['mae'].avg
                mse = np.sqrt(cnt_errors['mse'].avg)
                nae = cnt_errors['nae'].avg
                msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
                       nae: {: 4.4f}, ppmae: {: 4.4f} Class IoU: '.format(mae,
                                                          mse, nae, cnt_errors['ppmae'].avg)
                logging.info(msg)
        mae = cnt_errors['mae'].avg
        mse = np.sqrt(cnt_errors['mse'].avg)
        nae = cnt_errors['nae'].avg
        final_ppmae = cnt_errors['ppmae'].avg
    return  mae, mse, nae,save_count_txt, final_ppmae

def test_loc(config, test_dataset, testloader, model
            ,mean, std, sv_dir='', sv_pred=False,logger=None,loc_gt=None):

    model.eval()
    device = torch.cuda.current_device()
    cnt_errors = {'mae': AverageMeter(), 'mse': AverageMeter(), 'nae': AverageMeter()}
    num_classes = 6
    max_dist_thresh = 100
    metrics_s = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes),
                 'fn_c': AverageCategoryMeter(num_classes)}
    metrics_l = {'tp': AverageMeter(), 'fp': AverageMeter(), 'fn': AverageMeter(), 'tp_c': AverageCategoryMeter(num_classes),
                 'fn_c': AverageCategoryMeter(num_classes)}

    loc_100_metrics = {'tp_100': AverageCategoryMeter(max_dist_thresh), 'fp_100': AverageCategoryMeter(max_dist_thresh), 'fn_100': AverageCategoryMeter(max_dist_thresh)}

    MLE_metric = AverageMeter()
    with torch.no_grad():
        for index, batch in enumerate(tqdm(testloader)):
            image, label, size_factor, name = batch
            # if name[0] != '1202':
            #     continue

            image = image.to(device)
            for i in range(len(label)):
                label[i] = label[i].to(device)

            b, c, h, w = image.size()

            result = model(image, label, 'val')
            # result = patch_forward(model, image, label,
            #                        config.test.patch_batch_size, mode='val')
            # import pdb
            # pdb.set_trace()

            losses=result['losses']
            pre_den=result['pre_den']['1']
            # pre_den_x2=result['pre_den']['2']
            pre_den_x4=result['pre_den']['4']
            pre_den_x8=result['pre_den']['8']

            gt_den = result['gt_den']['1']
            # gt_den_x8 = result['gt_den']['8']

            gt_data = loc_gt[int(name[0])]

            pred_data = local_maximum_points(pre_den.detach(),model.gaussian_maximum, patch_size=32,threshold=config.test.loc_threshold)
            # pred_data_x2 = local_maximum_points(pre_den_x2.detach(),model.gaussian_maximum,patch_size=64,den_scale=2)
            pred_data_x4 = local_maximum_points(pre_den_x4.detach(),model.gaussian_maximum,patch_size=32,den_scale=4,threshold=config.test.loc_threshold)
            pred_data_x8 = local_maximum_points(pre_den_x8.detach(),model.gaussian_maximum,patch_size=16,den_scale=8,threshold=config.test.loc_threshold)

            def nms4points(pred_data, pred_data_x8, threshold):
                points = torch.from_numpy(pred_data['points']).unsqueeze(0)
                points_x8 =  torch.from_numpy(pred_data_x8['points']).unsqueeze(0)


                dist = torch.cdist(points,points_x8)     #torch.Size([1, 16, 16])
                dist = dist.squeeze(0)
                min_val, min_idx = torch.min(dist,0)
                keep_idx_bool = (min_val>threshold)


                keep_idx=torch.where(keep_idx_bool==1)[0]
                if keep_idx.size(0)>0:
                    app_points = (pred_data_x8['points'][keep_idx]).reshape(-1,2)
                    pred_data['points'] = np.concatenate([pred_data['points'], app_points],0)
                    pred_data['num'] =  pred_data['num'] +keep_idx_bool.sum().item()
                return pred_data
            #
            # if name[0] == '3613':
            #     import pdb
            #     pdb.set_trace()
            for idx, down_scale in enumerate([pred_data_x4,pred_data_x8]):
                if pred_data['points'].shape[0]==0 and down_scale['points'].shape[0]>0:
                    pred_data = down_scale
                if pred_data['points'].shape[0]>0  and down_scale['points'].shape[0]>0:
                    pred_data = nms4points(pred_data, down_scale,threshold=(2**(idx+1))*16)

            pred_data_4val  = pred_data.copy()
            pred_data_4val['points'] = pred_data_4val['points']/size_factor.numpy()
            tp_s, fp_s, fn_s, tp_c_s, fn_c_s, tp_l, fp_l, fn_l, tp_c_l, fn_c_l = eval_loc_F1_boxes(num_classes, pred_data_4val, gt_data)

            tp_100, fp_100, fn_100 =  0,0,0 #eval_loc_F1_point(pred_data['points'],gt_data['points'],max_dist_thresh = max_dist_thresh)
            Distance_Sum = eval_loc_MLE_point(pred_data['points'], gt_data['points'], 16)

            #    -----------Counting performance------------------
            gt_count, pred_cnt = label[0].sum().item(), pre_den.sum().item() #
            msg = '{}: gt:{} pre:{}' .format(name, gt_count,pred_cnt)
            logger.info(msg)
            # print(name,':', gt_count, pred_cnt)
            s_mae = abs(gt_count - pred_cnt)
            s_mse = ((gt_count - pred_cnt) * (gt_count - pred_cnt))
            cnt_errors['mae'].update(s_mae)
            cnt_errors['mse'].update(s_mse)
            if gt_count != 0:
                s_nae = (abs(gt_count - pred_cnt) / gt_count)
                cnt_errors['nae'].update(s_nae)

            MLE_metric.update(Distance_Sum/(gt_data['num']+1e-20), gt_data['num'])

            metrics_l['tp'].update(tp_l)
            metrics_l['fp'].update(fp_l)
            metrics_l['fn'].update(fn_l)
            metrics_l['tp_c'].update(tp_c_l)
            metrics_l['fn_c'].update(fn_c_l)

            metrics_s['tp'].update(tp_s)
            metrics_s['fp'].update(fp_s)
            metrics_s['fn'].update(fn_s)
            metrics_s['tp_c'].update(tp_c_s)
            metrics_s['fn_c'].update(fn_c_s)

            loc_100_metrics['tp_100'].update(tp_100)
            loc_100_metrics['fp_100'].update(fp_100)
            loc_100_metrics['fn_100'].update(fn_100)


            image = image[0]
            if sv_pred:
                for t, m, s in zip(image, mean, std):
                    t.mul_(s).add_(m)

                save_results_more(name, sv_dir, image.cpu().data, \
                                  pre_den[0].detach().cpu(), gt_den[0].detach().cpu(),pred_cnt,gt_count,
                                  pred_data['points'],gt_data['points']*size_factor.numpy() )

        # confusion_matrix = torch.from_numpy(confusion_matrix).to(device)
        # reduced_confusion_matrix = reduce_tensor(confusion_matrix)
        #
        # confusion_matrix = reduced_confusion_matrix.cpu().numpy()
        # pos = confusion_matrix.sum(1)
        # res = confusion_matrix.sum(0)
        # tp = np.diag(confusion_matrix)
        # IoU_array = (tp / np.maximum(1.0, pos + res - tp))
        # mean_IoU = IoU_array.mean()
            if index % 100 == 0:
                logging.info('processing: %d images' % index)
                mae = cnt_errors['mae'].avg
                mse = np.sqrt(cnt_errors['mse'].avg)
                nae = cnt_errors['nae'].avg
                msg = 'mae: {: 4.4f}, mse: {: 4.4f}, \
                       nae: {: 4.4f}, Class IoU: '.format(mae,
                                                          mse, nae)
                logging.info(msg)

        ap_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fp'].sum + 1e-20)
        ar_l = metrics_l['tp'].sum / (metrics_l['tp'].sum + metrics_l['fn'].sum + 1e-20)
        f1m_l = 2 * ap_l * ar_l / (ap_l + ar_l + 1e-20)
        ar_c_l = metrics_l['tp_c'].sum / (metrics_l['tp_c'].sum + metrics_l['fn_c'].sum + 1e-20)

        ap_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fp'].sum + 1e-20)
        ar_s = metrics_s['tp'].sum / (metrics_s['tp'].sum + metrics_s['fn'].sum + 1e-20)
        f1m_s = 2 * ap_s * ar_s / (ap_s + ar_s)
        ar_c_s = metrics_s['tp_c'].sum / (metrics_s['tp_c'].sum + metrics_s['fn_c'].sum + 1e-20)

        pre_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum + loc_100_metrics['fp_100'].sum + 1e-20)
        rec_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum + loc_100_metrics['fn_100'].sum + 1e-20)  # True pos rate
        f1_100 = 2 * (pre_100 * rec_100) / (pre_100 + rec_100 + + 1e-20)

        logging.info('-----Localization performance with box annotations-----')
        logging.info('AP_small: '+str(ap_s))
        logging.info('AR_small: '+str(ar_s))
        logging.info('F1m_small: '+str(f1m_s))
        logging.info('AR_small_category: '+str(ar_c_s))
        logging.info('    avg: '+str(ar_c_s.mean()))
        logging.info('AP_large: '+str(ap_l))
        logging.info('AR_large: '+str(ar_l))
        logging.info('F1m_large: '+str(f1m_l))
        logging.info('AR_large_category: '+str(ar_c_l))
        logging.info('    avg: '+str(ar_c_l.mean()))

        logging.info('-----Localization performance with points annotations-----')
        logging.info('avg precision_overall:{}'.format(pre_100.mean()))
        logging.info('avg recall_overall:{}'.format(rec_100.mean()))
        logging.info('avg F1_overall:{}'.format(f1_100.mean()))
        logging.info('Mean Loclization Error:{}'.format(MLE_metric.avg))


        mae = cnt_errors['mae'].avg
        mse = np.sqrt(cnt_errors['mse'].avg)
        nae = cnt_errors['nae'].avg

        logging.info('-----Counting performance-----')
        logging.info('MAE: ' + str(mae))
        logging.info('MSE: ' + str(mse))
        logging.info('NAE: ' + str(nae))
            # pred = test_dataset.multi_scale_inference(
            #             model,
            #             image,
            #             scales=config.TEST.SCALE_LIST,
            #             flip=config.TEST.FLIP_TEST)
            #

    return  mae, mse, nae

def test(config, test_dataset, testloader, model, 
        sv_dir='', sv_pred=True):
    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(tqdm(testloader)):
            image, size, name = batch
            size = size[0]
            pred = test_dataset.multi_scale_inference(
                        model, 
                        image, 
                        scales=config.TEST.SCALE_LIST, 
                        flip=config.TEST.FLIP_TEST)
            
            if pred.size()[-2] != size[0] or pred.size()[-1] != size[1]:
                pred = F.upsample(pred, (size[-2], size[-1]), 
                                   mode='bilinear')

            if sv_pred:
                sv_path = os.path.join(sv_dir,'test_results')
                if not os.path.exists(sv_path):
                    os.mkdir(sv_path)
                test_dataset.save_pred(pred, sv_path, name)
