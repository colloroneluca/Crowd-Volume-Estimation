import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.heads.head_selector import  HeadSelector
from lib.models.heads.moe import upsample_module
from lib.utils.Gaussianlayer import Gaussianlayer
from PIL import Image
import sys
import os
cwd = os.getcwd()

def save_image(tensor, filename):
    # Convert the tensor to a PIL Image
    image = Image.fromarray(tensor.numpy().astype('uint8').transpose(1, 2, 0))
    
    # Save the image
    image.save(filename)

from PIL import Image, ImageDraw, ImageFont

import torch



class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        
        # Flow branch - start with 6 input channels
       
        
        # Merged branch - start with 3 input channels

        self.cm1 = nn.Conv2d(80, 48, kernel_size=2, stride=2, padding=0) # Output: [4, 48, 256, 1792]  # TODO
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.cm2 = nn.Conv2d(48, 96, kernel_size=2, stride=2, padding=0)  # Output: [4, 96, 128, 896]
        self.cm3 = nn.Conv2d(96, 192, kernel_size=2, stride=2, padding=0) # Output: [4, 192, 64, 448]
        self.cm4 = nn.Conv2d(192, 384, kernel_size=2, stride=2, padding=0) # Output: [4, 384, 32, 224]
        self.merge = nn.ModuleList([nn.Conv2d(96, 48, kernel_size=1, stride=1, padding=0),  # TODO
                                    nn.Conv2d(192, 96, kernel_size=1, stride=1, padding=0),  # TODO
                                    nn.Conv2d(384, 192, kernel_size=1, stride=1, padding=0),  # TODO
                                    nn.Conv2d(768, 384, kernel_size=1, stride=1, padding=0)])  # TODO


    def forward(self, warped, in_list):
        # Process flow
        cm1_out = self.relu(self.mp1(self.cm1(warped)))
        cm2_out = self.relu(self.cm2(cm1_out))
        cm3_out = self.relu(self.cm3(cm2_out))
        cm4_out = self.relu(self.cm4(cm3_out))
        warped_features = [cm1_out,cm2_out,cm3_out,cm4_out]
        merged_features = []
        for i in range(len(warped_features)):
            x = torch.concat((in_list[i],warped_features[i]),dim=1)
            merged_features.append(self.relu(self.merge[i](x)))

        return merged_features

def crop_tensor(tensor, point):
    # Check if the input is of tensor type
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Input should be of tensor type")

    # Check if the tensor has only one channel
    if tensor.shape[0] != 1:
        raise ValueError("Input tensor should have only one channel")

    # Calculate the coordinates of the bounding box
    top = max(0, point[0] - 32)
    left = max(0, point[1] - 32)
    bottom = min(tensor.shape[1], point[0] + 32)
    right = min(tensor.shape[2], point[1] + 32)

    # Crop the tensor
    tensor_cropped = tensor[:, top:bottom, left:right]

    return tensor_cropped



def write_text_on_image(image, position, text):
    # Convert tensor to PIL Image
    image = Image.fromarray(image.cpu().numpy().astype('uint8').transpose(1, 2, 0))
    
    # Create ImageDraw object
    draw = ImageDraw.Draw(image)
    
    # Specify font
    #font = ImageFont.truetype('arial.ttf', 15)
    
    # Write text on image
    
    draw.text(position, text, (255,255,255))
    
    return image

def expand_and_sum(image_3d, image_2d, one_hot_image, info, i, print_images=False):
    # Check if the input images are of tensor type
    if not isinstance(image_3d, torch.Tensor) or not isinstance(image_2d, torch.Tensor) or not isinstance(one_hot_image, torch.Tensor):
        raise ValueError("All images should be of tensor type")

    # Check the shape of the images
    if len(image_3d.shape) != 3 or len(image_2d.shape) != 2 or len(one_hot_image.shape) != 2:
        raise ValueError("Image_3d should be of shape (3,d,d), image_2d and one_hot_image should be of shape (d,d)")

    # Expand the 2D image to 3D by copying it three times along a new dimension
    image_2d_expanded = image_2d.unsqueeze(0).repeat(3, 1, 1)
    un_norm_image = image_3d
    image_3d = image_3d-image_3d.min()
    image_3d= image_3d/image_3d.max()

    # Sum the two images
    result = image_3d*255 + image_2d_expanded*1000

    # Find the position from the one-hot encoded image
    position = torch.nonzero(one_hot_image != 0).cpu().numpy()

    # Write text on the image at the specified position
    if print_images:
        if len(position)>0:
            for j in range(len(position)):
                res = crop_tensor(image_2d.unsqueeze(0), position[j])

                result_ = write_text_on_image(result, (position[j][1], position[j][0]), str((res.sum()/100).item())[:6])
                result_.save(f'/storage/hdd-1/LucaC/root/STEERER/debug_images/result_{i}_{j}.png')
                #print(f'/storage/hdd-1/LucaC/root/STEERER/debug_images/saved result_{i}_{j}.png')
        else:
            # Save the resulting image
            save_image(result.cpu(), f'/storage/hdd-1/LucaC/root/STEERER/debug_images/result_{i}_empty.png')
            print(f'saved /storage/hdd-1/LucaC/root/STEERER/debug_images/result_{i}.png')

    #return result




class UncertaintyLoss(nn.Module):

    def __init__(self, v_num):
        super(UncertaintyLoss, self).__init__()
        sigma = torch.tensor([1,0.5, 0.25, 0.125])
        sigma = torch.tensor(-torch.log(2*sigma))
        self.sigma = nn.Parameter(sigma)
        self.v_num = v_num
        self.count = 0

    def forward(self, input):
        loss = 0
        # import pdb
        # pdb.set_trace()
        for i in range(self.v_num):
            loss += input[i]*0.5*torch.exp(-self.sigma[i]) #/(2 * self.sigma[i] ** 2)
        loss +=0.01* torch.exp(0.5*self.sigma).sum()  #(torch.exp(-self.sigma).sum()-2)**2 #torch.log(self.sigma.pow(2).prod())
        self.count+=1
        if self.count %100 == 0:
            print(self.sigma.data)
        return loss

def freeze_model(model):
    for (name, param) in model.named_parameters():
            param.requires_grad = False


class Baseline_Counter(nn.Module):
    def __init__(self, config=None,weight=200, route_size=(64,64),device=None):
        super(Baseline_Counter, self).__init__()
        self.config = config
        self.device =device
        self.resolution_num = config.resolution_num
        self.backbone = BackboneSelector(self.config).get_backbone()
        self.gaussian = Gaussianlayer(config.sigma, config.gau_kernel_size)
        self.gaussian_maximum=self.gaussian.gaussian.gkernel.weight.max()
        self.mse_loss = nn.MSELoss()
        if self.config.counter_type == 'withMOE':
            self.multi_counters = HeadSelector(self.config.head).get_head()
            self.counter_copy=HeadSelector(self.config.head).get_head()
            freeze_model(self.counter_copy)
            self.upsample_module = upsample_module(self.config.head)

        elif self.config.counter_type == 'single_resolution':
            self.count_head = HeadSelector(self.config.head).get_head()
        else:
            raise ValueError('COUNTER must be basleline or withMOE')
        self.weight = weight
        self.route_size = (route_size[0] //(2**self.resolution_num[0]),
                            route_size[1] //(2**self.resolution_num[0]))
        self.label_start = self.resolution_num[0]
        self.label_end = self.resolution_num[-1]+1


    def forward(self,inputs, labels=None, mode='train', info=None):
        if self.config.counter_type == 'single_resolution':
            x_list = self.backbone(inputs)
            x0_h, x0_w = x_list[0].size(2), x_list[0].size(3)
            y = [x_list[0]]
            for i in range(1, len(x_list)):
                y.append(F.upsample(x_list[i], size=(x0_h, x0_w), mode='bilinear'))
            y = torch.cat(y, 1)

            outputs = self.count_head(y)

            # used for flops calculating and model testing
            if labels is None:
                return  outputs

            labels = labels[0].unsqueeze(1)
            labels =  self.gaussian(labels)

            if mode =='train' or mode =='val':
                loss = self.mse_loss(outputs, labels*self.weight)
                gt_cnt = labels.sum().item()
                pre_cnt = outputs.sum().item()/self.weight

                result = {
                    'x4': {'gt': gt_cnt, 'error':max(0, gt_cnt-abs(gt_cnt-pre_cnt))},
                    'x8': {'gt': 0, 'error': 0},
                    'x16': {'gt': 0, 'error': 0},
                    'x32': {'gt': 0, 'error': 0},
                    'acc1': {'gt': 0, 'error': 0},
                    'losses':loss,
                    'pre_den':
                        {
                            '1':outputs/self.weight,
                         },

                    'gt_den':{'1':labels}

                }
                return  result

            elif mode == 'test':
                return outputs / self.weight


        elif self.config.counter_type == 'withMOE':
            
            result = {'pre_den':{},'gt_den':{}}
            # if labels is not None:
            #     return self.fake_result
            in_list = self.backbone(inputs)
            self.counter_copy.load_state_dict(self.multi_counters.state_dict())
            freeze_model(self.counter_copy)

            in_list = in_list[self.resolution_num[0]:self.resolution_num[-1]+1]

            out_list =self.upsample_module(in_list,self.multi_counters,self.counter_copy)
            # import pdb
            # pdb.set_trace()

            if labels is None:
                return  out_list

            label_list = []
            ohe = []

            labels = labels[self.label_start:self.label_end]

            for i, label in enumerate(labels):
                ohe.append(label)
                label_list.append(self.gaussian(label.unsqueeze(1))*self.weight)
            # for i in range(inputs.shape[0]):
            #     expand_and_sum(inputs[i],label_list[0][i].squeeze(0), ohe[0][i], 'text', i)
                


            # moe_label,score_gt = self.get_moe_label(out_list, label_list, (64,64))

            # import numpy as np
            # import cv2
            # import pdb
            # pred_color_map= moe_label.cpu().numpy()
            # np.save('./exp/moe/{}.npy'.format(moe_label.size(2)),pred_color_map)
            # pred_color_map = cv2.applyColorMap(
            #     (255 * pred_color_map / (pred_color_map.max() + 1e-10)).astype(np.uint8).squeeze(), cv2.COLORMAP_JET)
            # cv2.imwrite('./exp/moe/moe_label_{}.png'.format(moe_label.size(2)), pred_color_map)
            # pdb.set_trace()
            if mode =='patch':
                result.update({'losses':None})
                result['pre_den'].update({'1': out_list[0]/self.weight})
                result['pre_den'].update({'2': out_list[-3]/self.weight})
                result['pre_den'].update({'4': out_list[-2]/self.weight})
                result['pre_den'].update({'8': out_list[-1]/self.weight})

                result['gt_den'].update({'1': label_list[0]/self.weight})
                result['gt_den'].update({'2': label_list[-3]/self.weight})
                result['gt_den'].update({'4': label_list[-2]/self.weight})
                result['gt_den'].update({'8': label_list[-1]/self.weight})
                return result

            if mode =='val':
                result.update({'losses':self.mse_loss(out_list[0],label_list[0])})
                result['pre_den'].update({'1': out_list[0]/self.weight})
                result['pre_den'].update({'2': out_list[-3]/self.weight})
                result['pre_den'].update({'4': out_list[-2]/self.weight})
                result['pre_den'].update({'8': out_list[-1]/self.weight})

                result['gt_den'].update({'1': label_list[0]/self.weight})
                result['gt_den'].update({'2': label_list[-3]/self.weight})
                result['gt_den'].update({'4': label_list[-2]/self.weight})
                result['gt_den'].update({'8': label_list[-1]/self.weight})
                return result

            moe_label,score_gt = self.get_moe_label(out_list, label_list, self.route_size)

            mask_gt = torch.zeros_like(score_gt)

            if mode =='train' or mode =='val':
                mask_gt = mask_gt.scatter_(1,moe_label, 1)

            loss_list = []
            outputs = torch.zeros_like(out_list[0])
            label_patch = torch.zeros_like(label_list[0])

            result.update({'acc1': {'gt':0, 'error':0}})

            # import pdb
            # pdb.set_trace()
            mask_add = torch.ones_like(mask_gt[:,0].unsqueeze(1))
            for i in range(mask_gt.size(1)):

                kernel = (int(self.route_size[0] / (2 ** i)), int(self.route_size[1] / (2 ** i)))
                loss_mask = F.upsample_nearest(mask_add,   size=(out_list[i].size()[2:]))
                hard_loss=self.mse_loss(out_list[i]*loss_mask,label_list[i]*loss_mask)
                loss_list.append(hard_loss)

                # import pdb
                # pdb.set_trace()

                if i == 0:
                    label_patch += (label_list[0] * F.upsample_nearest(mask_gt[:,i].unsqueeze(1),
                                                           size=(out_list[i].size()[2:])))
                    label_patch = F.unfold(label_patch, kernel,  stride=kernel)
                    B_, _, L_ = label_patch.size()
                    label_patch = label_patch.transpose(2, 1).view(B_, L_, kernel[0], kernel[1])
                else:
                    gt_slice  = F.unfold(label_list[i], kernel,stride=kernel)
                    B, KK, L = gt_slice.size()

                    pick_gt_idx = (moe_label.flatten(start_dim=1) == i).unsqueeze(2).unsqueeze(3)
                    gt_slice = gt_slice.transpose(2,1).view(B, L,kernel[0], kernel[1])
                    pad_w, pad_h =(self.route_size[1] - kernel[1])//2, (self.route_size[0] - kernel[0])//2
                    gt_slice = F.pad(gt_slice, [pad_w,pad_w,pad_h,pad_h], "constant", 0.2)
                    gt_slice = (gt_slice * pick_gt_idx)
                    label_patch += gt_slice

                gt_cnt = (label_list[i]*loss_mask).sum().item()/self.weight
                pre_cnt = (out_list[i]*loss_mask).sum().item()/self.weight
                result.update({f"x{2**(self.resolution_num[i]+2)}": {'gt': gt_cnt,
                                            'error':max(0, gt_cnt-abs(gt_cnt-pre_cnt))}})
                mask_add -=mask_gt[:,i].unsqueeze(1)

            B_num, C_num, H_num, W_num =  out_list[0].size()
            patch_h, patch_w = H_num // self.route_size[0], W_num // self.route_size[1]
            label_patch =label_patch.view(B_num, patch_h*patch_w, -1).transpose(1,2)
            label_patch = F.fold(label_patch, output_size=(H_num, W_num), kernel_size=self.route_size, stride=self.route_size)

            if mode =='train' or mode =='val':
                loss = 0
                if self.config.baseline_loss:
                    loss = loss_list[0]
                else:
                    for i in range(len(self.resolution_num)):
                        # if self.config.loss_weight:
                        loss +=loss_list[i]*self.config.loss_weight[i]
                        # else:
                        #     loss += loss_list[i] /(2**(i))

                for i in ['x4','x8','x16', 'x32']:
                    if i not in result.keys():
                        result.update({i: {'gt': 0, 'error': 0}})
                result.update({'moe_label':moe_label})
                result.update({'losses':torch.unsqueeze(loss,0)})
                result['pre_den'].update({'1':out_list[0]/self.weight})
                result['pre_den'].update({'8': out_list[-1]/self.weight})
                result['gt_den'].update({'1': label_patch/self.weight})
                result['gt_den'].update({'8': label_list[-1]/self.weight})

                return result

            elif mode == 'test':

                return outputs / self.weight

    def get_moe_label(self, out_list, label_list, route_size):
        """
        :param out_list: (N,resolution_num,H, W) tensor
        :param in_list:  (N,resolution_num,H, W) tensor
        :param route_size: 256
        :return:
        """
        B_num, C_num, H_num, W_num =  out_list[0].size()
        patch_h, patch_w = H_num // route_size[0], W_num // route_size[1]
        errorInslice_list = []

        for i, (pre, gt) in enumerate(zip(out_list, label_list)):
            pre, gt= pre.detach(), gt.detach()
            kernel = (int(route_size[0]/(2**i)), int(route_size[1]/(2**i)))

            weight = torch.full(kernel,1/(kernel[0]*kernel[1])).expand(1,pre.size(1),-1,-1)
            weight =  nn.Parameter(data=weight, requires_grad=False).to(self.device)

            error= (pre - gt)**2
            patch_mse=F.conv2d(error, weight,stride=kernel)

            weight = torch.full(kernel,1.).expand(1,pre.size(1),-1,-1)
            weight =  nn.Parameter(data=weight, requires_grad=False).to(self.device)

            # mask = (gt>0).float()
            # mask = F.max_pool2d(mask, kernel_size=7, stride=1, padding=3)
            patch_error=F.conv2d(error, weight,stride=kernel)  #(pre-gt)*(gt>0)
            fractions = F.conv2d(gt, weight, stride=kernel)

            instance_mse = patch_error/(fractions+1e-10)

            errorInslice_list.append(patch_mse + instance_mse)


        score = torch.cat(errorInslice_list, dim=1)
        moe_label = score.argmin(dim=1, keepdim=True)


        # mask = mask.view(mask.size(0),mask.size(1),patch_h, patch_w).float()
        # import pdb
        # pdb.set_trace()

        return  moe_label, score



class Baseline_Classifier(nn.Module):
    def __init__(self, config=None):
        super(Baseline_Classifier, self).__init__()
        self.config = config
        self.backbone = BackboneSelector(self.config).get_backbone()

        # self.cls_head0 = HeadSelector(self.config.head0).get_head()
        # self.cls_head1 = HeadSelector(self.config.head1).get_head()
        # self.cls_head2 = HeadSelector(self.config.head2).get_head()
        # self.cls_head3 = HeadSelector(self.config.head3).get_head()
        # self.wrap_clshead = nn.ModuleList([HeadSelector(self.config.head0).get_head(),
        #                                  HeadSelector(self.config.head1).get_head(),
        #                                  HeadSelector(self.config.head2).get_head(),
        #                                  HeadSelector(self.config.head3).get_head()])

        self.wrap_clshead = HeadSelector(self.config.head0).get_head()
        self.counter = 0

    def forward(self,x, batch_idx=None):
        # if batch_idx is not None:
        #     seed = batch_idx % 4
        #     x_list= self.backbone(x,seed)
        #     x = self.wrap_clshead[seed](x_list[seed:])
        # import pdb
        # pdb.set_trace()
        #     return x
        # else:
        x_list= self.backbone(x)
        return self.wrap_clshead(x_list)

        # y = self.wrap_clshead[0](x_list[0:])
        # for i in range(1,4,1):
        #     x_list= self.backbone(x,i)
        #     y = y+self.wrap_clshead[i](x_list[i:])
        # return  y/4

if __name__ == "__main__":
    from mmcv import  Config
    cfg_data = Config.fromfile(
        '/mnt/petrelfs/hantao/HRNet-Semantic-Segmentation/configs/NWPU/hrformer_b.py')

    print(cfg_data)
    # import pdb
    # pdb.set_trace()
    model = Baseline_Counter(cfg_data.network)
    print(model)

##
