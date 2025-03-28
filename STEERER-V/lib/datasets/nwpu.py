# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
from tqdm import tqdm
import cv2
import numpy as np
import heapq
from PIL import Image
import json
import torch
from torch.nn import functional as F
import random
from .base_dataset import BaseDataset
import wandb


class NWPU(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=1,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=2048,
                 crop_size=(512, 1024),
                 min_unit = (32,32),
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=(0.5,1/0.5),
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 scale=1, testing=False, head=False, phase='train'
                 ):

        self.testing = testing

        self.phase = phase
        super(NWPU, self).__init__(ignore_label, base_size,
                                   crop_size, downsample_rate, scale_factor, mean, std, scale)

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.class_weights = torch.FloatTensor([1]).cuda()

        self.multi_scale = multi_scale
        self.flip = flip
        self.scale_factor =scale_factor
        self.scale = scale
        a = np.arange(scale_factor[0],1.,0.05)
        # b = np.linspace(1, scale_factor[1],a.shape[0])

        self.scale_factor = np.concatenate([a,1/a],0)

        self.img_list = [line.strip().split() for line in open(os.path.join(root,'txts' , list_path))] 
        self.img_list_clean = []
        self.blacklist = []
        for img in self.img_list: 
            f_number = int(img[0].split('/')[-1].split('.')[0])
            if (f_number < 850 and f_number%2==0) or 'test' in self.phase:
                self.img_list_clean.append(img)
        self.img_list = self.img_list_clean

        self.box_gt = []
        self.min_unit = min_unit

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]



        self.resize_memory_pool = {}
        self.AI_resize = False
        
    def read_files(self):

        files = []
        if 'test'in self.list_path:
            for item in self.img_list:
                image_id, _, __ = item
                files.append({
                    "img": 'images/' + image_id + '.jpg',
                    "label": 'jsons/' + image_id + '.json',
                    "name": image_id,
                })
        else:
            for item in self.img_list:

                image_id, _, __ = item
                files.append({
                    "img": 'images/' + image_id + '.jpg',
                    "label": 'jsons/' + image_id + '.json',
                    "name": image_id
                })
        return files

    def read_box_gt(self, box_gt_file):
        gt_data = {}
        # with open(box_gt_file) as f:
        #      for line in f.readlines():
        #         line = line.strip().split(' ')

        #         line_data = [int(i) for i in line]
        #         idx, num = [line_data[0], line_data[1]]
        #         points_r = []
        #         if num > 0:
        #             points_r = np.array(line_data[2:]).reshape(((len(line) - 2) // 5, 5))
        #             gt_data[idx] = {'num': num, 'points': points_r[:, 0:2], 'sigma': points_r[:, 2:4], 'level': points_r[:, 4]}
        #         else:
        #             gt_data[idx] = {'num': 0, 'points': [], 'sigma': [], 'level': []}

        return gt_data
    
    def process_path(self,path):
        # Splitting the path to extract the number
        parts = path.split('/')
        new_parts = []

        # Loop through the parts of the path
        for part in parts:
            # If the part has digits, we assume it's the one to convert
            if any(char.isdigit() for char in part):
                # Extract the digits, convert to int and back to string
                number_str = ''.join(filter(str.isdigit, part))
                number_int = int(number_str)
                number_str = str(number_int)
                # Replace the original part with the new string
                new_part = number_str +'.json'
                new_parts.append(new_part)
            else:
                new_parts.append(part)

        # Reassemble the path
        new_path = '/'.join(new_parts)
        return new_path


    def get_json_path(self, image_path):
    # Split the image path into directory and filename
        dir_path, filename = os.path.split(image_path)
        
        # Remove the 'img1' part from the directory path
        dir_path = os.path.dirname(dir_path)
        
        # Remove the leading zeros from the image filename and change the extension
        file_id = filename.split('.')[0].lstrip('0')
        json_filename = f"{file_id}.json"
        
        # Join the directory path and the new filename
        json_path = os.path.join(dir_path, json_filename)
        
        return json_path
    
    def replace_frame_number(self, path, new_num1, new_num2):
        # Convert the numbers to zero-padded strings with 6 figures
        new_num1_str = str(new_num1).zfill(6)
        new_num2_str = str(new_num2).zfill(6)
        
        # Extract directory and file name
        directory, filename = os.path.split(path)
        
        # Split the filename into the name part and the extension
        name, ext = os.path.splitext(filename)
        
        # Replace the old frame number with the new ones
        new_name1 = f"{name[:-6]}{new_num1_str}{ext}"
        new_name2 = f"{name[:-6]}{new_num2_str}{ext}"
        
        # Create the full paths for the new filenames
        new_path1 = os.path.join(directory, new_name1)
        new_path2 = os.path.join(directory, new_name2)
        
        return new_path1, new_path2

    def __getitem__(self, index):
        self.scale=1000
        item = self.files[index]
        name = item["name"]
        
        path_i1 = name

        i1 = cv2.imread(os.path.join(self.root,'images',path_i1), cv2.IMREAD_COLOR)
        size = i1.shape

        with open(os.path.join(self.root,item['label']), 'r') as f:
            info = json.load(f)

        points = np.array(info['points']).astype('float32')

        if  len(points)>0:
            idx_in_joint = np.array(info['out_frame_per_joint'])<1
            idx_not_occ_joint = np.array(info['occlusion_per_joint_level'])<0.4
            idx_not_occ_joint = idx_not_occ_joint.any(axis=1)[...,None]
            mask = idx_not_occ_joint & idx_in_joint
            points = points[mask]
            volumes = points[:,2:]
            points = points[:,:2]
        else:
            volumes = None
            points = points.reshape(-1,2)
        if self.base_size is not None:
            i1,points,ratio = self.image_points_resize(i1, self.base_size,points)
        else:
            ratio = 1.
        if 'test' in self.list_path :
            
            image=i1
            image =self.check_img(image,32)
            image = self.input_transform(image)
            label = self.label_transform(points, image.shape[:2], volumes, scale = self.scale)
            image = image.transpose((2, 0, 1))

            return image.copy(), label, ratio, name, info['human_num']


        image, label,idx,resize_factor = self.gen_sample(i1, points,name,
                                       self.multi_scale, self.flip, volumes, scale =self.scale
                                       )
        
       
        return image.copy(), label, np.array(size), [name, idx, resize_factor], info['human_num']

    def check_img(self, image,divisor ):
        h, w = image.shape[:2]
        if h % divisor != 0:
            real_h = h+(divisor - h % divisor)
        else:
            real_h = h
        if w % divisor != 0:
            real_w = w+(divisor - w % divisor)
        else:
            real_w = 0
        image = self.pad_image(image, size=(real_h, real_w), h=h,w=w, padvalue= (0.0, 0.0, 0.0))
        return image

    def gen_sample(self, image, points, name, multi_scale=True, is_flip=True, volumes=None, scale=1):

        if multi_scale:
            scale_f = random.choice(self.scale_factor)
            if not self.AI_resize:
                image_list, points, index = self.crop_then_scale(image, points, scale_f,self.crop_size)
                idx =[-1]
                resize_factor= [1]
            else:
                image, points, idx,resize_factor = self.AI_tcrop_then_scale(image,points, name)
        
        image = image_list[0]
        image = self.input_transform(image)
        label = self.label_transform(points, image.shape[:2], volumes, index, scale = self.scale)
        image = image.transpose((2, 0, 1))

        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            for i in range(len(label)):
                label[i] = label[i][:, ::flip].copy()

        return image, label, idx, resize_factor
    
    def AI_tcrop_then_scale(self,image, points, name):
        # th, tw = self.crop_size[0], self.crop_size[1]

        # print(len(self.resize_memory_pool.keys()),self.resize_memory_pool.keys())
        avg_size = self.resize_memory_pool[name]['avg_size'].copy()
        load_num = self.resize_memory_pool[name]['load_num'].copy()
        ph,pw = self.min_unit[0],self.min_unit[1]
        th, tw = self.crop_size[0], self.crop_size[1]
        topk = (th//ph)*(tw//pw)
        load_ = load_num.flatten()
        actual_topk = min(len(load_),topk)
        idx = np.argpartition(-load_,-actual_topk)[-actual_topk:]
        # import pdb
        # pdb.set_trace()
        size_factor= avg_size.flatten()[idx]

        image = self.check_img(image,self.min_unit[0])
        new_image = np.zeros((th,tw,3),dtype=np.float32)
        new_points = []
        resize_factor = np.zeros(topk,dtype=np.float32)
        for i in range(len(idx)):
            y = idx[i]//avg_size.shape[1]
            x = idx[i]%avg_size.shape[1]
            s_h, s_w = y*ph,  x*ph
            image_patch = image[s_h:s_h + ph, s_w:s_w + pw].copy()
            index = (points[:,0]>=s_w) & (points[:,0]<s_w + pw) & (points[:,1]>=s_h) & (points[:,1]<s_h + ph)
            points_patch = points[index].reshape(-1,2).copy()
            points_patch -= np.array([s_w, s_h]).astype('float32')

            a = np.linspace(1/2**size_factor[i],1,30)
            b = np.linspace(1,2**(self.num_classes-size_factor[i]),30)
            tmp_factor= random.choice(np.concatenate([a,b],0))

            tmp_factor = random.choice([tmp_factor, 1.])
            resize_factor[i]=tmp_factor

            image_patch,points_patch,idx = self.crop_then_scale(image_patch, points_patch, 1,(ph,pw))
            y = i//(tw//pw)
            x = i%(tw//pw)
            new_image[y*ph:y*ph +ph, x*pw:x*pw +pw,:] = image_patch
            new_points.append(points_patch+np.array([x*pw,y*ph]).astype('float32'))
            # import pdb
            # pdb.set_trace()
        new_points = np.concatenate(new_points,0)

        # print(np.pad(idx,((0,topk-len(idx))),constant_values=-1))
        # print(resize_factor)
        idx = np.pad(idx,((0,topk-len(idx))),constant_values=-1) # make sure idx and resize_factor has the same shape on each sample
        assert idx.shape == resize_factor.shape
        return new_image, new_points,idx, resize_factor


    def crop_then_scale(self,image, points, scale_factor, crop_size):
        # th, tw = self.crop_size[0], self.crop_size[1]
        
        i1=image
        th, tw = int(round(crop_size[0]/scale_factor)), int(round(crop_size[1]/scale_factor))
        h, w = i1.shape[:2]
        
        i1 = self.pad_image(image, h, w, (th, tw), (0.0, 0.0, 0.0))

        h, w = i1.shape[:2]

        x = random.randint(0, w - tw)
        y = random.randint(0, h - th)

        imgs = [i1]
        processed_imgs = []
        for n,img in enumerate(imgs):
            img = img[y:y + th, x:x + tw]
            img = cv2.resize(img, (crop_size[1], crop_size[0]),
                            interpolation=cv2.INTER_LINEAR)  # width, height
            if n==len(imgs)-1:
                index = (points[:,0]>=x) & (points[:,0]<x + tw) & (points[:,1]>=y) & (points[:,1]<y + th)
                points = points[index].reshape(-1,2).copy()
                points -= np.array([x, y]).astype('float32')
                points *= scale_factor
            processed_imgs.append(img)
        return [processed_imgs[0]], points, index

    def label_transform(self, points, shape, volumes=None, index=None, scale=1):
        #LC: here we need to put the volume instead of the "1"
        label = np.zeros(shape).astype('float32')
        labelx2= np.zeros((shape[0]//2,shape[1]//2) ).astype('float32')
        labelx4 = np.zeros((shape[0]//4,shape[1]//4) ).astype('float32')
        labelx8 = np.zeros((shape[0]//8,shape[1]//8) ).astype('float32')
        
        # index = np.round(points).astype('int32')
        # index[:, 0]=np.clip(index[:, 0], 0,  shape[1]-1)
        # index[:, 1] = np.clip(index[:, 1], 0, shape[0]-1)
        if index is not None:
            points_indices = []
            i=0
            for ind in index:
                if ind:
                    points_indices.append(i)
                i+=1
        
        elif volumes is not None:
            points_indices = [i for i in range(len(volumes))]
            
        all_volumes =  []
        for i in range(points.shape[0]):
            if volumes is not None:
                add = volumes[points_indices[i]]*scale
                if add<0:
                    #print('add < 0', add)
                    add=0
                all_volumes.append([points[i], add])
            else:
                add = 1
            point = points[i]
            w_idx = np.clip(int(point[0].round()), 0, shape[1] - 1)
            h_idx = np.clip(int(point[1].round()), 0, shape[0] - 1)
            
            label [h_idx,w_idx] += add

            w_idx = np.clip(int((point[0]/2).round()), 0, shape[1]//2 - 1)
            h_idx = np.clip(int((point[1]/2).round()), 0, shape[0]//2 - 1)
            labelx2[h_idx, w_idx] += add

            w_idx = np.clip(int((point[0]/4).round()), 0, shape[1]//4 - 1)
            h_idx = np.clip(int((point[1]/4).round()), 0, shape[0]//4 - 1)
            labelx4[h_idx, w_idx] += add

            w_idx = np.clip(int((point[0]/8).round()), 0, shape[1]//8 - 1)
            h_idx = np.clip(int((point[1]/8).round()), 0, shape[0]//8 - 1)
            labelx8[h_idx, w_idx] += add


        # import pdb
        # pdb.set_trace()
        #s = torch.tensor(label).sum()
        #print('\n s', s)
        return [label, labelx2, labelx4, labelx8]

    def image_resize(self, image, long_size, label=None):
        h, w = image.shape[:2]
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
        else:
            return image

        return image, label
    def multi_scale_aug(self, image, label=None,
            rand_scale=1, rand_crop=True):
        long_size = np.int(self.base_size * rand_scale + 0.5)
        if label is not None:
            image, label = self.image_resize(image, long_size, label)
            if rand_crop:
                image, label = self.rand_crop(image, label)
            return image, label
        else:
            image = self.image_resize(image, long_size)
            return image
    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i] + '.png'))
    def inference(self, model, image, flip=False):
        size = image.size()
        pred = model.val(image)
        pred = F.upsample(input=pred,
                            size=(size[-2], size[-1]),
                            mode='bilinear')
        if flip:
            flip_img = image.numpy()[:,:,:,::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))
            flip_output = F.upsample(input=flip_output,
                            size=(size[-2], size[-1]),
                            mode='bilinear')
            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(flip_pred[:,:,:,::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()

    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        import pdb
        pdb.set_trace()
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        device = torch.device("cuda:%d" % model.device_ids[0])
        image = image.numpy()[0].transpose((1, 2, 0)).copy()
        stride_h = np.int(self.crop_size[0] * 2.0 / 3.0)
        stride_w = np.int(self.crop_size[1] * 2.0 / 3.0)
        final_pred = torch.zeros([1, self.num_classes,
                                  ori_height, ori_width]).to(device)
        padvalue = -1.0 * np.array(self.mean) / np.array(self.std)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]

            if max(height, width) <= np.min(self.crop_size):
                new_img = self.pad_image(new_img, height, width,
                                         self.crop_size, padvalue)
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                if height < self.crop_size[0] or width < self.crop_size[1]:
                    new_img = self.pad_image(new_img, height, width,
                                             self.crop_size, padvalue)
                new_h, new_w = new_img.shape[:-1]
                rows = np.int(np.ceil(1.0 * (new_h -
                                             self.crop_size[0]) / stride_h)) + 1
                cols = np.int(np.ceil(1.0 * (new_w -
                                             self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                     new_h, new_w]).to(device)
                count = torch.zeros([1, 1, new_h, new_w]).to(device)

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        if h1 == new_h or w1 == new_w:
                            crop_img = self.pad_image(crop_img,
                                                      h1 - h0,
                                                      w1 - w0,
                                                      self.crop_size,
                                                      padvalue)
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)

                        preds[:, :, h0:h1, w0:w1] += pred[:, :, 0:h1 - h0, 0:w1 - w0]
                        count[:, :, h0:h1, w0:w1] += 1
                preds = preds / count
                preds = preds[:, :, :height, :width]
            preds = F.upsample(preds, (ori_height, ori_width),
                               mode='bilinear')
            final_pred += preds
        return final_pred

def patch_forward(model, img, gt_map, num_patches):
    # crop the img and gt_map with a max stride on x and y axis
    # size: HW: __C_NWPU.TRAIN_SIZE
    # stack them with a the batchsize: __C_NWPU.TRAIN_BATCH_SIZE
    crop_imgs, crop_dots, crop_masks = [], [], []


    b, c, h, w = img.shape
    rh, rw = 768, 1024
    gt_map = gt_map.unsqueeze(0)

    for i in range(0, h, rh):
        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
        for j in range(0, w, rw):
            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)
            crop_imgs.append(img[:, :, gis:gie, gjs:gje])
            crop_dots.append(gt_map[:, :, gis:gie, gjs:gje])
            mask = torch.zeros_like(gt_map).cpu()
            mask[:, :, gis:gie, gjs:gje].fill_(1.0)
            crop_masks.append(mask)
    crop_imgs, crop_dots, crop_masks = map(lambda x: torch.cat(x, dim=0), (crop_imgs, crop_dots, crop_masks))

    # forward may need repeatng
    crop_preds, crop_labels =  [], []
    nz, bz = crop_imgs.size(0), num_patches
    for i in range(0, nz, bz):
        gs, gt = i, min(nz, i + bz)

        crop_pred, crop_label = model.val(crop_imgs[gs:gt], crop_dots[gs:gt].squeeze_(1))
        crop_pred = crop_pred.cpu()
        crop_label = crop_label.cpu()
        crop_preds.append(crop_pred)
        crop_labels.append(crop_label)


    crop_preds = torch.cat(crop_preds, dim=0)
    crop_labels = torch.cat(crop_labels, dim=0)
    # splice them to the original size
    idx = 0
    pred_map = torch.zeros_like(gt_map).squeeze_(0).cpu().float()
    labels = torch.zeros_like(gt_map).squeeze_(0).cpu().float()
    for i in range(0, h, rh):
        gis, gie = max(min(h - rh, i), 0), min(h, i + rh)
        for j in range(0, w, rw):
            gjs, gje = max(min(w - rw, j), 0), min(w, j + rw)

            # print(pred_map[:, gis:gie, gjs:gje].shape)
            # print(crop_preds[idx].shape)
            pred_map[:, gis:gie, gjs:gje] += crop_preds[idx]
            labels[:, gis:gie, gjs:gje] += crop_labels[idx]
            idx += 1

    # for the overlapping area, compute average value
    mask = crop_masks.sum(dim=0).unsqueeze(0)
    pred_map = (pred_map / mask)
    labels = (labels / mask)
    return pred_map, labels