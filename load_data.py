import fnmatch
import math
import os
import sys
import time
from operator import itemgetter
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from darknet import Darknet

from median_pool import MedianPool2d

from utils import *

print('starting test read')
im = Image.open('data/horse.jpg').convert('RGB')
print('img read!')


class MaxDetectionScore(nn.Module):
    """MaxProbExtractor: extracts max class probability for class from YOLO output.

    Module providing the functionality necessary to extract the max class probability for one class from YOLO output.

    """

    def __init__(self, cls_id, num_cls, config):
        super(MaxDetectionScore, self).__init__()
        self.cls_id = cls_id
        self.num_cls = num_cls
        self.config = config

    def forward(self, YOLOoutput):
             
        # loop over 3 detection scales
        max_confs = dict()
        for i in range(len(YOLOoutput)):
        
            output = YOLOoutput[i]['x'] 
            
            batch = output.size(0)
            w = output.size(2) 
            h = output.size(3) 
            
            # transform the output tensor            
            output = output.view(batch, 3, 5 + self.num_cls , h * w)    
            output = output.transpose(1, 2).contiguous()                
            output = output.view(batch, 5 + self.num_cls , 3 * h * w)
            
            # perform sigmoid on object score
            output_objectness = torch.sigmoid(output[:, 4, :])
            
            # perform softmax on object classes
            normal_confs = torch.nn.Softmax(dim=1)(output[:, 5:5 + self.num_cls , :])
            
            # extarct probabilities of the class of interest (type of car)
            confs_for_class = normal_confs[:, self.cls_id, :]
            
            # SELECT ONE:
            # confs_if_object = confs_for_class 
            # confs_if_object = output_objectness         
            # confs_if_object = confs_for_class * output_objectness  
            confs_if_object = self.config.loss_target(output_objectness, confs_for_class) 
            
            # find the max probability of car
            max_conf, max_conf_idx = torch.max(confs_if_object, dim=1)          

            # save the max probability of car
            max_confs[i] = max_conf 
            
        maximum = torch.stack((max_confs[0], max_confs[1], max_confs[2]), dim=0) 
        
        return torch.max(maximum, dim=0).values # torch.Size([batch])


class NPSCalculator(nn.Module):
    """NMSCalculator: calculates the non-printability score of a patch.

    Module providing the functionality necessary to calculate the non-printability score (NMS) of an adversarial patch.

    """

    def __init__(self, printability_file, patch_size):
        super(NPSCalculator, self).__init__()
        self.printability_array = nn.Parameter(self.get_printability_array(printability_file, patch_size),requires_grad=False)

    def forward(self, adv_patch):
        # calculate euclidian distance between colors in patch and colors in printability_array 
        # square root of sum of squared difference
                    
        nps = 0
        for i in range(adv_patch.size(0)):
        
            color_dist = adv_patch[i] - self.printability_array + 0.000001
            color_dist = color_dist ** 2
            color_dist = torch.sum(color_dist, 1) + 0.000001
            color_dist = torch.sqrt(color_dist)
            # only work with the min distance
            color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
            # calculate the nps by summing over all pixels
            nps_score = torch.sum(color_dist_prod,0)
            nps_score = torch.sum(nps_score, 0)
  
            nps += nps_score
        
        return nps/torch.numel(adv_patch)

    def get_printability_array(self, printability_file, size):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full((size[0], size[1]), red))
            printability_imgs.append(np.full((size[0], size[1]), green))
            printability_imgs.append(np.full((size[0], size[1]), blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array)
        
        return pa


class TVCalculator(nn.Module):
    """TotalVariation: calculates the total variation of a patch.

    Module providing the functionality necessary to calculate the total vatiation (TV) of an adversarial patch.

    """

    def __init__(self):
        super(TVCalculator, self).__init__()

    def forward(self, adv_patch):
        
        tv = 0
        for i in range(adv_patch.size(0)):
        
            tvcomp1 = torch.sum(torch.abs(adv_patch[i, :, :, 1:] - adv_patch[i, :, :, :-1]+0.000001),0)
            tvcomp1 = torch.sum(torch.sum(tvcomp1,0),0)
            tvcomp2 = torch.sum(torch.abs(adv_patch[i, :, 1:, :] - adv_patch[i, :, :-1, :]+0.000001),0)
            tvcomp2 = torch.sum(torch.sum(tvcomp2,0),0)
 
            tv += tvcomp1 + tvcomp2
        
        return tv/torch.numel(adv_patch)


class PatchTransformations(nn.Module):
    """PatchTransformer: transforms batch of patches

    Module providing the functionality necessary to transform a batch of patches by: 
        - randomly adjusting brightness and contrast, 
        - adding random amount of noise,
        - rotating randomly, and 
        - resizing patches according to as size based on the batch of labels, 
          and pads them to the dimension of an image.

    """

    def __init__(self):
        super(PatchTransformations, self).__init__()

        # EOT
        self.min_contrast = 0.70
        self.max_contrast = 1.30
        self.min_brightness = -0.60
        self.max_brightness = 0.60
        self.noise_factor = 0.10

        # EOT+Weather
        # self.min_contrast = 0.50
        # self.max_contrast = 0.90
        # self.min_brightness = 0.30
        # self.max_brightness = 0.60
        # self.noise_factor = 0.10
              
        self.minangle = -20 / 180 * math.pi # -20 degrees to radians
        self.maxangle = 20 / 180 * math.pi  # 20 degrees to radians
        self.medianpooler = MedianPool2d(7,same=True)

    def forward(self, adv_patch, lab_batch, img_size, size, do_rotate=True, rand_loc=True):
         
        # Apply medium pooling on patch
        adv_patch = self.medianpooler(adv_patch) 
    
        # Make a batch of patches
        adv_patch = adv_patch.unsqueeze(0).unsqueeze(0) 
        adv_batch = adv_patch.expand(lab_batch.size(0), lab_batch.size(1), -1, -1, -1, -1) 
    
        # Define batch 
        batch_size = torch.Size((lab_batch.size(0), lab_batch.size(1)))

        #######################################################################
        ### Contrast, brightness and noise transforms ###
        #######################################################################
        # Create random contrast tensor
        contrast = torch.cuda.FloatTensor(batch_size).uniform_(self.min_contrast, self.max_contrast) 
        contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        contrast = contrast.expand(-1, -1, adv_batch.size(-4), adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1)) 
        contrast = contrast.cuda()        

        # Create random brightness tensor           
        brightness = torch.cuda.FloatTensor(batch_size).uniform_(self.min_brightness, self.max_brightness) 
        brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) 
        brightness = brightness.expand(-1, -1, adv_batch.size(-4), adv_batch.size(-3), adv_batch.size(-2), adv_batch.size(-1)) 
        brightness = brightness.cuda()
        
        # Create random noise tensor
        noise = torch.cuda.FloatTensor(adv_batch.size()).uniform_(-1, 1) * self.noise_factor

        # Apply contrast, brightness and noise to patch
        adv_batch = adv_batch                                    # NONE
        # adv_batch = (adv_batch * contrast) + brightness + noise  # EOT

        # Clamp batch of patches
        adv_batch = torch.clamp(adv_batch, 0.000001, 0.99999)
        
        #######################################################################
        # Number of patches per image
        #######################################################################           
        cls_ids = torch.narrow(lab_batch, 2, 0, 1) 
        cls_mask = cls_ids.expand(-1, -1, adv_batch.size(-4)) 
        cls_mask = cls_mask.unsqueeze(-1) 
        cls_mask = cls_mask.expand(-1, -1, -1, adv_batch.size(-3)) 
        cls_mask = cls_mask.unsqueeze(-1) 
        cls_mask = cls_mask.expand(-1, -1, -1, -1, adv_batch.size(-2)) 
        cls_mask = cls_mask.unsqueeze(-1) 
        cls_mask = cls_mask.expand(-1, -1, -1, -1, -1, adv_batch.size(-1)) 

        msk_batch = torch.clamp(cls_mask, 0.0, 1.0)
        
        # Size of padding
        pad_w = (img_size - adv_patch.size(-1)) / 2
        pad_h = (img_size - adv_patch.size(-2)) / 2 
        
        # Pad patch and mask to image dimensions      
        mypad = nn.ConstantPad2d((int(pad_w + 0.5), int(pad_w), int(pad_h + 0.5), int(pad_h)), 0)
        adv_batch = mypad(adv_batch) 
        msk_batch = mypad(msk_batch) 

        # NOTE: patch and mask are located at the centre of the image

        #######################################################################
        ### Rotation and rescaling transforms ###
        #######################################################################

        # Define patch size
        current_patch_size = adv_patch.size(-1) # set in patch_config.py
        
        # Scale the labels according to input image size
        lab_batch_scaled = torch.cuda.FloatTensor(lab_batch.size()).fill_(0)
        lab_batch_scaled[:, :, 1] = lab_batch[:, :, 1] * img_size # x of centre
        lab_batch_scaled[:, :, 2] = lab_batch[:, :, 2] * img_size # y of centre
        lab_batch_scaled[:, :, 3] = lab_batch[:, :, 3] * img_size # width of bounding box
        lab_batch_scaled[:, :, 4] = lab_batch[:, :, 4] * img_size # height of bounding box
        
        #######################################################################

        # Size of patch relative to width and height of bounding box
        width_size = torch.sqrt(((lab_batch_scaled[:, :, 3].mul(0.2)) ** 2) + ((lab_batch_scaled[:, :, 4].mul(0.2)) ** 2))       
        
        # Size of patch is fixed
        width_size[width_size > 0] = size
          
        #######################################################################   
        #######################################################################
        
        # Define lists
        adv_list = []
        msk_list = []
        adv_msk_list = []
        
        offset_x = 0
        offset_y = 0
        for i in range(adv_batch.size(2)):

            if adv_batch.size(2) == 1:
                
                # Define angle - ON car
                anglesize = (lab_batch.size(0) * lab_batch.size(1))
                if do_rotate:
                    angle = torch.cuda.FloatTensor(anglesize).uniform_(self.minangle, self.maxangle)
                else: 
                    angle = torch.cuda.FloatTensor(anglesize).fill_(0)
                
                "Place patch (ON CAR) on each car detected relative to the centre of bounding box"
                target_x = lab_batch[:, :, 1].view(np.prod(batch_size)) # x centre of bounding box
                target_y = lab_batch[:, :, 2].view(np.prod(batch_size)) # y centre of bounding box
                targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size)) # width of bounding box
                targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size)) # height of bounding box
      
                # Random offsets from centre of bounding boxes
                if(rand_loc):
                    offset_x = targetoff_x*(torch.cuda.FloatTensor(targetoff_x.size()).uniform_(-0.4,0.4))
                    target_x = target_x + offset_x
                    
                    offset_y = targetoff_y*(torch.cuda.FloatTensor(targetoff_y.size()).uniform_(-0.4,0.4))
                    target_y = target_y + offset_y
            
            elif adv_batch.size(2) == 2:
                
                # Define angle - OFF car
                anglesize = (lab_batch.size(0) * lab_batch.size(1))
                if i == 2:
                    angle = torch.cuda.FloatTensor(anglesize).uniform_(90/180*math.pi, 90/180*math.pi)
                elif i == 0 or 1: 
                    angle = torch.cuda.FloatTensor(anglesize).fill_(0)
                
                "Place 2 patches (OFF CAR) on each car detected relative to the centre of bounding box"
                # Place patch on each car detected relative to the centre of bounding box
                target_x = lab_batch[:, :, 1].view(np.prod(batch_size)) # x centre of bounding box
                target_y = lab_batch[:, :, 2].view(np.prod(batch_size)) # y centre of bounding box
                targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size)) # width of bounding box
                targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size)) # height of bounding box

                if i == 0:
                    # target_x = target_x + (10/256) # x-right
                    target_y = target_y + (15/256) - torch.randint(0, 3, (1,)).to(device='cuda', dtype=torch.float)/256 # y-bottom
                elif i == 1:
                    # target_x = target_x + (10/256) # x-right
                    target_y = target_y - (15/256) + torch.randint(0, 3, (1,)).to(device='cuda', dtype=torch.float)/256 # y-top

            elif adv_batch.size(2) == 3:
                
                # Define angle - OFF car
                anglesize = (lab_batch.size(0) * lab_batch.size(1))
                if i == 2:
                    angle = torch.cuda.FloatTensor(anglesize).uniform_(90/180*math.pi, 90/180*math.pi)
                elif i == 0 or 1: 
                    angle = torch.cuda.FloatTensor(anglesize).fill_(0)

                "Place 3 patches (OFF CAR) on each car detected relative to the centre of bounding box"
                # Place patch on each car detected relative to the centre of bounding box
                target_x = lab_batch[:, :, 1].view(np.prod(batch_size)) # x centre of bounding box
                target_y = lab_batch[:, :, 2].view(np.prod(batch_size)) # y centre of bounding box
                targetoff_x = lab_batch[:, :, 3].view(np.prod(batch_size)) # width of bounding box
                targetoff_y = lab_batch[:, :, 4].view(np.prod(batch_size)) # height of bounding box

                if i == 0:
                    target_y = target_y + (15/256) - torch.randint(0, 3, (1,)).to(device='cuda', dtype=torch.float)/256 # y-bottom
                elif i == 1:
                    target_y = target_y - (15/256) + torch.randint(0, 3, (1,)).to(device='cuda', dtype=torch.float)/256 # y-top 
                elif i == 2:    
                    target_x = target_x + (22/256) + torch.randint(0, 2, (1,)).to(device='cuda', dtype=torch.float)/256 # x-right
                    
                    # flip = random.randint(0,1) 
                    # if flip == 0:
                    #     target_x = target_x + (22/256) + torch.randint(0, 2, (1,)).to(device='cuda', dtype=torch.float)/256 # x-right
                    # elif flip == 1:
                    #     target_x = target_x - (22/256) - torch.randint(0, 2, (1,)).to(device='cuda', dtype=torch.float)/256 # x-left
            
            # Scaling factor
            scale = width_size / current_patch_size
            scale = scale.view(anglesize) 
    
            # Reshape batches
            s = adv_batch[:,:,i,:,:,:].size() # torch.Size([4, 10, 3, 256, 256])
            adv_batch_temp = adv_batch[:,:,i,:,:,:].view(s[0] * s[1], s[2], s[3], s[4]) 
            msk_batch_temp = msk_batch[:,:,i,:,:,:].view(s[0] * s[1], s[2], s[3], s[4]) 

            # Define terms of theta
            tx = (-target_x+0.5)*2 
            ty = (-target_y+0.5)*2 
            sin = torch.sin(angle) 
            cos = torch.cos(angle) 
            
            # Theta = rotation, rescale matrix
            theta = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
            theta[:, 0, 0] = cos/(scale)
            theta[:, 0, 1] = sin/scale
            theta[:, 0, 2] = tx*cos/scale+ty*sin/scale
            theta[:, 1, 0] = -sin/scale
            theta[:, 1, 1] = cos/(scale)
            theta[:, 0, 1] = sin/scale
            theta[:, 1, 2] = -tx*sin/scale+ty*cos/scale
    
            # Transform batches
            grid = F.affine_grid(theta, adv_batch_temp.shape) 
            adv_batch_t = F.grid_sample(adv_batch_temp, grid) 
            msk_batch_t = F.grid_sample(msk_batch_temp, grid)
    
            '''
            # Theta2 = translation matrix
            theta2 = torch.cuda.FloatTensor(anglesize, 2, 3).fill_(0)
            theta2[:, 0, 0] = 1
            theta2[:, 0, 1] = 0
            theta2[:, 0, 2] = (-target_x + 0.5) * 2
            theta2[:, 1, 0] = 0
            theta2[:, 1, 1] = 1
            theta2[:, 1, 2] = (-target_y + 0.5) * 2
    
            grid2 = F.affine_grid(theta2, adv_batch.shape)
            adv_batch_t = F.grid_sample(adv_batch_t, grid2)
            msk_batch_t = F.grid_sample(msk_batch_t, grid2)
    
            '''
            
            # Reshape batch of patches and masks
            adv_batch_t = adv_batch_t.view(s[0], s[1], s[2], s[3], s[4]) 
            msk_batch_t = msk_batch_t.view(s[0], s[1], s[2], s[3], s[4]) 
    
            # Clip batch of patches
            adv_batch_t = torch.clamp(adv_batch_t, 0.000001, 0.999999)
            
            # Update X offset
            # offset_x += (size+3)/256
            
            # Store patches and masks of each piece of the patch
            adv_list.append(adv_batch_t)
            msk_list.append(msk_batch_t)
            adv_msk_list.append(adv_batch_t * msk_batch_t)

        return adv_msk_list


class PatchApplier(nn.Module):
    """PatchApplier: applies adversarial patches to images.

    Module providing the functionality necessary to apply a patch to all detections in all images in the batch.

    """

    def __init__(self):
        super(PatchApplier, self).__init__()

    def forward(self, img_batch, adv_batch):
    
        for i in range(len(adv_batch)):
        
            advs = torch.unbind(adv_batch[i], 1)
            
            for adv in advs:
                img_batch = torch.where((adv == 0), img_batch, adv)
                          
        return img_batch

class LoadDataset(Dataset):
    """InriaDataset: representation of the INRIA person dataset.

    Internal representation of the commonly used INRIA person dataset.
    Available at: http://pascal.inrialpes.fr/data/human/

    Attributes:
        len: An integer number of elements in the
        img_dir: Directory containing the images of the INRIA dataset.
        lab_dir: Directory containing the labels of the INRIA dataset.
        img_names: List of all image file names in img_dir.
        shuffle: Whether or not to shuffle the dataset.

    """

    def __init__(self, img_dir, lab_dir, max_lab, imgsize, shuffle=True):
        n_png_images = len(fnmatch.filter(os.listdir(img_dir), '*.png'))
        n_jpg_images = len(fnmatch.filter(os.listdir(img_dir), '*.jpg'))
        n_images = n_png_images + n_jpg_images
        n_labels = len(fnmatch.filter(os.listdir(lab_dir), '*.txt'))
        assert n_images == n_labels, "Number of images and number of labels don't match"
        self.len = n_images
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.imgsize = imgsize
        self.img_names = fnmatch.filter(os.listdir(img_dir), '*.png') + fnmatch.filter(os.listdir(img_dir), '*.jpg')
        self.shuffle = shuffle
        self.img_paths = []
        for img_name in self.img_names:
            self.img_paths.append(os.path.join(self.img_dir, img_name))
        self.lab_paths = []
        for img_name in self.img_names:
            lab_path = os.path.join(self.lab_dir, img_name).replace('.jpg', '.txt').replace('.png', '.txt')
            self.lab_paths.append(lab_path)
        self.max_n_labels = max_lab

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        lab_path = os.path.join(self.lab_dir, self.img_names[idx]).replace('.jpg', '.txt').replace('.png', '.txt')
        image = Image.open(img_path).convert('RGB')
        
        if os.path.getsize(lab_path):       #check to see if label file contains data. 
            label = np.loadtxt(lab_path)
        else:
            label = np.ones([5])

        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)

        image, label = self.pad_and_scale(image, label)
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.pad_lab(label)
        
        return image, label

    def pad_and_scale(self, img, lab):
        """

        Args:
            img:

        Returns:

        """
        w,h = img.size
        if w==h:
            padded_img = img
        else:
            dim_to_pad = 1 if w<h else 2
            if dim_to_pad == 1:
                padding = (h - w) / 2
                padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                padded_img.paste(img, (int(padding), 0))
                lab[:, [1]] = (lab[:, [1]] * w + padding) / h
                lab[:, [3]] = (lab[:, [3]] * w / h)
            else:
                padding = (w - h) / 2
                padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                padded_img.paste(img, (0, int(padding)))
                lab[:, [2]] = (lab[:, [2]] * h + padding) / w
                lab[:, [4]] = (lab[:, [4]] * h  / w)
        resize = transforms.Resize((self.imgsize,self.imgsize))
        padded_img = resize(padded_img)     #choose here
        return padded_img, lab

    def pad_lab(self, lab):
        pad_size = self.max_n_labels - lab.shape[0]
        if(pad_size>0):
            padded_lab = F.pad(lab, (0, 0, 0, pad_size), value=0)
        else:
            padded_lab = lab
            
        return padded_lab
