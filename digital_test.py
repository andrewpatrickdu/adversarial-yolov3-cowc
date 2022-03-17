#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:53:37 2022

A script that saves the objectness score of cars detected into json files. These
files are used to calculate AORR in aorr.py script. 

@author: andrew
"""

import pickle
import sys
import time
import os
import torch
torch.cuda.set_device(0) # select gpu to run on

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from utils import *
from darknet import *
from load_data import PatchTransformations, PatchApplier, LoadDataset
import json
    
import random
import weather

# Set random seed for reproducibility
torch.backends.cudnn.deterministic = True
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

###############################################################################
############################### ATTACK SETTINGS ###############################
###############################################################################
savedir = "digital_test"
scene = 'sidestreet' # sidestreet, carpark
mode = 'adversarial'  # clean, adversarial, random
weather_augmentations = 'on'  # on, off
setting = 'test' # test, train

# select a patch to evaluate
patch_name = 'saved_patches/sidestreet-on-gc'     # exp12
# patch_name = 'saved_patches/sidestreet-on-gcw'    # exp11
# patch_name = 'saved_patches/sidestreet-off-gc'    # exp6C
# patch_name = 'saved_patches/carpark-on-gc'        # exp04
# patch_name = 'saved_patches/carpark-on-gcw'       # exp03

patch_num = 1 # 1 for ON patches, 3 for OFF patches

###############################################################################
###############################################################################
###############################################################################
# model parameters
cfgfile = "cfg/yolov3-cowc.cfg"
weightfile = "weights/yolov3-cowc-256/yolov3-cowc_best_256.weights"
namesfile = "data/cowc.names"

# load model
darknet_model = Darknet(cfgfile)
darknet_model.load_weights(weightfile)
darknet_model = darknet_model.eval().cuda()

# patch functions
patch_applier = PatchApplier().cuda()
patch_transformations = PatchTransformations().cuda()

# define model parameters
batch_size = 1
max_lab = 10
img_size = darknet_model.height

# load patch 
patch_list = []
for i in range(patch_num):
    patchfile = f"{patch_name}/patch_{i}.jpg" 
    patch_img = Image.open(patchfile).convert('RGB')
    tf = transforms.ToTensor()
    adv_patch_cpu = tf(patch_img)
    adv_patch = adv_patch_cpu.cuda()
    adv_patch = adv_patch.unsqueeze(0)
    patch_list.append(adv_patch)

if patch_num == 1:
    adv_patch = torch.cat([patch_list[0]], dim=0)
elif patch_num == 2 or patch_num == 3:
    adv_patch = torch.cat([patch_list[0], patch_list[1], patch_list[2]], dim=0)


# define list
clean_results = []
random_results = []
patch_results = []

object_score = []

final_results = []

print("Done")

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

if mode == 'clean':

    "CLEAN IMAGE"
    
    # create folders to save results
    if setting == 'train':
        imgdir = f'#{scene}/data/train_images' # select folder of full sized images to run detection on
        if weather_augmentations == 'off':
            folder1 = 'clean_train'
            folder2 = 'clean_train_bb'
        else:
            folder1 = 'clean_train_weather'
            folder2 = 'clean_train_weather_bb'
            
    elif setting == 'test':
        imgdir = f'#{scene}/data/test_images' # select folder of full sized images to run detection on
        if weather_augmentations == 'off':
            folder1 = 'clean_test'
            folder2 = 'clean_test_bb'  
        else:
            folder1 = 'clean_test_weather'
            folder2 = 'clean_test_weather_bb'            
    
    os.makedirs(f'{savedir}/{scene}/{mode}/' + folder1)
    os.makedirs(f'{savedir}/{scene}/{mode}/' + folder1 + '/yolo-labels')
    os.makedirs(f'{savedir}/{scene}/{mode}/' + folder2)
    
    for imgfile in os.listdir(imgdir):
        print("new clean image")
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
            
            # clean image file without extension (e.g. .jpg or .png)
            name = os.path.splitext(imgfile)[0]
            
            # label file
            txtname = name + '.txt'
    
            # directory path of label file (to save in)
            if setting == 'train':
                txtpath = os.path.abspath(os.path.join(savedir, scene, mode, folder1, 'yolo-labels/', txtname))
            elif setting == 'test':
                txtpath = os.path.abspath(os.path.join(savedir, scene, mode, folder1, 'yolo-labels/', txtname))
            
            # directory path of clean image file
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
            
            # open clean image
            img = Image.open(imgfile).convert('RGB')
           
            #######################################################################
            # RESIZE IMAGE AND RUN DETECTION 
           
            # width and height of clean image
            w,h = img.size
            
            # pad clean image with width = height
            if w==h:
                padded_img = img
            else:
                dim_to_pad = 1 if w<h else 2
                if dim_to_pad == 1:
                    padding = (h - w) / 2
                    padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                    padded_img.paste(img, (int(padding), 0))
                else:
                    padding = (w - h) / 2
                    padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                    padded_img.paste(img, (0, int(padding)))
            
            if weather_augmentations == 'on':
                
                """ WEATHER TRANSFORMATION """
                # apply weather augmentations  
                transform = transforms.ToTensor()
                padded_img = transform(padded_img).cuda()
                padded_img  = padded_img.unsqueeze(0)
                
                weather_type = random.randint(0,6) 
                # print('weather:', weather_type)
                
                if weather_type == 0:
                    padded_img = weather.brighten(padded_img)
                elif weather_type == 1:
                    padded_img= weather.darken(padded_img)
                elif weather_type == 2:
                    padded_img = weather.add_snow(padded_img)
                elif weather_type == 3:
                    padded_img = weather.add_rain(padded_img)
                elif weather_type == 4:
                    padded_img= weather.add_fog(padded_img)
                elif weather_type == 5:
                    padded_img = weather.add_autumn(padded_img)
                elif weather_type == 6:
                    padded_img = padded_img   
                
            # resize clean image
            resize = transforms.Resize((img_size,img_size))
            padded_img = resize(padded_img)
            
            if weather_augmentations == 'on':
                padded_img  = padded_img.squeeze(0)
                padded_img  = transforms.ToPILImage('RGB')(padded_img.cpu())
        
            # clean image file
            cleanname = name + ".jpg"
    
            # padded_img.save(os.path.join(savedir, 'clean/', cleanname))
            
            # input clean image into detector
            boxes = do_detect(darknet_model, padded_img, 0.50, 0.40, True)
            # boxes = do_detect(darknet_model, padded_img, 0.01, 0.40, True)


            # save clean image with labels if car is detected
            if len(boxes) >= 0:             
                if setting =='train':
                    padded_img.save(os.path.join(savedir, scene, mode, folder1, cleanname)) # select folder to save detected images
                elif setting == 'test':
                    padded_img.save(os.path.join(savedir, scene, mode, folder1, cleanname)) # select folder to save detected images
            
                # save label
                textfile = open(txtpath,'w+')
                for box in boxes:
                    cls_id = box[6]
                    if(cls_id != 0):
                        x_center = box[0]
                        y_center = box[1]
                        width = box[2]
                        height = box[3]
                        # cls_id = 0      # 0 for json file / yolo labels and comment out for patch location 
                        textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
                        
                        clean_results.append({'image_id': name, 'bbox': [x_center - width / 2,
                                                                          y_center - height / 2,
                                                                          width,
                                                                          height],
                                              'obj_score': box[4],
                                              'category_id': 1})
                        
                        object_score.append(box[4])
                        
                textfile.close()
                
                # save image with plot of bounding box
                class_names = load_class_names(namesfile)
                plot_boxes(padded_img, boxes, f'{savedir}/{scene}/{mode}/{folder2}/{cleanname}', class_names) # select folder to save detected images

    if setting == 'train':
        if weather_augmentations == 'off':
            with open(f'{savedir}/{scene}/{mode}/clean_train_detections.json', 'w') as fp:
                json.dump(clean_results, fp)
        else:
            with open(f'{savedir}/{scene}/{mode}/clean_train_w_detections.json', 'w') as fp:
                json.dump(clean_results, fp)                        
            
    elif setting == 'test':
        if weather_augmentations == 'off':    
            with open(f'{savedir}/{scene}/{mode}/clean_test_detections.json', 'w') as fp:
                json.dump(clean_results, fp)
        else:
            with open(f'{savedir}/{scene}/{mode}/clean_test_w_detections.json', 'w') as fp:
                json.dump(clean_results, fp)
    
    # print('length of object score list:', len(object_score))
    # print('average objectness score:', sum(object_score)/len(object_score))

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

elif mode == 'adversarial':
    
    "ADVERSARIAL PATCH"
    
    # create folders to save results
    if setting == 'test':
        imgdir = f'#{scene}/data/test_images' # select folder of images to run detection on
        if weather_augmentations == 'off':
            folder1 = 'patch_test'
            folder2 = 'patch_test_bb'
        else:
            folder1 = 'patch_test_weather'
            folder2 = 'patch_test_weather_bb'

    os.makedirs(f'{savedir}/{scene}/{mode}/' + folder1)
    os.makedirs(f'{savedir}/{scene}/{mode}/' + folder1 + '/yolo-labels')
    os.makedirs(f'{savedir}/{scene}/{mode}/' + folder2)
    
    for imgfile in os.listdir(imgdir):
        print("new patched image")
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
    
            # clean image file without extension (e.g. .jpg)
            name = os.path.splitext(imgfile)[0]
    
            # label file
            txtname = name + '.txt'
    
            # directory path of label file (to load from)
            if setting =='test':
                if weather_augmentations == 'off':
                    txtpath = os.path.abspath(os.path.join(f'#{scene}', 'data', 'test_labels', 'noweather', 'yolo-labels/', txtname))
                else:
                    txtpath = os.path.abspath(os.path.join(f'#{scene}', 'data', 'test_labels', 'weather', 'yolo-labels/', txtname))
    
            # load label
            label = np.loadtxt(txtpath)
    
            # # check to see if label file contains data.
            # if os.path.getsize(txtpath): # file size (in bytes)
            #     label = np.loadtxt(txtpath)
            # else:
            #     label = np.ones([5])
    
            # convert label (numpy to tensor)
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)
    
            # directory path of clean image file
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
    
            # open clean image (PIL)
            img = Image.open(imgfile).convert('RGB')
    
            # width and height of clean image
            w,h = img.size
    
            # pad clean image if neccessary
            if w==h:
                padded_img = img
            else:
                dim_to_pad = 1 if w<h else 2
                if dim_to_pad == 1:
                    padding = (h - w) / 2
                    padded_img = Image.new('RGB', (h,h), color=(127,127,127))
                    padded_img.paste(img, (int(padding), 0))
                else:
                    padding = (w - h) / 2
                    padded_img = Image.new('RGB', (w, w), color=(127,127,127))
                    padded_img.paste(img, (0, int(padding)))
    
            # # resize clean image
            # resize = transforms.Resize((img_size,img_size))
            # padded_img = resize(padded_img)
    
            # convert clean image and its label to tensor
            transform = transforms.ToTensor()
            padded_img = transform(padded_img).cuda()
            img_fake_batch = padded_img.unsqueeze(0)
            lab_fake_batch = label.unsqueeze(0).cuda()

            if weather_augmentations == 'off':                
                # apply transformation on patch
                if scene == 'sidestreet':
                    if patch_num == 1:
                        adv_batch_t = patch_transformations(adv_patch, lab_fake_batch, img.size[0], 40, do_rotate=True, rand_loc=False)    # ON patch
                    elif patch_num == 2 or patch_num == 3: 
                        adv_batch_t = patch_transformations(adv_patch, lab_fake_batch, img.size[0], 160, do_rotate=False, rand_loc=False)  # OFF patch         
                elif scene == 'carpark':
                    adv_batch_t = patch_transformations(adv_patch, lab_fake_batch, img.size[0], 85, do_rotate=True, rand_loc=False)
            
                # apply patch to clean image
                p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
        
                # resize image to 256x256
                p_img_batch = F.interpolate(p_img_batch, (darknet_model.height, darknet_model.width))
    
            else:
                # only apply weather augmentations on clean images if no detections
                if lab_fake_batch.nelement() == 0:
        
                    p_img_batch = img_fake_batch
        
                    """ WEATHER TRANSFORMATION """
                    weather_type = random.randint(0,6)
                    # print('weather:', weather_type)
        
                    if weather_type == 0:
                        p_img_batch = weather.brighten(p_img_batch)
                    elif weather_type == 1:
                        p_img_batch = weather.darken(p_img_batch)
                    elif weather_type == 2:
                        p_img_batch = weather.add_snow(p_img_batch)
                    elif weather_type == 3:
                        p_img_batch = weather.add_rain(p_img_batch)
                    elif weather_type == 4:
                        p_img_batch = weather.add_fog(p_img_batch)
                    elif weather_type == 5:
                        p_img_batch = weather.add_autumn(p_img_batch)
                    elif weather_type == 6:
                        p_img_batch = p_img_batch
        
                    # resize image to 256x256
                    p_img_batch = F.interpolate(p_img_batch, (darknet_model.height, darknet_model.width))
        
                else:
                    # apply transformation on patch
                    if scene == 'sidestreet':
                        if patch_num == 1:
                            adv_batch_t = patch_transformations(adv_patch, lab_fake_batch, img.size[0], 40, do_rotate=True, rand_loc=False)    # ON patch
                        elif patch_num == 2 or patch_num == 3: 
                            adv_batch_t = patch_transformations(adv_patch, lab_fake_batch, img.size[0], 160, do_rotate=False, rand_loc=False)  # OFF patch
                    elif scene == 'carpark':
                        adv_batch_t = patch_transformations(adv_patch, lab_fake_batch, img.size[0], 85, do_rotate=True, rand_loc=False)
        
                    # apply patch to clean image
                    p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
        
                    """ WEATHER TRANSFORMATION """
                    weather_type = random.randint(0,6)
                    # print('weather:', weather_type)
        
                    if weather_type == 0:
                        p_img_batch = weather.brighten(p_img_batch)
                    elif weather_type == 1:
                        p_img_batch = weather.darken(p_img_batch)
                    elif weather_type == 2:
                        p_img_batch = weather.add_snow(p_img_batch)
                    elif weather_type == 3:
                        p_img_batch = weather.add_rain(p_img_batch)
                    elif weather_type == 4:
                        p_img_batch = weather.add_fog(p_img_batch)
                    elif weather_type == 5:
                        p_img_batch = weather.add_autumn(p_img_batch)
                    elif weather_type == 6:
                        p_img_batch = p_img_batch
        
                    # resize image to 256x256
                    p_img_batch = F.interpolate(p_img_batch, (darknet_model.height, darknet_model.width))
        
                # # plot patched image
                # img = p_img_batch[0, :, :, :]
                # img = transforms.ToPILImage()(img.detach().cpu())
                # img.show()
    
            # save patched image
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = name + "_p.png"    
            p_img_pil.save(os.path.join(savedir, scene, mode, folder1, properpatchedname))

            # patched label file
            txtname = properpatchedname.replace('.png', '.txt')
    
            # directory path of patched label file (save in)
            txtpath = os.path.abspath(os.path.join(savedir, scene, mode, folder1, 'yolo-labels/', txtname))
    
            # clean image file
            cleanname = name + ".jpg"
    
            # input patched image into detector
            # boxes = do_detect(darknet_model, p_img_pil, 0.50, 0.40, True)
            boxes = do_detect(darknet_model, p_img_pil, 0.01, 0.40, True)
    
    
            """ PR curve and AP calculation - remember to set objectness score threshold to 0.01 """
            # save labels
            textfile = open(txtpath,'w+')
            real_boxes = []
            for box in boxes:
    
                cls_id = box[6]
                if(cls_id != 0): # if car
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    cls_id = 0
                    textfile.write(f'{cls_id} {x_center} {y_center} {width} {height}\n')
    
                    patch_results.append({'image_id': name, 'bbox': [x_center - width / 2,
                                                                      y_center - height / 2,
                                                                      width,
                                                                      height],
                                          'obj_score': box[4],
                                          'category_id': 1})
    
            textfile.close()
            """"""
    
    
            """ AORR calculation - obtain highest objectness score for each ground truth bounding box - remember to set objectness score threshold to 0.01 """
            ground = []
            final_boxes = []
            for i in range(lab_fake_batch.shape[1]):
                ground.append(lab_fake_batch[0][i].tolist())
    
            # compare ground truth against yolo detections
            if ground[0] != []:
                tolerance = 0.05
                for i in range(len(ground)):
    
                    temp_boxes = []
                    for box in boxes:
                        # compare centre points (could also use IOU here)
                        # x centre
                        if abs(ground[i][1] - box[0]) < tolerance:
                            # y centre
                            if abs(ground[i][2] - box[1]) < tolerance:
                                temp_boxes.append(box)
    
                    # remove multiple detections of same object (car)
                    if len(temp_boxes) == 0:
                        final_boxes.append([ground[i][1], ground[i][2], ground[i][3], ground[i][4], 0, 0, 0])
    
                        final_results.append({'image_id': name,
                                            'centre_points': [ground[i][1], ground[i][2]],
                                            'obj_score': 0})
    
                    elif len(temp_boxes) == 1:
                        final_boxes.append(temp_boxes[0])
    
                        final_results.append({'image_id': name,
                                            'centre_points': [temp_boxes[0][0], temp_boxes[0][1]],
                                            'obj_score': temp_boxes[0][4]})
    
                    elif len(temp_boxes) > 1:
                        from operator import itemgetter
                        def max_val(l, i):
                            return max(enumerate(map(itemgetter(i), l)),key=itemgetter(1))
    
                        index, obj_score = max_val(temp_boxes, -3)
    
                        final_boxes.append(temp_boxes[index])
    
                        final_results.append({'image_id': name,
                                            'centre_points': [temp_boxes[index][0], temp_boxes[index][1]],
                                            'obj_score': temp_boxes[index][4]})
            """"""
    
            # save image with plot of bounding box
            class_names = load_class_names(namesfile)
            plot_boxes(p_img_pil, final_boxes, f'{savedir}/{scene}/{mode}/{folder2}/{cleanname}', class_names)
            # plot_boxes(p_img_pil, boxes, f'{savedir}/{scene}/{mode}/{folder2}/{cleanname}', class_names)
    
    if weather_augmentations == 'off':
        with open(f'{savedir}/{scene}/{mode}/patch_test_detections.json', 'w') as fp:
            json.dump(final_results, fp)
    else:
        with open(f'{savedir}/{scene}/{mode}/patch_test_w_detections.json', 'w') as fp:
            json.dump(final_results, fp)      

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################

elif mode == 'random':

    "RANDOM PATCH"

    random_patch = torch.rand(adv_patch.size()).cuda()
    
    # create folders to save results
    if setting == 'test':
        imgdir = f'#{scene}/data/test_images' # select folder of images to run detection on
        if weather_augmentations == 'off':
            folder1 = 'random_test'
            folder2 = 'random_test_bb'
        else:
            folder1 = 'random_test_weather'
            folder2 = 'random_test_weather_bb'            
        
    os.makedirs(f'{savedir}/{scene}/{mode}/' + folder1)
    os.makedirs(f'{savedir}/{scene}/{mode}/' + folder1 + '/yolo-labels')
    os.makedirs(f'{savedir}/{scene}/{mode}/' + folder2)
    
    for imgfile in os.listdir(imgdir):
        print("new patched image")
        if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
    
            # clean image file without extension (e.g. .jpg)
            name = os.path.splitext(imgfile)[0]
    
            # label file
            txtname = name + '.txt'
    
            # directory path of label file (load from)
            if setting =='test':
                if weather_augmentations == 'off':
                    txtpath = os.path.abspath(os.path.join(f'#{scene}', 'data', 'test_labels', 'noweather', 'yolo-labels/', txtname))
                else:
                    txtpath = os.path.abspath(os.path.join(f'#{scene}', 'data', 'test_labels', 'weather', 'yolo-labels/', txtname))
    
            # load label
            label = np.loadtxt(txtpath)
    
            # # check to see if label file contains data.
            # if os.path.getsize(txtpath): # file size (in bytes)
            #     label = np.loadtxt(txtpath)
            # else:
            #     label = np.ones([5])
    
            # convert label (numpy to tensor)
            label = torch.from_numpy(label).float()
            if label.dim() == 1:
                label = label.unsqueeze(0)
    
            # directory path of clean image file
            imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
    
            # open clean image (PIL)
            img = Image.open(imgfile).convert('RGB')
    
            # width and height of clean image
            w, h = img.size
    
            # pad clean image if neccessary
            if w == h:
                padded_img = img
            else:
                dim_to_pad = 1 if w < h else 2
                if dim_to_pad == 1:
                    padding = (h - w) / 2
                    padded_img = Image.new('RGB', (h, h), color=(127, 127, 127))
                    padded_img.paste(img, (int(padding), 0))
                else:
                    padding = (w - h) / 2
                    padded_img = Image.new('RGB', (w, w), color=(127, 127, 127))
                    padded_img.paste(img, (0, int(padding)))
    
            # # resize clean image
            # resize = transforms.Resize((img_size,img_size))
            # padded_img = resize(padded_img)
    
            # convert clean image and its label to tensor
            transform = transforms.ToTensor()
            padded_img = transform(padded_img).cuda()
            img_fake_batch = padded_img.unsqueeze(0)
            lab_fake_batch = label.unsqueeze(0).cuda()

            if weather_augmentations == 'off':
                # apply transformation on patch
                if scene == 'sidestreet':
                    if patch_num == 1:
                        adv_batch_t = patch_transformations(random_patch, lab_fake_batch, img.size[0], 40, do_rotate=True, rand_loc=False)     # ON patch
                    elif patch_num == 3:
                        adv_batch_t = patch_transformations(random_patch, lab_fake_batch, img.size[0], 160, do_rotate=False, rand_loc=False)   # OFF patch
                elif scene == 'carpark':
                    adv_batch_t = patch_transformations(random_patch, lab_fake_batch, img.size[0], 85, do_rotate=True, rand_loc=False)
                    
                # apply patch to clean image
                p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
        
                # resize image to 256x256
                p_img_batch = F.interpolate(p_img_batch, (darknet_model.height, darknet_model.width))
    
            else:
                # only apply weather augmentations on clean images if no detections
                if lab_fake_batch.nelement() == 0:
        
                    p_img_batch = img_fake_batch
        
                    """ WEATHER TRANSFORMATION """
                    weather_type = random.randint(0,6)
                    # print('weather:', weather_type)
        
                    if weather_type == 0:
                        p_img_batch = weather.brighten(p_img_batch)
                    elif weather_type == 1:
                        p_img_batch = weather.darken(p_img_batch)
                    elif weather_type == 2:
                        p_img_batch = weather.add_snow(p_img_batch)
                    elif weather_type == 3:
                        p_img_batch = weather.add_rain(p_img_batch)
                    elif weather_type == 4:
                        p_img_batch = weather.add_fog(p_img_batch)
                    elif weather_type == 5:
                        p_img_batch = weather.add_autumn(p_img_batch)
                    elif weather_type == 6:
                        p_img_batch = p_img_batch
        
                    # resize image to 256x256
                    p_img_batch = F.interpolate(p_img_batch, (darknet_model.height, darknet_model.width))
        
                else:
                    # apply transformation on patch
                    if scene == 'sidestreet':
                        if patch_num == 1:
                            adv_batch_t = patch_transformations(random_patch, lab_fake_batch, img.size[0], 40, do_rotate=True, rand_loc=False)     # ON patch
                        elif patch_num == 2 or patch_num == 3:
                            adv_batch_t = patch_transformations(random_patch, lab_fake_batch, img.size[0], 160, do_rotate=False, rand_loc=False)   # OFF patch
                    elif scene == 'carpark':
                        adv_batch_t = patch_transformations(random_patch, lab_fake_batch, img.size[0], 85, do_rotate=True, rand_loc=False)     # ON patch
                    
                    # apply patch to clean image
                    p_img_batch = patch_applier(img_fake_batch, adv_batch_t)
        
                    """ WEATHER TRANSFORMATION """
                    weather_type = random.randint(0,6)
                    # print('weather:', weather_type)
        
                    if weather_type == 0:
                        p_img_batch = weather.brighten(p_img_batch)
                    elif weather_type == 1:
                        p_img_batch = weather.darken(p_img_batch)
                    elif weather_type == 2:
                        p_img_batch = weather.add_snow(p_img_batch)
                    elif weather_type == 3:
                        p_img_batch = weather.add_rain(p_img_batch)
                    elif weather_type == 4:
                        p_img_batch = weather.add_fog(p_img_batch)
                    elif weather_type == 5:
                        p_img_batch = weather.add_autumn(p_img_batch)
                    elif weather_type == 6:
                        p_img_batch = p_img_batch
        
                    # resize image to 256x256
                    p_img_batch = F.interpolate(p_img_batch, (darknet_model.height, darknet_model.width))
        
                # # plot patched image
                # img = p_img_batch[0, :, :, :]
                # img = transforms.ToPILImage()(img.detach().cpu())
                # img.show()
    
            # save patched image
            p_img = p_img_batch.squeeze(0)
            p_img_pil = transforms.ToPILImage('RGB')(p_img.cpu())
            properpatchedname = name + "_r.png"
            p_img_pil.save(os.path.join(savedir, scene, mode, folder1, properpatchedname))

            # patched label file
            txtname = properpatchedname.replace('.png', '.txt')
    
            # directory path of patched label file (save in)
            txtpath = os.path.abspath(os.path.join(savedir, scene, mode, folder1, 'yolo-labels/', txtname))
    
            # clean image file
            cleanname = name + ".jpg"
    
            # input patched image into detector
            boxes = do_detect(darknet_model, p_img_pil, 0.01, 0.40, True)        # for json files
    
    
            """ PR curve and AP calculation - remember to set objectness score threshold to 0.01 """
            # save labels
            textfile = open(txtpath, 'w+')
            real_boxes = []
            for box in boxes:
                cls_id = box[6]
                if(cls_id != 0):
                    x_center = box[0]
                    y_center = box[1]
                    width = box[2]
                    height = box[3]
                    cls_id = 0
                    textfile.write(
                        f'{cls_id} {x_center} {y_center} {width} {height}\n')
    
                    random_results.append({'image_id': name, 'bbox': [x_center - width / 2,
                                                                      y_center - height / 2,
                                                                      width,
                                                                      height],
                                          'score': box[4],
                                          'category_id': 1})
    
            textfile.close()
            """"""
    
    
            """ AORR calculation - obtain highest objectness score for each ground truth bounding box - remember to set objectness score threshold to 0.01 """
            ground = []
            final_boxes = []
            for i in range(lab_fake_batch.shape[1]):
                ground.append(lab_fake_batch[0][i].tolist())
                
            # COMPARE GROUND TRUTH AGAINST YOLO DETECTIONS
            if ground[0] != []:
                tolerance = 0.05
                for i in range(len(ground)):
                    
                    temp_boxes = []
                    for box in boxes:
                        # COMPARE CENTRE POINTS (COULD ALSO USE IOU HERE)
                        # x centre
                        if abs(ground[i][1] - box[0]) < tolerance:
                            # y centre
                            if abs(ground[i][2] - box[1]) < tolerance:
                                temp_boxes.append(box)
                    
                    # REMOVE MULTIPLE DETECTIONS
                    if len(temp_boxes) == 0:
                        final_boxes.append([ground[i][1], ground[i][2], ground[i][3], ground[i][4], 0, 0, 0])
                        
                        final_results.append({'image_id': name,
                                            'centre_points': [ground[i][1], ground[i][2]],
                                            'obj_score': 0})
                    
                    elif len(temp_boxes) == 1:
                        final_boxes.append(temp_boxes[0])
                        
                        final_results.append({'image_id': name,
                                            'centre_points': [temp_boxes[0][0], temp_boxes[0][1]],
                                            'obj_score': temp_boxes[0][4]})
                        
                    elif len(temp_boxes) > 1:
                        from operator import itemgetter
                        def max_val(l, i):
                            return max(enumerate(map(itemgetter(i), l)),key=itemgetter(1))
                        
                        index, obj_score = max_val(temp_boxes, -3)
                        
                        final_boxes.append(temp_boxes[index])
                    
                        final_results.append({'image_id': name,
                                            'centre_points': [temp_boxes[index][0], temp_boxes[index][1]],
                                            'obj_score': temp_boxes[index][4]})  
            """"""

            # save image with plot of bounding box
            class_names = load_class_names(namesfile)
            plot_boxes(p_img_pil, final_boxes, f'{savedir}/{scene}/{mode}/{folder2}/{cleanname}', class_names)
            # plot_boxes(p_img_pil, boxes, f'{savedir}/{scene}/{mode}/{folder2}/{cleanname}', class_names)
    
    if weather_augmentations == 'off':
        with open(f'{savedir}/{scene}/{mode}/random_test_detections.json', 'w') as fp:
            json.dump(final_results, fp)
    else:
        with open(f'{savedir}/{scene}/{mode}/random_test_w_detections.json', 'w') as fp:
            json.dump(final_results, fp)  
