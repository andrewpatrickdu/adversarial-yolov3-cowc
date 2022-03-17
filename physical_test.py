#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:53:37 2022

A script to calculate the average objectness score of a targeted car with and 
without a patch. This score is used to calculate OSR.This script also calculates 
the number of detections of a targeted car for a given objectness threshold 
which is used to calculate NDR. 

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
''' 
We provide 3 attack scenarios:
    (1) Carpark - Patch ON - Blue car
    (2) Sidestreet - Patch ON - Gray car
    (3) Sidestreet - Patch OFF - White car
Please select variables (scene, mode, car, imgdir) accordingly!
'''
savedir = "physical_test"

scene = 'sidestreet' # sidestreet, carpark
mode = 'patch_off' # clean, patch_off, patch_on
car = 'white' # blue, gray, white

# select folder of full sized images to run detection on
imgdir = f'{savedir}/{scene}/{mode}/white'
# imgdir = f'{savedir}/{scene}/{mode}/gray'
# imgdir = f'{savedir}/{scene}/{mode}/blue'

folder = 'physical_bb'
os.makedirs(f'{savedir}/{scene}/results/' + folder)

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

# define model parameters
batch_size = 1
max_lab = 10
img_size = darknet_model.height


# define list
physical_results = []

object_score = []

final_results = []
undetections = 0

print("Done")

###########################################################################################################################
###########################################################################################################################
###########################################################################################################################
for imgfile in os.listdir(imgdir):
    print("new image")
    if imgfile.endswith('.jpg') or imgfile.endswith('.png'):
        
        # clean image file without extension (e.g. .jpg or .png)
        name = os.path.splitext(imgfile)[0]
        
        # label file
        txtname = name + '.txt'
    
        # directory path of label file (to load in)
        txtpath = os.path.abspath(os.path.join(savedir, scene, mode, car, 'yolo-labels/', txtname))
        
        # load label
        label = np.loadtxt(txtpath)    
        
        # convert label (numpy to tensor)
        label = torch.from_numpy(label).float()
        if label.dim() == 1:
            label = label.unsqueeze(0)
        
        # directory path of image file
        imgfile = os.path.abspath(os.path.join(imgdir, imgfile))
        
        # open image
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
            
        # resize clean image
        resize = transforms.Resize((img_size,img_size))
        padded_img = resize(padded_img)
    
        lab_fake_batch = label.unsqueeze(0).cuda()
    
        p_img_batch = padded_img
        
        properpatchedname = name + "_p.png"
        
        # label file to save
        txtname = properpatchedname.replace('.png', '.txt')
    
        # clean image file
        cleanname = name + ".jpg"
    
        # padded_img.save(os.path.join(savedir, 'clean/', cleanname))
        
        # input clean image into detector
        boxes = do_detect(darknet_model, padded_img, 0.01, 0.40, True)

        # extract highest objectness score of target car bounding box
        ground = []
        final_boxes = []
        for i in range(lab_fake_batch.shape[1]):
            ground.append(lab_fake_batch[0][i].tolist())

        # COMPARE GROUND TRUTH AGAINST YOLO DETECTIONS
        if ground[0] != []:
            # tolerance = 0.03
            tolerance = 0.06
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
                    undetections = undetections + 1
                    
                    object_score.append(0)

                elif len(temp_boxes) == 1:
                    final_boxes.append(temp_boxes[0])

                    final_results.append({'image_id': name,
                                        'centre_points': [temp_boxes[0][0], temp_boxes[0][1]],
                                        'obj_score': temp_boxes[0][4]})
                    
                    object_score.append(temp_boxes[0][4])

                elif len(temp_boxes) > 1:
                    from operator import itemgetter
                    def max_val(l, i):
                        return max(enumerate(map(itemgetter(i), l)),key=itemgetter(1))

                    index, obj_score = max_val(temp_boxes, -3)

                    final_boxes.append(temp_boxes[index])

                    final_results.append({'image_id': name,
                                        'centre_points': [temp_boxes[index][0], temp_boxes[index][1]],
                                        'obj_score': temp_boxes[index][4]})
                    
                    object_score.append(temp_boxes[index][4])            

            # save image with plot of bounding box
            class_names = load_class_names(namesfile)
            plot_boxes(padded_img, final_boxes, f'{savedir}/{scene}/results/{folder}/{cleanname}', class_names) # select folder to save detected images
    

with open(f'{savedir}/{scene}/results/physical_detections.json', 'w') as fp:
    json.dump(object_score, fp)

print('\n')
average_obj = sum(object_score)/(len(object_score))
print('average objectness score:', average_obj)
print('number of detections:', len(object_score))


