#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:53:37 2022

A script to calculate AORR.

@author: andrew
"""

""" WACV DIGITAL RESULTS """ 
import json
print('\n')
print('DIGITAL TEST ON SIDESTREET SCENE')

''' NO WEATHER '''
print('Without weather transformations:')
# open json files and calculate aorr
with open(f'json/sidestreet/sidestreet_clean.json') as jsonFile:
    clean = json.load(jsonFile)
    jsonFile.close()


###############################################################################

###############################################################################
with open(f'json/sidestreet/sidestreet_clean_patch_gcw.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD) - ON G/C+W:', sum(obj_score)/len(obj_score)*100)

###############################################################################

###############################################################################
with open(f'json/sidestreet/sidestreet_clean_patch_gc.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD) - ON G/C:', sum(obj_score)/len(obj_score)*100)

###############################################################################

###############################################################################
with open(f'json/sidestreet/sidestreet_clean_random.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD) - ON Control:', sum(obj_score)/len(obj_score)*100)

############################################################################### 

###############################################################################   
with open(f'json/sidestreet/sidestreet_clean_patch_off.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
    
print('AORR (STD) - OFF G/C:', sum(obj_score)/len(obj_score)*100)

###############################################################################

###############################################################################
with open(f'json/sidestreet/sidestreet_clean_random_off.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
    
print('AORR (STD) - OFF Control:', sum(obj_score)/len(obj_score)*100)


###############################################################################
''' WEATHER '''
print('With weather transformations:')
# open json files and calculate aorr
with open(f'json/sidestreet/sidestreet_weather.json') as jsonFile:
    clean = json.load(jsonFile)
    jsonFile.close()


###############################################################################

###############################################################################
with open(f'json/sidestreet/sidestreet_weather_patch_gcw.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD+W) - ON G/C+W:', sum(obj_score)/len(obj_score)*100)

###############################################################################

###############################################################################
with open(f'json/sidestreet/sidestreet_weather_patch_gc.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD+W) - ON G/C:', sum(obj_score)/len(obj_score)*100)

###############################################################################

###############################################################################
with open(f'json/sidestreet/sidestreet_weather_random.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD+W) - ON Control:', sum(obj_score)/len(obj_score)*100)

############################################################################### 

###############################################################################   
with open(f'json/sidestreet/sidestreet_weather_patch_off.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
    
print('AORR (STD+W) - OFF G/C:', sum(obj_score)/len(obj_score)*100)

###############################################################################

###############################################################################
with open(f'json/sidestreet/sidestreet_weather_random_off.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
    
print('AORR (STD+W) - OFF Control:', sum(obj_score)/len(obj_score)*100)

###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
print('\n')
print('DIGITAL TEST ON CARPARK SCENE')

''' NO WEATHER '''
print('Without weather transformations:')
# open json files and calculate aorr
with open(f'json/carpark/carpark_clean.json') as jsonFile:
    clean = json.load(jsonFile)
    jsonFile.close()


###############################################################################

###############################################################################
with open(f'json/carpark/carpark_clean_patch_gcw.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD) - ON G/C+W:', sum(obj_score)/len(obj_score)*100)

###############################################################################

###############################################################################
with open(f'json/carpark/carpark_clean_patch_gc.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD) - ON G/C:', sum(obj_score)/len(obj_score)*100)

###############################################################################

###############################################################################
with open(f'json/carpark/carpark_clean_random.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD) - ON Control:', sum(obj_score)/len(obj_score)*100)


###############################################################################
''' WEATHER '''
print('With weather transformations:')
# open json files and calculate aorr
with open(f'json/carpark/carpark_weather.json') as jsonFile:
    clean = json.load(jsonFile)
    jsonFile.close()


###############################################################################

###############################################################################
with open(f'json/carpark/carpark_weather_patch_gcw.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD+W) - ON G/C+W:', sum(obj_score)/len(obj_score)*100)

###############################################################################

###############################################################################
with open(f'json/carpark/carpark_weather_patch_gc.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD+W) - ON G/C:', sum(obj_score)/len(obj_score)*100)

###############################################################################

###############################################################################
with open(f'json/carpark/carpark_weather_random.json') as jsonFile:
    patch = json.load(jsonFile)
    jsonFile.close()

obj_score = []
for i in range(len(clean)):
    clean_obj = clean[i]['obj_score']
    patch_obj = patch[i]['obj_score']
    obj_score.append((clean_obj - patch_obj) / clean_obj)
print('AORR (STD+W) - ON Control:', sum(obj_score)/len(obj_score)*100)

