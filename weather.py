from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import random

def visualize(image):
    
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image)
    
def hls(image, src='RGB'):

    image_HLS = eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2HLS)')
    
    return image_HLS

def rgb(image, src='BGR'):

    image_RGB = eval('cv2.cvtColor(image,cv2.COLOR_'+src.upper()+'2RGB)')
    
    return image_RGB

""" """
def change_light(image, coeff):
    
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    image_HLS[:,:,1] = image_HLS[:,:,1]*coeff ## Scale pixel values up or down for channel 1(Lightness)
    
    if(coeff>1):
        image_HLS[:,:,1][image_HLS[:,:,1]>255] = 255 ## Sets all values above 255 to 255
    else:
        image_HLS[:,:,1][image_HLS[:,:,1]<0] = 0
    
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB) ## Conversion to RGB
    
    return image_RGB 

""" """

""" LIGHTEN """
def brighten(image, brightness_coeff=-1): ##function to brighten the image

    # create mask
    brighten_mask = image.squeeze(0)
    brighten_mask = brighten_mask.detach().cpu().numpy()
    brighten_mask = np.moveaxis(brighten_mask, 0, -1)
    brighten_mask = brighten_mask*255
    brighten_mask = brighten_mask.astype('uint8')

    if(brightness_coeff == -1):
        brightness_coeff_t = 1 + random.uniform(0,1) ## coeff between 1.0 and 1.5
    else:
        brightness_coeff_t = 1 + brightness_coeff ## coeff between 1.0 and 2.0
            
    brighten_mask = change_light(brighten_mask, brightness_coeff_t)
    
    # normalise mask to 0-1
    brighten_mask  = brighten_mask/255
    
    # convert to tensor
    tf = transforms.ToTensor()
    brighten_mask  = tf(brighten_mask )
    brighten_mask  = brighten_mask.unsqueeze(0)
    brighten_mask  = brighten_mask.type(torch.cuda.FloatTensor)

    brighten_mask  = brighten_mask - image
    
    # add brighten mask to image
    image_rgb = image + brighten_mask 
    
    image_rgb = torch.clamp(image_rgb, 0.000001, 0.99999) 
    
    return image_rgb

""" """

""" DARKEN """
def darken(image, darkness_coeff=-1): ##function to darken the image

    # create mask    
    darken_mask = image.squeeze(0)
    darken_mask = darken_mask.detach().cpu().numpy()
    darken_mask = np.moveaxis(darken_mask, 0, -1)
    darken_mask = darken_mask*255
    darken_mask = darken_mask.astype('uint8')

    if(darkness_coeff == -1):
         darkness_coeff_t = 1 - random.uniform(0,1)
    else:
        darkness_coeff_t = 1- darkness_coeff  
        
    darken_mask = change_light(darken_mask, darkness_coeff_t)
    
    # normalise mask to 0-1
    darken_mask  = darken_mask/255
    
    # convert to tensor
    tf = transforms.ToTensor()
    darken_mask = tf(darken_mask)
    darken_mask = darken_mask.unsqueeze(0)
    darken_mask = darken_mask.type(torch.cuda.FloatTensor)

    darken_mask = image - darken_mask
    
    # add darken mask to image
    image_rgb = image - darken_mask 
    
    image_rgb = torch.clamp(image_rgb, 0.000001, 0.99999)     
    
    return image_rgb

""" """

""" SNOW """
def snow_process(image, snow_coeff):
    
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    
    # define parameters
    brightness_coefficient = 2.5 
    imshape = image.shape
    snow_point = snow_coeff ## increase this for more snow
    
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255] = 255 ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    
    return image_RGB

def add_snow(image, snow_coeff=-1):

    # create mask
    snow_mask = image.squeeze(0)
    snow_mask = snow_mask.detach().cpu().numpy()
    snow_mask = np.moveaxis(snow_mask, 0, -1)
    snow_mask = snow_mask*255
    snow_mask = snow_mask.astype('uint8')
    
    # define parameters
    snow_coeff = random.uniform(0,1)
    snow_coeff *= 255/2
    snow_coeff += 255/3
    
    snow_mask = snow_process(snow_mask, snow_coeff)
    
    # normalise mask to 0-1
    snow_mask = snow_mask/255
    
    # convert to tensor
    tf = transforms.ToTensor()
    snow_mask = tf(snow_mask)
    snow_mask = snow_mask.unsqueeze(0)
    snow_mask = snow_mask.type(torch.cuda.FloatTensor)

    snow_mask = snow_mask - image
    
    # add snow to image
    image_rgb = image + snow_mask
    
    image_rgb = torch.clamp(image_rgb, 0.000001, 0.99999) 

    return image_rgb

""" """

""" RAIN """
def generate_random_lines(imshape, slant, drop_length, rain_type):
    
    drops = []
    area = imshape[0]*imshape[1]
    no_of_drops = area//600

    if rain_type.lower() == 'drizzle':
        no_of_drops = area//770
        drop_length = 10
    elif rain_type.lower() == 'heavy':
        drop_length = 30
    elif rain_type.lower() == 'torrential':
        no_of_drops = area//500
        drop_length = 60

    for i in range(no_of_drops): ## If You want heavy rain, try increasing this
        if slant < 0:
            x = np.random.randint(slant, imshape[1])
        else:
            x = np.random.randint(0, imshape[1]-slant)
            
        y = np.random.randint(0, imshape[0]-drop_length)
        drops.append((x, y))
    
    return drops, drop_length

def rain_process(image, slant, drop_length, drop_color, drop_width, rain_drops):
    
    imshape = image.shape  
    image_t = image.copy()
    
    for rain_drop in rain_drops:
        cv2.line(image_t, (rain_drop[0],rain_drop[1]), (rain_drop[0]+slant,rain_drop[1]+drop_length), drop_color, drop_width)
        
    image = cv2.blur(image_t,(7,7)) ## rainy view are blurry
    
    brightness_coefficient = 0.7 ## rainy days are usually shady 
    
    image_HLS = hls(image) ## Conversion to HLS
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    
    image_RGB = rgb(image_HLS, 'hls') ## Conversion to RGB
    
    return image_RGB

##rain_type='drizzle','heavy','torrential'
def add_rain(image, slant=-1, drop_length=20, drop_width=1, drop_color=(200,200,200), rain_type='None'): ## (200,200,200) a shade of gray

    rain_mask = image.squeeze(0)
    rain_mask = rain_mask.detach().cpu().numpy()
    rain_mask = np.moveaxis(rain_mask, 0, -1)
    rain_mask = rain_mask*255
    rain_mask = rain_mask.astype('uint8')

    slant_extreme = slant
    imshape = rain_mask.shape
    
    if slant_extreme == -1:
        slant= np.random.randint(-10,10) ##generate random slant if no slant value is given
    
    rain_drops, drop_length = generate_random_lines(imshape, slant, drop_length, rain_type)
    
    rain_mask = rain_process(rain_mask, slant_extreme, drop_length, drop_color, drop_width, rain_drops)
    
    # normalise mask to 0-1
    rain_mask = rain_mask/255
    
    # convert mask to tensor
    tf = transforms.ToTensor()
    rain_mask = tf(rain_mask)
    rain_mask = rain_mask.unsqueeze(0)
    rain_mask = rain_mask.type(torch.cuda.FloatTensor)
    
    rain_mask = rain_mask - image
    
    # add rain to image 
    image_rgb = image + rain_mask
    
    image_rgb = torch.clamp(image_rgb, 0.000001, 0.99999)    

    return image_rgb

""" """

""" FOG """
def add_blur(image, x, y, hw, fog_coeff):
    
    overlay = image.copy()
    output = image.copy()
    alpha = 0.08*fog_coeff
    rad = hw//2
    point = (x+hw//2, y+hw//2)

    cv2.circle(overlay, point, int(rad), (255,255,255), -1)
    
    cv2.addWeighted(overlay, alpha, output, 1 -alpha , 0, output)
    
    return output

def generate_random_blur_coordinates(imshape, hw):
    
    blur_points = []
    midx = imshape[1]//2-2*hw
    midy = imshape[0]//2-hw
    index = 1
    
    while(midx > -hw or midy > -hw):
        for i in range(hw//10*index):
            x = np.random.randint(midx, imshape[1]-midx-hw)
            y = np.random.randint(midy, imshape[0]-midy-hw)
            blur_points.append((x,y))
            
        midx -= 3*hw*imshape[1]//sum(imshape)
        midy -= 3*hw*imshape[0]//sum(imshape)
        index += 1
        
    return blur_points

def add_fog(image, fog_coeff=-1):
    
    fog_mask = image.squeeze(0)
    fog_mask = fog_mask.detach().cpu().numpy()
    fog_mask = np.moveaxis(fog_mask, 0, -1)
    fog_mask = fog_mask*255
    fog_mask = fog_mask.astype('uint8')
    
    imshape = fog_mask.shape
    
    if fog_coeff == -1:
        fog_coeff_t = random.uniform(0.3,1)
    else:
        fog_coeff_t = fog_coeff
        
    hw = int(imshape[1]//3*fog_coeff_t)
    haze_list = generate_random_blur_coordinates(imshape, hw)
    
    for haze_points in haze_list: 
        fog_mask = add_blur(fog_mask, haze_points[0], haze_points[1], hw, fog_coeff_t) 
    
    fog_mask = cv2.blur(fog_mask, (hw//10,hw//10))
    
    # normalise mask to 0-1
    fog_mask = fog_mask/255
    
    # convert mask to tensor
    tf = transforms.ToTensor()
    fog_mask = tf(fog_mask)
    fog_mask = fog_mask.unsqueeze(0)
    fog_mask = fog_mask.type(torch.cuda.FloatTensor)
    
    fog_mask = fog_mask - image
    
    # add fog to image 
    image_rgb = image + fog_mask
    
    image_rgb = torch.clamp(image_rgb, 0.000001, 0.99999)
    
    return image_rgb
""" """

""" AUTUMN """
def autumn_process(image):
    
    image_t = image.copy()
    imshape = image_t.shape
    image_hls = hls(image_t)
    
    step = 8
    aut_colors = [1,5,9,11]
    col = aut_colors[random.randint(0,3)]
    
    for i in range(0,imshape[1],step):
        for j in range(0,imshape[0],step):
            avg = np.average(image_hls[j:j+step,i:i+step,0])
#             print(avg)
            if(avg >20 and avg< 100 and np.average(image[j:j+step,i:i+step,1])<100):
                image_hls[j:j+step,i:i+step,0] = col
                image_hls[j:j+step,i:i+step,2] = 255
    
    return rgb(image_hls,'hls')


def add_autumn(image):

    autumn_mask = image.squeeze(0)
    autumn_mask = autumn_mask.detach().cpu().numpy()
    autumn_mask = np.moveaxis(autumn_mask, 0, -1)
    autumn_mask = autumn_mask*255
    autumn_mask = autumn_mask.astype('uint8')

    autumn_mask = autumn_process(autumn_mask)
    
    # normalise mask to 0-1
    autumn_mask  = autumn_mask/255
    
    # convert to tensor
    tf = transforms.ToTensor()
    autumn_mask = tf(autumn_mask)
    autumn_mask = autumn_mask.unsqueeze(0)
    autumn_mask = autumn_mask.type(torch.cuda.FloatTensor)

    autumn_mask = autumn_mask - image
    
    # add autumn leaves to image
    image_rgb = image + autumn_mask
    
    image_rgb = torch.clamp(image_rgb, 0.000001, 0.99999) 

    return image_rgb