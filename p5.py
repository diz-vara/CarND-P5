# -*- coding: utf-8 -*-
"""
Created on Mon May 15 22:36:12 2017

@author: diz
"""

import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from skimage.feature import hog


os.chdir('D:\\WORK\\CarND\\p5\\CarND-P5')
test_dir = './test_images/'
out_dir = './out/'


#%%

images = []
for entry in os.scandir(test_dir):
    if entry.is_file():
        print(entry.path)
        img = cv2.imread(entry.path)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        images.append(img)


#%%
# Read in our vehicles and non-vehicles

def get_image_list(path):
    images = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(path)
        for f in files if f.endswith('.png')]
    return images
    
path = 'e:/data/carnd/p5/'
cars = get_image_list(path+'vehicles')
notcars = get_image_list(path+'non-vehicles')

myPath = 'e:/data/carnd/p5/my/'
myCars = get_image_list(myPath + 'Car')
myNotCars = get_image_list(myPath + 'Etc')

#%%
heat = np.zeros_like(img[:,:,0]).astype(np.float)
thr = 10
tau = 0.96
from moviepy.editor import VideoFileClip
#from IPython.display import HTML
     
video_output = 'out/p5_096_10.mp4'
clip = VideoFileClip('project_video.mp4')
#clip = VideoFileClip('E:\\Data\\USA\\Video\\cuts\\multiple_01.avi') 
first_clip = clip.fl_image(process_image)
get_ipython().magic('time first_clip.write_videofile(video_output, audio=False)')

