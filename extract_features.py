# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:30:24 2017

@author: Anton Varfolomeev
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid for scikit-learn version <= 0.17

from sklearn.cross_validation import train_test_split
# for scikit-learn >= 0.18 use:
#from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split


conversions = {'HSV':cv2.COLOR_BGR2HSV,
               'HLS':cv2.COLOR_BGR2HLS,
               'LUV':cv2.COLOR_BGR2LUV,
               'YUV':cv2.COLOR_BGR2YUV,
               'YCrCb':cv2.COLOR_BGR2YCrCb,
               'LAB':cv2.COLOR_BGR2LAB,
               'Lab':cv2.COLOR_BGR2Lab,
               'XYZ':cv2.COLOR_BGR2XYZ
               }
    

# Define a function to return HOG features
# use hog feature from cv2
# NB! use uint8!!!

def get_hog_features(image, orient = 9, pix_per_cell = 8, cell_per_block = 2, 
                     winsize = 32, winstride = 1):

    #translate sklearn parameters into OpenCV    
    block_size = pix_per_cell * cell_per_block;
    block_stride = pix_per_cell;
    
    hog_cv = cv2.HOGDescriptor((winsize,winsize),
                               (block_size,block_size), 
                               (block_stride,block_stride), 
                               (pix_per_cell,pix_per_cell), 
                               orient)
    stride = pix_per_cell * winstride 
    
    features = hog_cv.compute(image, (stride,stride));
    
    return features
        

            

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    
    
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = cv2.imread(file)
        if (image != None and image.size > 0):
            # apply color conversion if other than 'RGB'
            image = cv2.resize(image, (32,32))
            if (cspace in conversions.keys()):        
                feature_image = cv2.cvtColor(image, conversions[cspace])
            else:
                feature_image = np.copy(image)        
    
            # Call get_hog_features() with vis=False, feature_vec=True
            if type(hog_channel) == list or type(hog_channel) == tuple:
                hog_features = []
                for channel in hog_channel:
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 32));
                                        
                hog_features = np.ravel(hog_features)        
            elif hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 32)); 
                                        
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, 32); 
                            
            # Append the new feature vector to the features list
            features.append(hog_features)
    # Return list of feature vectors
    return features

