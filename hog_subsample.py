# -*- coding: utf-8 -*-
"""
Created on Sun May 14 18:53:35 2017

@author: diz
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
#from lesson_functions import *

#dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
#svc = dist_pickle["svc"]
#X_scaler = dist_pickle["scaler"]
#orient = dist_pickle["orient"]
#pix_per_cell = dist_pickle["pix_per_cell"]
#cell_per_block = dist_pickle["cell_per_block"]
#spatial_size = dist_pickle["spatial_size"]
#ist_bins = dist_pickle["hist_bins"]

#img = mpimg.imread('test_image.jpg')
ystart = 300
ystop = 620
scale = 4
# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, 
              cell_per_block, spatial_size, cells_per_step):
    global hot_features
    global patches
    
    draw_img = np.copy(img)
    #img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = spatial_size
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    #cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # size of one hog vector
    hogSize = nblocks_per_window * nblocks_per_window * cell_per_block * cell_per_block * orient

    #define shape for target vectors
    hogShape = (nysteps, nxsteps, hogSize)
    # Compute individual channel HOG features for the entire image
    #done hog_cv.compute (ch1, (16,16))
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, window, cells_per_step).reshape(hogShape)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, window, cells_per_step).reshape(hogShape)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, window, cells_per_step).reshape(hogShape)
    
    patches = np.zeros((nysteps, nxsteps, window, window,3),np.uint8)
    
    
    #todo: loop
    for xb in range(nxsteps):
        for yb in range(nysteps):
            # Extract HOG for this patch
            
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            
            # Extract HOG for this patch
            hog_feat1 = hog1[yb, xb]
            hog_feat2 = hog2[yb, xb]
            hog_feat3 = hog3[yb, xb]
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            # Extract the image patch
            subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
            patches[yb,xb] = subimg
          
            # Get color features
            #spatial_features = bin_spatial(subimg, size=spatial_size)
            #hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(hog_features.reshape(1,-1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = unSvm.predict(test_features)[1].ravel()
            
            xbox_left = np.int(xleft*scale)
            ytop_draw = np.int(ytop*scale)

            if test_prediction == 1:
                color = (0,0,255)
                thickness = 6
                print ("detected at", xb,yb, xbox_left, ytop_draw)
                win_draw = np.int(window*scale)
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),
                              (xbox_left+win_draw,ytop_draw+win_draw+ystart),
                              color,thickness) 
                
    return draw_img
    

#%%
    
out_img = find_cars(images[3], ystart, ystop, 3.2, mySvm, X_scaler, 11, 8, 2, 32, 1)

plt.imshow(out_img)