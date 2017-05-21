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
    win_draw = np.int(window*scale)

    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    #cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # size of one hog vector
    hogSize = nblocks_per_window * nblocks_per_window * cell_per_block * cell_per_block * orient

    #define shape for target matrices
    hogShape = (nysteps, nxsteps, hogSize)
    # Compute individual channel HOG features for the entire image
    #done hog_cv.compute (ch1, (16,16))
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, window, cells_per_step).reshape(hogShape)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, window, cells_per_step).reshape(hogShape)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, window, cells_per_step).reshape(hogShape)
    
    #patches = np.zeros((nysteps, nxsteps, window, window,3),np.uint8)
    boxes = np.zeros((nxsteps*nysteps,2,2), np.int32)
    bboxes = []
    features = np.zeros((nxsteps*nysteps, hogSize*3), np.float32)
    #todo: loop
    for yb in range(nysteps):
        for xb in range(nxsteps):
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
            #subimg = ctrans_tosearch[ytop:ytop+window, xleft:xleft+window]
            #patches[yb,xb] = subimg
          
            # Get color features
            #spatial_features = bin_spatial(subimg, size=spatial_size)
            #hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            features[yb*nxsteps + xb] = (X_scaler.transform(hog_features.reshape(1,-1)))
            boxes[yb*nxsteps + xb]  = ((xleft*scale, ytop*scale + ystart), 
                                       ((xleft+window)*scale, (ytop+window)*scale+ystart))
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
    
    prediction = unSvm.predict(features)[1].ravel()
         
    for i in range(len(prediction)):
        if (prediction[i] == 1):
            bboxes.append( boxes[i] );
                
    return bboxes
    
#%%
def draw_bboxes(img, bboxes):
    draw_img = np.copy(img)

    color = (0,0,255)
    thickness = 6


    for box in boxes:
        #print ("detected at", xb,yb, xbox_left, ytop_draw)
        cv2.rectangle(draw_img, tuple(box[0]), tuple(box[1]), color, thickness) 
    return draw_img    

#%%
img = images[4]
t=time.time()
boxes = find_cars(img, 400, 600, 5, mySvm, X_scaler, 11, 8, 2, 32, 2)
t2=time.time()
print(round((t2-t)*1000), 'ms for image')

out_img = draw_bboxes(img, boxes)
plt.imshow(out_img)

#%%

def boxes_multy_scale(img):
    global scales
    global window
    top = 400
    cell = 8 #cell size in pixels
    shift = 2 #shift in cells
    boxes = []
    for scale in scales:
        #bottom = top + np.int(window * scale) + 1
        bottom = top + np.int(window * scale + cell * shift * scale) + 1
        boxes.extend(find_cars(img, top, bottom, scale, mySvm, X_scaler, 11, 
                               cell, 2, window, shift))
    return boxes
    
def add_heat(heatmap, bbox_list, tau=0.9):
    # Iterate through list of bboxes
    heatmap = heatmap * tau
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        width = box[1][1]-box[0][1];
        height = box[1][0]-box[0][0];
        bx = np.ones(( width, height), np.float32) #/2
        #bx2 = np.ones((width//2, height//2), np.float32)/2
        #bx[width//4:(width + width//2)//2, height//4:(height+height//2)//2] += bx2                
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += bx

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        x0 = np.min(nonzerox)
        x1 = np.max(nonzerox)
        y0 = np.min(nonzeroy)
        y1 = np.max(nonzeroy)
        w = x1 - x0
        h = y1 - y0
        if (w > window * 2 and h > window * 2):
            bbox = ((x0, y0), (x1, y1))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,100,0), 3)
    # Return the image
    return img
    
#%%

heat = np.zeros_like(img[:,:,0]).astype(np.float)
tau = 0.9
t=time.time()

for i in [3,4,5]:
    img = images[i]
    boxes = boxes_multy_scale(img)
    heat = add_heat(heat,boxes,tau)
t2=time.time()
print(round((t2-t)*1000), 'ms for image')

#out_img = draw_bboxes(img, boxes)
thr = 6
#heat[ heat < thr] = 0
#plt.imshow(heat)

from scipy.ndimage.measurements import label

labels = label(heat)
draw_img = draw_labeled_bboxes(np.copy(img), labels)
plt.imshow(draw_img)

#%%
#main processing pipeline
def process_image(image):
    global heat;


    boxes = boxes_multy_scale(image)
    heat = add_heat(heat,boxes,tau)
    
    heat_thr = heat.copy();
    heat_thr[heat_thr < thr] = 0;

    labels = label(heat_thr)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img
    

#%%

