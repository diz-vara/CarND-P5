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

#%%    

# Divide up into cars and notcars
#images = glob.glob('*.jpeg')
#cars = []
#notcars = []
#for image in images:
#    if 'image' in image or 'extra' in image:
#        notcars.append(image)
#    else:
#        cars.append(image)

# Reduce the sample size because HOG features are slow to compute
# The quiz evaluator times out after 13s of CPU time
#sample_size = 500
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]

### TODO: Tweak these parameters and see how the results change.
colorspace = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 11
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' #(1,2) # Can be 0, 1, 2, or "ALL"

t=time.time()
car_features = extract_features(cars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
notcar_features = extract_features(notcars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X).astype(np.float32)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), 
               np.zeros(len(notcar_features)))).astype(np.int32)


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 


#%%             
# Check the training time for the SVC
uSvm = cv_svm (X_train, X_test, y_train, y_test)
#%%
# Check the prediction time for a single sample

idx = np.arange(len(y_test))
np.random.shuffle(idx)

t=time.time()
n_predict = 1000
yr = np.zeros(n_predict)

#for i in range (n_predict):
yr = uSvm.predict(X_test[idx[:n_predict]])[1].ravel()
print(sum( yr != y_test[idx[0:n_predict]]), " mistakes from ", n_predict)
#print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


#%%
t=time.time()
myCarFeatures = extract_features(myCars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
myNotCarFeatures = extract_features(myNotCars, cspace=colorspace, orient=orient, 
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                        hog_channel=hog_channel)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to extract HOG features...')
# Create an array stack of feature vectors
myX = np.vstack((myCarFeatures, myNotCarFeatures)).astype(np.float64)                        
# Fit a per-column scaler
X_scaler = StandardScaler().fit(myX)
# Apply the scaler to X
scaled_X = X_scaler.transform(myX).astype(np.float32)

# Define the labels vector
myY = np.hstack((np.ones(len(myCarFeatures)), 
                 np.zeros(len(myNotCarFeatures)))).astype(np.int32)


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
myX_train, myX_test, myY_train, myY_test = train_test_split(
    scaled_X, myY, test_size=0.2, random_state=rand_state)
#%%
mySvm = cv_svm (myX_train, myX_test, myY_train, myY_test)

# Check the score of the SVC
print('Test Accuracy of mySVC on my data = ', round(score(mySvm,myX_test, myY_test), 4))
print('Test Accuracy of uSVC on my data  = ', round(score(uSvm,myX_test, myY_test), 4))
print('Test Accuracy of mySVC on u data = ', round(score(mySvm,X_test, y_test), 4))
print('Test Accuracy of uSVC on u data  = ', round(score(uSvm,X_test, y_test), 4))

#%%
idx = np.arange(len(myY_test))
np.random.shuffle(idx)

t=time.time()
n_predict = 1000
yr = uSvm.predict(myX_test[idx[0:n_predict]])[1].ravel()
#print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print("u on my",sum( yr != myY_test[idx[0:n_predict]]), " mistakes from ", n_predict)
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

t=time.time()
yr = mySvm.predict(myX_test[idx[0:n_predict]])[1].ravel()
#print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print("my on my",sum( yr != myY_test[idx[0:n_predict]]), " mistakes from ", n_predict)
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

#%%
unX_train = np.vstack((myX_train, X_train));
unY_train = np.hstack((myY_train, y_train));
unX_test  = np.vstack((X_test, myX_test));
unY_test  = np.hstack((y_test, myY_test));

from sklearn.utils import shuffle

unX_train, unY_train = shuffle(unX_train, unY_train)

unX_test, unY_test = shuffle(unX_test, unY_test)

unSvm = cv_svm (unX_train, unX_test, unY_train, unY_test)

print('Test Accuracy of unSvm on my data = ', round(score(unSvm,myX_test, myY_test), 4))
print('Test Accuracy of unSvm on  u data = ', round(score(unSvm,X_test, y_test), 4))

