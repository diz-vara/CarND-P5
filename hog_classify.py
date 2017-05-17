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
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
#from sklearn.cross_validation import train_test_split


conversions = {'HSV':cv2.COLOR_RGB2HSV,
               'HLS':cv2.COLOR_RGB2HLS,
               'LUV':cv2.COLOR_RGB2LUV,
               'YUV':cv2.COLOR_RGB2YUV,
               'YCrCb':cv2.COLOR_RGB2YCrCb,
               'LAB':cv2.COLOR_RGB2LAB,
               'Lab':cv2.COLOR_RGB2Lab,
               'XYZ':cv2.COLOR_RGB2XYZ
               }
    

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
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
        image = mpimg.imread(file)
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
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        elif hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True)
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
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC 
svc = svm.SVC(C=2.3, kernel = 'rbf', gamma = 6.5e-4)

#%%             
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

#%%
# Check the prediction time for a single sample

idx = np.arange(len(y_test))
np.random.shuffle(idx)

t=time.time()
n_predict = 1000
yr = svc.predict(X_test[idx[0:n_predict]])
print(sum( yr != y_test[idx[0:n_predict]]), " mistakes from ", n_predict)
#print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

#%%

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
do_fit = False;

if (do_fit):
    parameters = {'C': np.arange(2.2, 2.4, .05), 'gamma': np.arange(6.4e-4, 6.6e-4, 1e-5)}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    np.random.shuffle(idx)
    X_tr = X_train[idx[:1000]]
    y_tr = y_train[idx[:1000]]
    
    clf.fit(X_tr, y_tr)
    
    print(clf.best_params_, clf.best_score_)

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
scaled_X = X_scaler.transform(myX)

# Define the labels vector
myY = np.hstack((np.ones(len(myCarFeatures)), np.zeros(len(myNotCarFeatures))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
myX_train, myX_test, myY_train, myY_test = train_test_split(
    scaled_X, myY, test_size=0.2, random_state=rand_state)
#%%
mySvc = svm.SVC(C=2.3, kernel = 'rbf', gamma = 6.5e-4)

           
# Check the training time for the SVC
t=time.time()
mySvc.fit(myX_train, myY_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of mySVC on my data = ', round(mySvc.score(myX_test, myY_test), 4))
print('Test Accuracy of uSVC on my data  = ', round(svc.score(myX_test, myY_test), 4))
print('Test Accuracy of mySVC on u data = ', round(mySvc.score(X_test, y_test), 4))
print('Test Accuracy of uSVC on u data  = ', round(svc.score(X_test, y_test), 4))

#%%
idx = np.arange(len(myY_test))
np.random.shuffle(idx)

t=time.time()
n_predict = 1000
yr = svc.predict(myX_test[idx[0:n_predict]])
#print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(sum( yr != myY_test[idx[0:n_predict]]), " mistakes from ", n_predict)
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

t=time.time()
yr = mySvc.predict(myX_test[idx[0:n_predict]])
#print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(sum( yr != myY_test[idx[0:n_predict]]), " mistakes from ", n_predict)
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')


