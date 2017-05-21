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

from extract_features import *
from process_image import *
import time



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
myNotCars = get_image_list(myPath + 'NotCars')

#%%
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
    scaled_X, myY, test_size=0.1, random_state=rand_state)
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

#%%
unSvm = cv_svm (unX_train, unX_test, unY_train, unY_test)

print('Test Accuracy of unSvm on my data = ', round(score(unSvm,myX_test, myY_test), 4))
print('Test Accuracy of unSvm on  u data = ', round(score(unSvm,X_test, y_test), 4))

#%%
#grid-search for optimal SVM parameters

from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
do_fit = False;

if (do_fit):
    parameters = {'C': np.arange(0.5, 2.4, .1), 'gamma': np.arange(6.4e-4, 6.6e-4, 1e-5)}
    svr = svm.SVC()
    clf = GridSearchCV(svr, parameters)
    np.random.shuffle(idx)
    X_tr = unX_train[idx[:1000]]
    y_tr = unY_train[idx[:1000]]
    
    clf.fit(X_tr, y_tr)
    
    print(clf.best_params_, clf.best_score_)

#%%-
from moviepy.editor import VideoFileClip

thr = 18
tau = 0.95
scales = [3, 4, 5]
heat = np.zeros_like(img[:,:,0]).astype(np.float)
scales = [3, 4, 5]

  
video_output = 'out/p5C08sc%d_%.2f_%.1f.mp4' % (345, tau, thr)
clip = VideoFileClip('project_video.mp4')#.subclip(30,45)
#clip = VideoFileClip('E:\\Data\\USA\\Video\\cuts\\multiple_01.avi') 
first_clip = clip.fl_image(process_image)
get_ipython().magic('time first_clip.write_videofile(video_output, audio=False)')


