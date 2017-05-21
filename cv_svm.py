# -*- coding: utf-8 -*-
"""
Created on Wed May 17 21:53:28 2017

@author: diz
"""
import numpy as np
import cv2

    

# Set up SVM from OpenCV 3
def cv_svm (X_train, X_test, y_train, y_test):
    C=0.8
    kernel = 'rbf'
    gamma = 6.5e-4

    t=time.time()
    
    svm = cv2.ml.SVM_create()
    # Set SVM type
    svm.setType(cv2.ml.SVM_C_SVC)
    # Set SVM Kernel to Radial Basis Function (RBF) 
    svm.setKernel(cv2.ml.SVM_RBF)
    # Set parameter C
    svm.setC(C)
    # Set parameter Gamma
    svm.setGamma(gamma)
     
    # Train SVM on training data  
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)

    t2 = time.time()
     
    # Save trained model 
    svm.save("./models/u_svm_model.yml");
     
    # Test on a held out test set
    testResponse = svm.predict(X_test)[1].ravel()
    accuracy = 1-sum(np.abs(testResponse-y_test))/y_test.size


    print(round(t2-t, 2), 'Seconds to train cv2.SVM...')
    # Check the score of the SVC
    print('Test Accuracy of cv2.SVM = ', round(accuracy, 4))
    return svm
    
def score (svm, X_test, y_test):
    testResponse = svm.predict(X_test)[1].ravel()
    accuracy = 1-sum(np.abs(testResponse-y_test))/y_test.size
    return accuracy
#%%
idx = np.arange(len(y_test))
np.random.shuffle(idx)

t=time.time()
n_predict = 1000
yr = uSvm.predict(X_test[idx[0:n_predict]].astype(np.float32))[1].ravel()
print(sum( yr != y_test[idx[0:n_predict]]), " mistakes from ", n_predict)
#print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
