import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from lesson_functions import *
import time
import cv2
import glob
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

def classify():
    # Read in our vehicles and non-vehicles
    images = glob.glob('vehicles/*/*.png')
    cars = []
    notcars = []

    for image in images:
        cars.append(image)

    images = glob.glob('non-vehicles/*/*.png')

    for image in images:
        notcars.append(image)   
    
    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 11  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (64, 64) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [None, None] # Min and max in y to search in slide_window()
    
    car_example = cv2.imread(cars[0])
    notcar_example = cv2.imread(notcars[0])

    plt.imshow(car_example)
    plt.show()
    plt.imshow(notcar_example)
    plt.show()
    car_example = cv2.cvtColor(car_example, cv2.COLOR_RGB2YCrCb)
    hog,image_hog = get_hog_features(car_example[:,:,2], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=True, feature_vec=True)
    plt.imshow(image_hog, cmap = "gray")
    plt.show()
    notcar_example = cv2.cvtColor(notcar_example, cv2.COLOR_RGB2YCrCb)
    hog,image_hog = get_hog_features(notcar_example[:,:,2], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=True, feature_vec=True)
    plt.imshow(image_hog, cmap = "gray")
    plt.show()
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, 
                            hist_feat=hist_feat, hog_feat=hog_feat, spatial_feat=spatial_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, 
                            hist_feat=hist_feat, hog_feat=hog_feat, spatial_feat=spatial_feat)

    car_features = np.asarray(car_features)
    notcar_features = np.asarray(notcar_features)
    print(car_features.shape)
    if len(car_features) > 0:
        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features))
        print(X.shape)
        X = X.astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    print(X_train.shape)
    print(y_train.shape)
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')

    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    return svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins


svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins = classify()

import pickle
with open("svc_pickle.p", 'wb') as pickle_file:
    s = pickle.dump({
            "svc" : svc,
            "X_scaler" : X_scaler,
            "orient" : orient,
            "pix_per_cell" : pix_per_cell,
            "cell_per_block" : cell_per_block,
            "spatial_size" : spatial_size,
            "hist_bins" : hist_bins
        },pickle_file)