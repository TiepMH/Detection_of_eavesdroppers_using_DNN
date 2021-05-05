import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score
from library.save_and_load_AE_model import load_my_model
from library.plot_original_and_decoded_imgs import plot_a_normal_img_and_its_reconstruction
from library.plot_original_and_decoded_imgs import plot_an_anomalous_img_and_its_reconstruction
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use CPU

# ============================================================================
anomalous_imgs_train = np.load('input/imgs_train_anomalous.npy')
anomalous_imgs_test = np.load('input/imgs_test_anomalous.npy')
normal_imgs_train = np.load('input/imgs_train_normal.npy')
normal_imgs_test = np.load('input/imgs_test_normal.npy')
imgs_test = np.load('input/imgs_test.npy')

# ============================================================================
""" System Paremeters"""
with open('input/mySysParam.pickle', 'rb') as f:
    SysParam = pickle.load(f)

num_angles = SysParam.num_angles
angles = np.linspace(-num_angles/2, num_angles/2, num_angles)

# ============================================================================
""" Load the trained AE model """
loaded_model = load_my_model(num_angles)

""" Plot a normal img, then plot its reconstruction """
normal_imgs_encoded = loaded_model.encoder(normal_imgs_test).numpy()
normal_imgs_decoded = loaded_model.decoder(normal_imgs_encoded).numpy()
#
plot_a_normal_img_and_its_reconstruction(normal_imgs_test[0], 
                                         normal_imgs_decoded[0],
                                         angles)


""" Plot an anomalous img, then plot its reconstruction """
anomalous_imgs_encoded = loaded_model.encoder(anomalous_imgs_test).numpy()
anomalous_imgs_decoded = loaded_model.decoder(anomalous_imgs_encoded).numpy()
#
plot_an_anomalous_img_and_its_reconstruction(anomalous_imgs_test[110],
                                             anomalous_imgs_decoded[110],
                                             angles)

# ============================================================================
"""Detect anomalies"""
