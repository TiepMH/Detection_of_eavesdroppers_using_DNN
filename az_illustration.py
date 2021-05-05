import numpy as np
from sklearn.metrics import auc
import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score
from library.save_and_load_AE_model import load_my_model
from library.mean_and_std_of_imgs import mean_of_imgs, std_of_imgs
from library.performance_metrics import detection_metrics

from library.plot_original_and_decoded_imgs import plot_a_normal_img_and_its_reconstruction
from library.plot_original_and_decoded_imgs import plot_an_anomalous_img_and_its_reconstruction
from library.plot_mean_and_var_of_imgs import plot_mean_and_std_of_anomalous_imgs
from library.plot_mean_and_var_of_imgs import plot_mean_and_std_of_normal_imgs
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
""" Plot the average imgs along with their associated stds """
plt.figure(1)
plot_mean_and_std_of_anomalous_imgs(anomalous_imgs_train, angles)
plot_mean_and_std_of_normal_imgs(normal_imgs_train, angles)
# plt.ylim(ymin=0, ymax=0.5)

""" Calculate the AUC for average curves """
# avg_anomalous_img = mean_of_imgs(anomalous_imgs_train)
# AUC_for_avg_anomalous_img = auc(angles, avg_anomalous_img)
# #
# avg_normal_img = mean_of_imgs(normal_imgs_train)
# AUC_for_avg_normal_img = auc(angles, avg_normal_img)
# #
# print("AUC for the average 'anomalous' curve: ", AUC_for_avg_anomalous_img)
# print("AUC for the average 'normal' curve: ", AUC_for_avg_normal_img)

# ============================================================================
""" Load the trained AE model """
model = load_my_model(num_angles)

""" Plot a normal img, then plot its reconstruction """
normal_imgs_test_encoded = model.encoder(normal_imgs_test).numpy()
normal_imgs_test_decoded = model.decoder(normal_imgs_test_encoded).numpy()
#
plt.figure(2)
plot_a_normal_img_and_its_reconstruction(normal_imgs_test[0],
                                         normal_imgs_test_decoded[0],
                                         angles)
plt.ylim(ymax=0.1)

""" Plot an anomalous img, then plot its reconstruction """
anomalous_imgs_test_encoded = model.encoder(anomalous_imgs_test).numpy()
anomalous_imgs_test_decoded = model.decoder(anomalous_imgs_test_encoded).numpy()
#
plt.figure(3)
plot_an_anomalous_img_and_its_reconstruction(anomalous_imgs_test[3],
                                             anomalous_imgs_test_decoded[3],
                                             angles)
plt.ylim(ymax=0.1)

# ============================================================================
"""
SSD = sum_of_squared_differences
"""
normal_imgs_train_encoded = model.encoder(normal_imgs_train).numpy()
normal_imgs_train_decoded = model.decoder(normal_imgs_train_encoded).numpy()

###
std = std_of_imgs(normal_imgs_train_decoded)
threshold = np.linalg.norm(std)**2

###
acc, TPR, FPR, FNR, TNR = detection_metrics(threshold,
                                            normal_imgs_train_decoded,
                                            normal_imgs_test_decoded,
                                            anomalous_imgs_test_decoded)

###
print("accuracy =", acc)
print("sensitivity =", TPR)
print("specificity =", TNR)

