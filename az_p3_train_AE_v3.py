import matplotlib.pyplot as plt
import numpy as np
from library.class_AE import get_compiled_model
from library.save_and_load_AE_model import save_my_model
from library.class_SysParam import SystemParameters
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use CPU

###
anomalous_imgs_train = np.load('input/imgs_train_anomalous.npy')
normal_imgs_train = np.load('input/imgs_train_normal.npy')
imgs_test = np.load('input/imgs_test.npy')

# ============================================================================
""" System Paremeters"""
# SysParam = SystemParameters(No_Attack=False)
# # save the SysParam object as a pickle-type file
# with open(f'input/mySysParam.pickle', 'wb') as file:
#     pickle.dump(SysParam, file)

# load the pickle-type file
with open('input/mySysParam.pickle', 'rb') as file:
    SysParam = pickle.load(file)

###
n_Rx = SysParam.n_Rx
n_Tx = SysParam.n_Tx
list_of_DOAs = SysParam.list_of_DOAs  # from -90 degree to +90 degree
num_angles = SysParam.num_angles

###
angles = np.linspace(-num_angles/2, num_angles/2, num_angles)

# ============================================================================
plt.grid()
plt.plot(angles, anomalous_imgs_train[0])
plt.title("An Anomalous Spectrum")
plt.show()

""" Plot an normal SPECTRUM """
plt.grid()
plt.plot(angles, normal_imgs_train[0])
plt.title("A Normal Spectrum")
plt.show()

# =============================================================================
""" Compile and train the model """
model = get_compiled_model(num_angles)
history = model.fit(normal_imgs_train,
                    normal_imgs_train,
                    epochs=100, batch_size=2**6,
                    validation_data=(imgs_test, imgs_test),
                    shuffle=True, verbose=False)

# =============================================================================
""" Plot the loss function during training """
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# =============================================================================
""" Save the trained AE model """
save_my_model(model)
del model
