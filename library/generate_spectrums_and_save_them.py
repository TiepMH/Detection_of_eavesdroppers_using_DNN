# https://dengjunquan.github.io/posts/2018/08/DoAEstimation_Python/
# https://github.com/dengjunquan/DoA-Estimation-MUSIC-ESPRIT

import numpy as np
import pickle
from library.class_MUSIC_spectrum_without_Eve import MUSIC_spectrum_without_Eve
from library.class_MUSIC_spectrum_with_Eve import MUSIC_spectrum_with_Eve
from library.class_SysParam import SystemParameters

import os
from pathlib import Path
cur_path = os.path.abspath(os.getcwd())  # path to library


""" No_Attack = True    >>>    There is no attack from any eavesdropper

    No_Attack = False   >>>    An eavesdropper is attacking the network """


# =============================================================================
def generate_spectrums(No_Attack):
    """ Create system parameters and save them """
    SysParam = SystemParameters(No_Attack)
    # save the SysParam object as a pickle-type file
    with open(os.path.join(cur_path, 'input/mySysParam.pickle'), 'wb') as temp:
        pickle.dump(SysParam, temp)
    """ Load system parameters """
    list_of_SNRs = SysParam.list_of_SNRs
    n_Rx = SysParam.n_Rx
    n_Tx = SysParam.n_Tx
    list_of_DOAs = SysParam.list_of_DOAs  # from -90 degree to +90 degree
    num_angles = SysParam.num_angles
    kappa = SysParam.Rician_factor
    n_NLOS_paths = SysParam.n_NLOS_paths
    max_delta_theta = SysParam.max_delta_theta
    ###
    table = np.empty([0, num_angles])
    num_windows = 1000
    for i in range(num_windows):
        if No_Attack is False:  # WITH Eve
            mySpectrum = MUSIC_spectrum_with_Eve(list_of_SNRs, n_Rx, n_Tx,
                                                 num_angles, list_of_DOAs,
                                                 kappa, n_NLOS_paths, max_delta_theta)
        if No_Attack is True:  # WITHOUT Eve
            mySpectrum = MUSIC_spectrum_without_Eve(list_of_SNRs, n_Rx, n_Tx,
                                                    num_angles, list_of_DOAs,
                                                    kappa, n_NLOS_paths, max_delta_theta)
        DoAs_MUSIC, spectrum_dB = mySpectrum.music()
        # DoAs_MUSIC is of integer type
        hv, powers = mySpectrum.correlation()
        #
        if i == 0:
            mySpectrum.plot_fig()

        # Dump spectrum into the table that will be then saved as csv file
        table = np.append(table,
                          np.reshape(spectrum_dB, [1, num_angles]),
                          axis=0)
    ###
    # table.shape = [num_windows, num_angles]
    # Append the label column to the existing table
    if No_Attack is False:  # WITH Eve: labels = 1
        y_label = np.ones([num_windows, 1], dtype='int')
    if No_Attack is True:  # WITHOUT Eve: labels = 0
        y_label = np.zeros([num_windows, 1], dtype='int')
    ###
    table = np.hstack((table, y_label))
    # Now, table.shape = [num_windows, num_angles+1]
    if No_Attack is False:  # WITH Eve: labels = 1
        # Save the table as csv
        np.savetxt('input/MUSIC_spectrums_label_1.csv', table, delimiter=',')
    if No_Attack is True:  # WITHOUT Eve: labels = 0
        # Save the table as csv
        np.savetxt('input/MUSIC_spectrums_label_0.csv', table, delimiter=',')
    return None


# =============================================================================
