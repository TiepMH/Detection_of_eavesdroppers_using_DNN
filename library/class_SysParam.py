import pickle


class SystemParameters:
    def __init__(self, No_Attack=True):
        self.snrdB_Bob = 10
        self.snrdB_Eve = 4
        ###
        self.SNR_Bob = 10**(self.snrdB_Bob/10)
        self.SNR_Eve = 10**(self.snrdB_Eve/10)
        self.list_of_SNRs = [self.SNR_Bob, self.SNR_Eve]
        ###
        self.n_Rx = 10
        self.n_Tx = self.num_of_Tx(No_Attack)
        ###
        self.DOA_Bob = - 40.5  # from -90 degree to +90 degree
        self.DOA_Eve = None  # from -90 degree to +90 degree
        self.list_of_DOAs = self.DOAs_of_Tx(No_Attack)
        self.num_angles = 180
        ###
        self.Rician_factor = 2.0
        self.n_NLOS_paths = 10
        self.max_delta_theta = 60.0  # 60 degree

    def num_of_Tx(self, No_Attack):
        if No_Attack is True:
            n_Tx = 1  # Bob
        else:
            n_Tx = 2  # Bob and Eve
        return n_Tx

    def DOAs_of_Tx(self, No_Attack):
        if No_Attack is True:
            list_of_DOAs = [self.DOA_Bob]
        else:
            self.DOA_Eve = float( input('Enter the DOA of Eve: ') )
            list_of_DOAs = [self.DOA_Bob, self.DOA_Eve]
        return list_of_DOAs


# =============================================================================
""" Save the system parameters as an object for later use """
# SysParam = SystemParameters(No_Attack=False)
# # save the SysParam object as a pickle-type file
# with open('input/mySysParam.pickle', 'wb') as f:
#     pickle.dump(SysParam, f)

# # load the pickle-type file
# with open('input/mySysParam.pickle', 'rb') as f:
#     SysParam = pickle.load(f)
