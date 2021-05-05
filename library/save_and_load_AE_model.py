import numpy as np
from library.class_AE import get_compiled_model


def save_my_model(model):
    path_to_my_model = './results/trained_model/model_weights'
    model.save_weights(path_to_my_model, save_format='tf')
    return None


def load_my_model(n_input):
    path_to_my_model = './results/trained_model/model_weights'
    XX = np.random.rand(1, n_input)
    """ initiate a new model before loading the saved model"""
    loaded_model = get_compiled_model(n_input)
    loaded_model.fit(XX, XX, verbose=False)
    """ we will load the saved model to the newly-created model """
    loaded_model.load_weights(path_to_my_model)
    return loaded_model