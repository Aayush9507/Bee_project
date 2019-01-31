from __future__ import division, print_function

import pickle as cPickle
import random

import numpy as np
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from data_loader import *
from network import Network


def save(obj, file_name):
    with open(file_name, 'wb') as fp:
        cPickle.dump(obj, fp)

def nn_arch_naming(netsize, eta, minibatch):
    return 'net'+'_'+'_'.join(map(str, netsize))+'_'+str(int(eta*100))+'_'+str(minibatch)+".pck"


"""=================================ARTIFICIAL NEURAL NETS======================================"""

eta_vals = (0.1, 0.25, 0.3, 0.4, 0.5)
mini_batch_sizes = (5, 10, 15, 20)

"""Image Nets"""
net1 = Network([1024, 120, 2])
net2 = Network([1024, 160, 301, 2])


def train_image_ann(network, eta_vals, mini_batch_sizes, num_epochs, path):
    """This function loads training,testing and validation data from load_data_wrapper_images()
     which is imported from data_loader.py.
    Image ANN takes the network list, eta values list , mini batch size list , number of
    epochs and path where trained network should be persisted as parameters.Save function
    inside will save the trained model to specified path with its name as it's architecture"""

    train_d, valid_d, test_d = load_data_wrapper_images()
    #Image Nets
    print("inside image train")
    for nets in network:
        eta = eta_vals[random.randrange(len(eta_vals))]
        mini_batch = mini_batch_sizes[random.randrange(len(mini_batch_sizes))]
        nets.SGD(train_d, num_epochs, mini_batch, eta, test_d)
        name = path+nn_arch_naming(nets.sizes, eta, mini_batch)
        save(nets, name)

# def train_audio_ann(network, eta_vals, mini_batch_sizes, num_epochs, path):
#     """This function loads training,testing and validation data from load_data_wrapper_audio()
#      which is imported from data_loader.py.
#     Image ANN takes the network list, eta values list , mini batch size list , number of
#     epochs and path where trained network should be persisted as parameters.Save function
#     inside will save the trained model to specified path with its name as it's architecture"""
#     #Audio Nets
#     train_d, valid_d, test_d = load_data_wrapper_audio()
#     print("inside audio train")
#     for nets in network:
#         eta = eta_vals[random.randrange(len(eta_vals))]
#         mini_batch = mini_batch_sizes[random.randrange(len(mini_batch_sizes))]
#         nets.SGD(train_d, num_epochs, mini_batch, eta, test_d)
#         name = path+nn_arch_naming(nets.sizes, eta, mini_batch)
#         save(nets, name)


"""=================================CONVOLUTION NEURAL NETS======================================"""


def train_image_cnn():
    """This function will load Training,Testing and Validation data from load_image_data() imported from data_loader.py

    Architecture:
    2 convolution layers , 2 fully connected layers , 2 Pooling layers
    Learning rate = 0.002
    Batch Size = 3
    Epochs = 30

    Training:
    Just call image_cnn() will start training and edit path in model.save() inside this function
    to save trained model in required folder.

     """
    train_data, valid_data, test_data = load_image_data()

    training_results = [vectorized_result(y) for y in train_data[1]]
    test_results = [vectorized_result(y) for y in test_data[1]]

    train_X, train_Y = np.asarray(train_data[0]), np.asarray(training_results)
    test_X, test_Y = np.asarray(test_data[0]), np.asarray(test_results)

    train_Y = np.reshape(train_Y, (-1, 2))
    test_Y = np.reshape(test_Y, (-1, 2))

    valid_X, valid_Y = np.asarray(valid_data[0]), np.asarray(valid_data[1])

    train_X = train_X.reshape([-1, 32, 32, 1])
    test_X = test_X.reshape([-1, 32, 32, 1])

    save(valid_X, '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/cnn_pck/valid_x.pck')
    save(valid_Y, '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/cnn_pck/valid_y.pck')
    input_layer = input_data(shape=[None, 32, 32, 1])
    conv_layer = conv_2d(input_layer, nb_filter=16, filter_size=3,
                         activation='relu', name='conv_layer_1',bias=0)
    pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_1')
    conv_layer2 = conv_2d(pool_layer, nb_filter=32, filter_size=3,
                          activation='relu', name='conv_layer_2',bias=0)
    pool_layer2 = max_pool_2d(conv_layer2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer2, 256, activation='relu', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2, activation='relu', name='fc_layer_2')

    network = regression(fc_layer_2, optimizer='sgd', loss='binary_crossentropy', learning_rate=0.01)
    network = dropout(network, 0.5)
    model = tflearn.DNN(network)
    model.fit(train_X, train_Y, n_epoch=30, shuffle=True, validation_set=(test_X, test_Y), show_metric=True, batch_size=10, run_id='BeeImage')
    model.save('/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/ImageCNNpck/imgcnn.tfl')
    return model

def train_audio_ann():
    """This function will load Training,Testing and Validation data from load_audio_data() imported from data_loader.py

    Architecture:
    3 Hidden layers

    Training:
    Just call audio_ann() will start training and edit path in model.save() inside this function
    to save trained model in required folder.
    Training start will take some time because FFT is performed on data so it will start after 6 red warnings.

     """

    train_data, valid_data, test_data = load_audio_data()

    training_results = [vectorized_result_audio(y) for y in train_data[1]]
    test_results = [vectorized_result_audio(y) for y in test_data[1]]

    train_X, train_Y = np.asarray(train_data[0]),np.asarray(training_results)
    test_X, test_Y = np.asarray(test_data[0]),np.asarray(test_results)

    train_Y = np.reshape(train_Y, (-1, 3))
    test_Y = np.reshape(test_Y, (-1, 3))

    valid_X, valid_Y = np.asarray(valid_data[0]), np.asarray(valid_data[1])
    save(valid_X, '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/audioAnnpck/validx.pck')
    save(valid_Y, '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/audioAnnpck/validy.pck')
    train_X = train_X.reshape([-1, 80, 80, 1])
    test_X = test_X.reshape([-1, 80, 80, 1])
    print ("Data loaded")
    input_layer = tflearn.input_data(shape=[None, 80, 80, 1])
    hidden1 = tflearn.fully_connected(input_layer, 50, activation='ReLu',
                                      regularizer='L2')
    hidden2 = tflearn.fully_connected(hidden1, 60, activation='ReLu',
                                      regularizer='L2')
    hidden3 = tflearn.fully_connected(hidden2, 3, activation='softmax',
                                      regularizer='L2')
    net = tflearn.regression(hidden3, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')

    # Training
    model = tflearn.DNN(net)
    model.fit(train_X, train_Y, n_epoch=100, validation_set=(test_X, test_Y),show_metric=True, run_id="AudioANN")
    model.save('AudioANN.tfl')

def train_audio_cnn():
    """This function will load Training,Testing and Validation data from load_audio_data() imported from data_loader.py

    Architecture:
    2 convolution layers , 2 fully connected layers , 2 Pooling layers
    Learning rate = 0.001
    Batch Size = 10
    Epochs = 10

    Training:
    Just call audio_cnn() will start training and edit path in model.save() inside this function
    to save trained model in required folder.
    Training start will take some time because FFT is performed on data so it will start after 6 red warnings.

     """
    train_data, valid_data, test_data = load_audio_data()

    training_results = [vectorized_result_audio(y) for y in train_data[1]]
    test_results = [vectorized_result_audio(y) for y in test_data[1]]

    train_X, train_Y = np.asarray(train_data[0]),np.asarray(training_results)
    test_X, test_Y = np.asarray(test_data[0]),np.asarray(test_results)

    train_Y = np.reshape(train_Y, (-1, 3))
    test_Y = np.reshape(test_Y, (-1, 3))

    valid_X, valid_Y = np.asarray(valid_data[0]), np.asarray(valid_data[1])
    train_X = train_X.reshape([-1, 70, 70, 1])
    test_X = test_X.reshape([-1, 70, 70, 1])

    save(valid_X, '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/AudioCNNpck/validx.pck')
    save(valid_Y, '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/AudioCNNpck/validy.pck')
    input_layer = input_data(shape=[None, 70, 70, 1])
    conv_layer = conv_2d(input_layer, nb_filter=16, filter_size=3,
                         activation='relu', name='conv_layer_1',regularizer='L2')
    pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_1')
    conv_layer2 = conv_2d(pool_layer, nb_filter=32, filter_size=3,
                          activation='relu', name='conv_layer_2',regularizer='L2')
    pool_layer2 = max_pool_2d(conv_layer2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer2, 64, activation='relu', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3, activation='softmax', name='fc_layer_2')


    network = regression(fc_layer_2, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)
    network = dropout(network, 0.5)
    model = tflearn.DNN(network)
    model.fit(train_X, train_Y, n_epoch=4, validation_set=(test_X, test_Y), show_metric=True, batch_size=2, run_id='BCNAudio')
    model.save('/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/cnn_pck22/BCNA.tfl')
    return model


"""Method Calling"""

# train_image_cnn()
# train_audio_cnn()
# train_image_ann(networks2, eta_vals, mini_batch_sizes, 50, "/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/imageANNpck")
# train_audio_ann()



