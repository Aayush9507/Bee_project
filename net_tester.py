import pickle as cPickle
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected

from data_loader import *


def extract_neurons(data):
    reduced_data = []
    for ls in range(len(data)):
        reduced_data.append(np.concatenate((data[ls][:2000],data[ls][20000:20900],data[ls][-2000:]), axis=None))
    return reduced_data

def extract_neurons2(data):
    reduced_data = []
    for ls in range(len(data)):
        reduced_data.append(np.concatenate((data[ls][:2000],data[ls][20000:22400],data[ls][-2000:]), axis=None))
    return reduced_data

def load(file_name):
    with open(file_name,'rb') as fp:
        nn = cPickle.load(fp)
    return nn

def fit_image_ann(ann, image_path):
    thresh = 0.5
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image / 255.0
    data = np.array(scaled_gray_image)
    flattened_image = data.flatten()
    flattened_image = np.reshape(flattened_image,(1024,1))
    print ann.feedforward(flattened_image)
    if (ann.feedforward(flattened_image))[1]<thresh:
        return [0,1]
    else:
        return [1,0]

def fit_image_cnn(cnn,image_path):
    thresh = 1
    img = cv2.imread(image_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image / 255.0
    data = np.array(scaled_gray_image)
    flattened_image = data.flatten()
    flattened_image = np.reshape(flattened_image,(-1, 32, 32, 1))
    print cnn.predict(flattened_image)
    if (cnn.predict(flattened_image))[0][0]<thresh:
        return [1,0]
    else:
        return [0,1]


def fit_audio_ann(ann, audio_path):

    mylist=[]
    samplerate, audio = wavfile.read(audio_path)
    audio = fft(audio)
    audio = audio/float(np.max(audio))
    mylist.append(audio)
    audio = extract_neurons2(np.asarray(mylist))
    audio = np.reshape(audio, (-1,80,80, 1))
    y_hat =  ann.predict(audio)
    prediction = np.zeros_like(y_hat)
    prediction[0][(np.argmax(y_hat))] = 1.0
    return prediction


def fit_audio_cnn(cnn,audio_path):
    mylist=[]
    tc = 0.9
    samplerate, audio = wavfile.read(audio_path)
    audio = fft(audio)
    audio = audio/float(np.max(audio))
    mylist.append(audio)
    audio = extract_neurons(np.asarray(mylist))
    audio = np.reshape(audio,(-1, 70, 70, 1))
    print cnn.predict(audio)
    if (cnn.predict(audio))[0][2] > tc:
        return [0,1,0]
    elif (cnn.predict(audio))[0][1] >tc:
        return [1,0,0]
    else:
        return [0,0,1]


def load_imageconvnet(path):

    input_layer = input_data(shape=[None, 32, 32, 1])
    conv_layer = conv_2d(input_layer, nb_filter=16, filter_size=3,
                         activation='relu', name='conv_layer_1',bias=0)
    pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_1')
    conv_layer2 = conv_2d(pool_layer, nb_filter=32, filter_size=3,
                          activation='relu', name='conv_layer_2',bias=0)
    pool_layer2 = max_pool_2d(conv_layer2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer2, 256, activation='relu', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 2, activation='relu', name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(path)
    return model


def load_audioconvnet(path):
    input_layer = input_data(shape=[None, 70, 70, 1])
    conv_layer = conv_2d(input_layer, nb_filter=16, filter_size=3,
                         activation='relu', name='conv_layer_1',regularizer='L2')
    pool_layer = max_pool_2d(conv_layer, 2, name='pool_layer_1')
    conv_layer2 = conv_2d(pool_layer, nb_filter=32, filter_size=3,
                          activation='relu', name='conv_layer_2',regularizer='L2')
    pool_layer2 = max_pool_2d(conv_layer2, 2, name='pool_layer_2')

    fc_layer_1 = fully_connected(pool_layer2, 64, activation='relu', name='fc_layer_1')
    fc_layer_2 = fully_connected(fc_layer_1, 3, activation='softmax', name='fc_layer_2')
    model = tflearn.DNN(fc_layer_2)
    model.load(path)
    return model


def load_audio_ann(path):
    input_layer = tflearn.input_data(shape=[None, 80, 80, 1])
    hidden1 = tflearn.fully_connected(input_layer, 50, activation='ReLu',
                                      regularizer='L2')
    hidden2 = tflearn.fully_connected(hidden1, 60, activation='ReLu',
                                      regularizer='L2')
    hidden3 = tflearn.fully_connected(hidden2, 3, activation='softmax',
                                      regularizer='L2')

    # Training
    model = tflearn.DNN(hidden3)
    model.load(path)
    return model



audio_path = '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/BUZZ2Set/test/bee_test/bee2294_192_168_4_9-2017-06-30_13-45-01.wav'
img_path = '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/BEE2Set/no_bee_test/img5/192_168_4_5-2017-05-12_18-38-06_192_32_6.png'

img_ann_path = '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/imageANNpck/Project1_ISnet_1024_160_301_2_10_15.pck'
img_cnn_path = '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/ImageCNNpck/imgcnn.tfl'

audio_ann_path = '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/audioAnnpck/AudioANN.tfl'
audio_cnn_path = '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/AudioCNNpck/BCNA.tfl'

valid_x_path_img = '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/cnn_pck/valid_x.pck'
valid_y_path_img = '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/cnn_pck/valid_y.pck'
validX_img = load(valid_x_path_img)
validY_img = load(valid_y_path_img)

valid_x_path_aud = '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/audioAnnpck/validx.pck'
valid_y_path_aud = '/Users/mymac/IdeaProjects/IntelligentSystems/Project1_IS/audioAnnpck/validy.pck'
validX_aud = load(valid_x_path_aud)
validY_aud = load(valid_y_path_aud)

"""Run Image ANN"""
# print(fit_image_ann(load(img_ann_path), img_path))

"""Run Image CNN"""
# conv_accuracy = fit_image_cnn(load_imageconvnet(img_cnn_path), img_path)
# print(conv_accuracy)
# tf.reset_default_graph()

"""Run Audio CNN"""
# conv_accuracy = fit_audio_cnn(load_audioconvnet(audio_cnn_path), audio_path)
# print(conv_accuracy)
# tf.reset_default_graph()

"""Run Audio ANN"""
tf.reset_default_graph()
ann_accuracy = fit_audio_ann(load_audio_ann(audio_ann_path), audio_path)
print(ann_accuracy)
