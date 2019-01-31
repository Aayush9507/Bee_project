import glob
import os
import random

import cv2
import numpy as np
from scipy.fftpack import fft
from scipy.io import wavfile

"""Folder Lists"""
bee_test_folderlist = []
no_bee_test_folderlist = []
bee_train_folderlist = []
no_bee_train_folderlist = []

"""Image Lists & Audio Lists"""
bee_test_images =[]
nobee_test_images =[]
bee_train_images =[]
no_bee_train_images =[]
processed_bee_test_images =[]
processed_nobee_test_images =[]
processed_bee_train_images =[]
processed_no_bee_train_images =[]
# Waves
buzz_bee_test_waves = []
buzz_bee_train_waves = []
buzz_cricket_train_waves = []
buzz_cricket_test_waves =[]
buzz_noise_test_waves = []
buzz_noise_train_waves = []
processed_buzzbee_test_waves = []
processed_buzzbee_train_waves = []
processed_buzzcricket_test_waves = []
processed_buzzcricket_train_waves = []
processed_buzznoise_test_waves = []
processed_buzznoise_train_waves = []

"""Path Lists"""
# Bee2set
bee_test_path = os.getcwd() + "/BEE2Set/bee_test"
no_bee_test_path = os.getcwd() + "/BEE2Set/no_bee_test"
bee_train_path = os.getcwd() + "/BEE2Set/bee_train"
no_bee_train_path = os.getcwd() + "/BEE2Set/no_bee_train"
# Buzz2set
buzz_bee_test_path = os.getcwd() + "/BUZZ2Set/test/bee_test/"
buzz_cricket_test_path = os.getcwd() + "/BUZZ2Set/test/cricket_test/"
buzz_noise_test_path = os.getcwd() + "/BUZZ2Set/test/noise_test/"
buzz_bee_train_path = os.getcwd() + "/BUZZ2Set/train/bee_train/"
buzz_cricket_train_path = os.getcwd() + "/BUZZ2Set/train/cricket_train/"
buzz_noise_train_path = os.getcwd() + "/BUZZ2Set/train/noise_train/"

"""---------------------------------buzzBee test waves--------------------------------"""


def process_bee_test_waves():
    waves = glob.glob(buzz_bee_test_path +"*.wav")
    for names in waves:
        if not names.startswith('.'):
            buzz_bee_test_waves.append(names)
    for waves1 in buzz_bee_test_waves:
        samplerate, audio = wavfile.read(waves1)
        audio = fft(audio)
        audio = audio/float(np.max(audio))
        processed_buzzbee_test_waves.append(audio)
    return processed_buzzbee_test_waves


"""---------------------------------buzzBee train waves--------------------------------"""


def process_bee_train_waves():
    waves = glob.glob(buzz_bee_train_path +"*.wav")
    for nmes2 in waves:
        if not nmes2.startswith('.'):
            buzz_bee_train_waves.append(nmes2)
    for waves2 in buzz_bee_train_waves:
        samplerate, audio = wavfile.read(waves2)
        audio = fft(audio)
        audio = audio/float(np.max(audio))
        processed_buzzbee_train_waves.append(audio)
    return processed_buzzbee_train_waves


"""---------------------------------buzzCricket test waves--------------------------------"""


def process_cricket_test_waves():
    waves = glob.glob(buzz_cricket_test_path +"*.wav")
    for nmes3 in waves:
        if not nmes3.startswith('.'):
            buzz_cricket_test_waves.append(nmes3)

    for waves3 in buzz_cricket_test_waves:
        samplerate, audio = wavfile.read(waves3)
        audio = fft(audio)
        audio = audio/float(np.max(audio))
        processed_buzzcricket_test_waves.append(audio)
    return processed_buzzcricket_test_waves


"""---------------------------------buzzCricket train waves--------------------------------"""


def process_cricket_train_waves():
    waves = glob.glob(buzz_cricket_train_path +"*.wav")
    for nmes4 in waves:
        if not nmes4.startswith('.'):
            buzz_cricket_train_waves.append(nmes4)

    for waves4 in buzz_cricket_train_waves:
        samplerate, audio = wavfile.read(waves4)
        audio = fft(audio)
        audio= audio/float(np.max(audio))
        processed_buzzcricket_train_waves.append(audio)
    return processed_buzzcricket_train_waves


"""---------------------------------buzzNoise test waves--------------------------------"""


def process_noise_test_waves():
    waves = glob.glob(buzz_noise_test_path +"*.wav")
    for nmes5 in waves:
        if not nmes5.startswith('.'):
            buzz_noise_test_waves.append(nmes5)

    for waves5 in buzz_noise_test_waves:
        samplerate, audio = wavfile.read(waves5)
        audio = fft(audio)
        audio= audio/float(np.max(audio))
        processed_buzznoise_test_waves.append(audio)
    return processed_buzznoise_test_waves


"""---------------------------------buzzNoise train waves--------------------------------"""


def process_noise_train_waves():
    waves = glob.glob(buzz_noise_train_path +"*.wav")
    for nmes6 in waves:
        if not nmes6.startswith('.'):
            buzz_noise_train_waves.append(nmes6)

    for waves6 in buzz_noise_train_waves:
        samplerate, audio = wavfile.read(waves6)
        audio = fft(audio)
        audio = audio/float(np.max(audio))
        processed_buzznoise_train_waves.append(audio)
    return processed_buzznoise_train_waves


"""---------------------------------bee test images--------------------------------"""


for x in os.listdir(bee_test_path):
    if not x.startswith('.'):
        bee_test_folderlist.append(x)
for i in bee_test_folderlist:
    imgnames = sorted(glob.glob(bee_test_path + "/" + i + "/*.png"))
    for nmes in imgnames:
        if not nmes.startswith('.'):
            bee_test_images.append(nmes)
for images in bee_test_images:
    img = cv2.imread(images)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image / 255.0
    data = np.array(scaled_gray_image)
    flattened_image = data.flatten()
    processed_bee_test_images.append(flattened_image)

"""---------------------------------no bee test images--------------------------"""


for x in os.listdir(no_bee_test_path):
    if not x.startswith('.'):
        no_bee_test_folderlist.append(x)
for i in no_bee_test_folderlist:
    imgnames = sorted(glob.glob(no_bee_test_path + "/" + i + "/*.png"))
    for nmes in imgnames:
        if not nmes.startswith('.'):
            nobee_test_images.append(nmes)
for images in nobee_test_images:
    img = cv2.imread(images)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image / 255.0
    data = np.array(scaled_gray_image)
    flattened_image = data.flatten()
    processed_nobee_test_images.append(flattened_image)

"""---------------------------------bee train images-----------------------------"""


for x in os.listdir(bee_train_path):
    if not x.startswith('.'):
        bee_train_folderlist.append(x)
for i in bee_train_folderlist:
    imgnames = sorted(glob.glob(bee_train_path + "/" + i + "/*.png"))
    for nmes in imgnames:
        if not nmes.startswith('.'):
            bee_train_images.append(nmes)

for images in bee_train_images:
    img = cv2.imread(images)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image / 255.0
    data = np.array(scaled_gray_image)
    flattened_image = data.flatten()
    processed_bee_train_images.append(flattened_image)

"""---------------------------------No bee train images---------------------------"""


for x in os.listdir(no_bee_train_path):
    if not x.startswith('.'):
        no_bee_train_folderlist.append(x)
for i in no_bee_train_folderlist:
    imgnames = sorted(glob.glob(no_bee_train_path + "/" + i + "/*.png"))
    for nmes in imgnames:
        if not nmes.startswith('.'):
            no_bee_train_images.append(nmes)
for images in no_bee_train_images:
    img = cv2.imread(images)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scaled_gray_image = gray_image / 255.0
    data = np.array(scaled_gray_image)
    flattened_image = data.flatten()
    processed_no_bee_train_images.append(flattened_image)



def shuffling(data, label):
    totalsamples = data.shape[0]
    n = np.random.permutation(totalsamples)
    return data[n], label[n]

def shuffle_data(a, b):
    c = list(zip(a.tolist(), b.tolist()))
    random.shuffle(c)
    a, b = zip(*c)
    return np.asarray(a), np.asarray(b)


def load_image_data():

    training_data = np.asarray(processed_bee_train_images+processed_no_bee_train_images)
    testing_data = np.asarray(processed_bee_test_images+processed_nobee_test_images)
    Label1_train = np.ones(len(processed_bee_train_images), dtype=int)
    Label0_train = np.zeros(len(processed_no_bee_train_images), dtype=int)
    training_labels = np.concatenate((Label1_train, Label0_train), axis=None)
    Train_data = (training_data, training_labels)
    Label1_test = np.ones(len(processed_bee_test_images), dtype=int)
    Label0_test = np.zeros(len(processed_nobee_test_images), dtype=int)
    testing_labels = np.concatenate((Label1_test, Label0_test), axis=None)
    Test_data0 = (testing_data, testing_labels)
    Validation_data = (Test_data0[0][11001:], Test_data0[1][11001:])
    Test_data = (Test_data0[0][0:11001], Test_data0[1][0:11001])
    return shuffling(training_data, training_labels), shuffling(Test_data0[0][11001:], Test_data0[1][11001:]),shuffling(Test_data0[0][0:11001], Test_data0[1][0:11001])


def extract_neurons(data):
    """Utility function to extract chunks of audio from beginning,middle and end"""
    reduced_data = []
    for ls in range(len(data)):
        reduced_data.append(np.concatenate((data[ls][:2000],data[ls][20000:22400],data[ls][-2000:]), axis=None))
    return reduced_data


def load_audio_data():
    """This function loads processed audio data , shuffles it and return it """
    processed_buzzbee_test_waves = process_bee_test_waves()
    processed_buzzbee_train_waves = process_bee_train_waves()
    processed_buzzcricket_test_waves = process_cricket_test_waves()
    processed_buzzcricket_train_waves = process_cricket_train_waves()
    processed_buzznoise_test_waves = process_noise_test_waves()
    processed_buzznoise_train_waves = process_noise_train_waves()

    training_data = np.asarray(processed_buzzbee_train_waves+processed_buzznoise_train_waves+processed_buzzcricket_train_waves)
    testing_data = np.asarray(processed_buzzbee_test_waves+processed_buzznoise_test_waves+processed_buzzcricket_test_waves)
    training_data2 = extract_neurons(training_data)
    testing_data2 = extract_neurons(testing_data)

    Label11_train = np.ones(len(processed_buzzbee_train_waves), dtype=int)
    Label12_train = np.zeros(len(processed_buzznoise_train_waves), dtype=int)
    Label13_train = np.repeat(2, len(processed_buzzcricket_train_waves))

    training_labels = np.concatenate((Label11_train, Label12_train, Label13_train), axis=None)

    Label11_test = np.ones(len(processed_buzzbee_test_waves), dtype=int)
    Label12_test = np.zeros(len(processed_buzznoise_test_waves), dtype=int)
    Label13_test = np.repeat(2, len(processed_buzzcricket_test_waves))

    testing_labels = np.concatenate((Label11_test, Label12_test, Label13_test), axis=None)

    Test_data0 = (np.asarray(testing_data2), testing_labels)
    return shuffling(np.asarray(training_data2), training_labels), shuffling(Test_data0[0][1601:], Test_data0[1][1601:]), shuffling(Test_data0[0][0:1600], Test_data0[1][0:1600])


def vectorized_result(j):
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e


def vectorized_result_audio(j):
    e = np.zeros((3, 1))
    e[j] = 1.0
    return e


def load_data_wrapper_images():
    tr_d, va_d, te_d = load_image_data()
    training_inputs = [np.reshape(x, (1024, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (1024, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (1024, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


def load_data_wrapper_audio():
    tr_d, va_d, te_d = load_audio_data()
    neurons = len(tr_d[0][0])
    training_inputs = [np.reshape(x, (neurons, 1)) for x in tr_d[0]]
    training_results = [vectorized_result_audio(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (neurons, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (neurons, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


