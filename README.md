Training_nets.py:

To start training Audio ANN and CNN just call train_audio_ann() and train_audio_cnn()
It will start in some time because FFT is performed on audio so after 6 red warnings training will start

 To start Image Ann and can just call train_image_ann() with networks2 , eta_vals , Mini batch size , epochs, path where trained model will be persisted and to start Image can just call train_image_cnn()


data_loader.py:

This file loads the data from directory. To load data just put bee2set and buzz2set in current directory.
For audio there are 6 functions which will load and process audio data and these functions are called inside load_audio_data()
data_loader is imported in training_nets.py and some functions in net_tester.py


net_tester.py

In this function just change the respective paths to run fit functions.
