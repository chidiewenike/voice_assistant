# voice_assistant
This is the voice assistant I created. 

# RecordToSpectrogram.py 
Allows for data collection. Audio is saved in 3 second segments and converted to a spectrogram. The spectrogram and raw audio wav files are 
saved in their respective directories.

# Preprocess.py
Takes the spectrogram intensity data, creates a lexicon of words within the data, and the one-hot encodes each array of data. The data and
labeling is saved in a pickle for training.

# VoiceModel.py
The model takes the pickle and creates the neural network using Tensorflow. 
