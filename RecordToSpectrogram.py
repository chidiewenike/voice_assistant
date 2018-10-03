
import pyaudio
import wave
import time
import random
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os
import scipy.fftpack
import struct

# CONSTANTS
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2 
RATE = 3000
RECORD_SECONDS = 3

#_______________________

def recordAudio (fileName, path, audPath):
        # the input number of recordings for each word 
        recordNum = int(input("\nHow many times would you like to record the word \"" + fileName + "\" ? "))
        recCount = 1
        recCountStr = "1"
        remainStr = str(recordNum)
        fileNameTemp = fileName

        # continuously loop recording until you reach the number of input recordings 
        while (recCount <= recordNum): 
                p = pyaudio.PyAudio()
                stream = p.open(format = FORMAT, channels = CHANNELS, 
                                        rate = RATE, input = True, frames_per_buffer = CHUNK)
                print("\n* Recordings for \"" + fileName + "\" Remaining: " + remainStr)
                inputVal = str(input("\nType \"r\" to record next data sample or \"p\" to record the previous sample "))
                
                # input "r" to record the next recording
                if (inputVal == "r"):
                        recordDummyAudio()
                        remainStr = str(recordNum - recCount)

                        # countdown for before the recording starts
                        print("\n***Recording Data Sample #" + recCountStr + " In... ***\n")
                        time.sleep(1)
                
                        print("*** 2 ***")

                        time.sleep(1)
                        print("*** 1 ***\n")

                        time.sleep(1)

                        # the start of the audio recording
                        print("***Started Recording***")

                        frames = []
                        for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
                                data = stream.read(CHUNK)
                                frames.append(data)
                        
                        # completes the recording
                        print("\n* done recording")
                        stream.stop_stream()
                        stream.close()
                        p.terminate()

                        # creates a unique ID for each data entry and saves the .wav file
                        fileID = str(random.randint(0,9999999999999))
                        fileNameTemp += (fileID + ".wav")

                        # dumps the recording into the Audio Data file
                        wf = wave.open(audPath + "\\" + fileNameTemp, 'wb')
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(p.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))
                        wf.close

                        print("\n**File: " + fileNameTemp + " has been recorded")
                        
                        # resets values and increments the counter
                        fileNameTemp = fileName
                        recCount+=1
                        recCountStr = str(recCount)

                        # sends the wav file to be converted to the spectrogram matrix
                        ConvertToSpect(fileName + fileID, path)

                # input 'p' if you would like to record the previous sample        
                elif (inputVal == "p"):
                        remainStr = str(recordNum - recCount + 2)
                        print("\n***Redoing the last sample***\n")
                        print("\n* Recordings for \"" + fileName + "\" Remaining: " + remainStr)
                        recordDummyAudio()
                        recCount-=1
                        recCountStr = str(recCount)
                        remainStr = str(recordNum - recCount)
                        print("\n***Recording Data Sample #" + recCountStr + " In... ***\n")
                        time.sleep(1)
                
                        print("*** 2 ***")

                        time.sleep(1)
                        print("*** 1 ***\n")

                        time.sleep(1)
                        print("\n***Started Recording***\n")

                        frames = []
                        for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
                                data = stream.read(CHUNK)
                                frames.append(data)
                        
                        print("\n* done recording")
                        stream.stop_stream()
                        stream.close()
                        p.terminate()

                        fileNameTemp += (fileID + ".wav")
                        wf = wave.open(fileNameTemp, 'wb')
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(p.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        wf.writeframes(b''.join(frames))
                        wf.close

                        print("\n**File: " + fileNameTemp + " has been recorded\n")
                        
                        fileNameTemp = fileName
                        recCount+=1
                        recCountStr = str(recCount)
                        ConvertToSpect(fileName+fileID, path)

                elif inputVal == "?":
                        print("*** exiting ***")
                        break  
                else:
                        print("Invalid input. Try again.")

def ConvertToSpect(fileName, path):
    print("\nCurrent File: " + fileName)
    audio_sig = getAudioData(fileName)
    saveSpectrogramData(audio_sig, fileName, path)
    print("\nSpectrogram Data Matrix for " + fileName + " is saved in " + os.getcwd() + "\MatrixData\\\n")

def getAudioData(fileName):
    #open the wave file
    wf = wave.open(os.getcwd() + '\\' + 'WavAudioData' + '\\' + fileName + '.wav', 'rb')
    #get the packed bytes
    raw_sig = wf.readframes(wf.getnframes())
    #convert the packed bytes into integers
    audio_sig = np.fromstring(raw_sig, 'Int16')
    wf.close()#close the file
    return audio_sig

def saveSpectrogramData(data, fileName, matPath):
    #get the full directory for the location of the png file
    directoryMat = matPath + fileName + ".txt"

    # converts the wav file to a spectrogram
    f, t, Sxx = signal.spectrogram(data, RATE)

    # converts the Sxx array to a Numpy array
    Sxx = np.array(Sxx)

    # flattens the 2D array to 1D
    Sxx = Sxx.flatten()

    # saves the Sxx matrix data to 
    np.savetxt(directoryMat, Sxx, fmt='%s')

def recordDummyAudio():
	#the mic would produce a wierd thumping for the first audio clip recorded
	#records and doesnot save a dummy clip to get rid of that thumping
	
	p = pyaudio.PyAudio()
	stream = p.open(format = FORMAT, channels = CHANNELS, 
				rate = RATE, input = True, frames_per_buffer = CHUNK)
	
	frames = []
	for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)
	
	stream.stop_stream()
	stream.close()
	p.terminate()

# takes the current directory of the python program
current_directory = os.getcwd()

# opens the directories to contain the audio and matrix data
mat_spec_path = os.path.join(current_directory, 'MatrixData\\')
aud_spec_path = os.path.join(current_directory, 'WavAudioData\\')

print ("When recording remember to speak clearly and loudly." + "\n"
		+ "Also try to keep the background noise to a minimum")

if not os.path.exists(mat_spec_path):
    #create the MatrixData folder if it doesn't already exist
    os.makedirs(mat_spec_path)
          
if not os.path.exists(aud_spec_path):
    #create the WavAudioData folder if it doesn't already exist
    os.makedirs(aud_spec_path)


recordDummyAudio()
fileName = "start"
while 1:
        fileName = input("\nEnter the word you will speak or enter ? to exit: ")

        if(fileName == "?"):
								
                print("*** exiting ***")
                break

        # invalid if the string is empty
        elif(fileName == ""):
                print("Error: Enter a valid name.")

        # invalid if the string contains only white space
        elif(fileName.isspace()):
                print("Error: Enter a valid name.")       

        else:
                print ("Will record \"" + fileName + "\"")
                recordAudio(fileName, mat_spec_path, aud_spec_path)
	
		
