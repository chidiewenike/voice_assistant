
import numpy as np
import random
import pickle
from collections import Counter
import os

global path # global variable containing the path to the data directory

# The lexicon is developed using the words within the files names of 
# the matrix data. The number ID is stripped from the file name and
# added to the lexicon array
def develop_lexicon():

	# empty lexicon array to be filled
	lexicon = []

	# pulls the list of files within the directory
	listing = os.listdir(path)

	# iterates through the list of files within the directory
	for fileName in listing:
		print ("current file is:" + fileName)
		inputFile = path + fileName

		# the File ID numbers are stripped from the file name
		word = ''.join([i for i in fileName if not i.isdigit()])

		# removes the .txt from files
		word = word.replace('.txt','')

		# older data samples had 'Mat' so it removes that string
		word = word.replace('Mat','')

		# if the word is not already within the lexicon
		if not (word.lower() in lexicon):
			# the name will be added to the lexicon
			lexicon.append(word.lower())

	# Unknown is the last word appended to the lexicon
	lexicon.append("Unknown")		
	print(lexicon)

	return lexicon

# 
def sample_handling(lexicon):

	# empty list to contain the Sxx 1D data
	featureset = []

	# pulls the list of files within the directory
	listing = os.listdir(path)

	# iterates through the list of files within the directory
	for fileName in listing:
		inputFile = path + fileName
		features = np.loadtxt(inputFile)
		features = features.flatten()
		word = ''.join([i for i in fileName if not i.isdigit()])
		
		# removes the .txt from files
		word = word.replace('.txt','')

		# older data samples had 'Mat' so it removes that string
		word = word.replace('Mat','')

		# the classification array is initialized to the size
		# of the lexicon with all zeros
		classification = np.zeros(len(lexicon))

		# if the processed is in the lexicon
		if word.lower() in lexicon:

			# the position of the lexicon array is found and assigned the value '1'
			index_value = lexicon.index(word.lower())
			classification[index_value] += 1
			classification = list(classification)

			# the classification vector is then paired with its
			# corresponding Sxx matrix and appended to the feature set
			featureset.append([features,classification])
			print("Word: " + word)
			print(lexicon)
			print(classification)

		# the last classification is made for unknown
		else:
			classification[len(lexicon) - 1] += 1
			classification = list(classification)
			featureset.append([features,classification])		

	return featureset


# the function which creates the feature set 
# containing the Sxx data and classification data
def create_feature_sets_and_labels(test_size = 0.1):

	# calls the function which creates the lexicon
	# and returns the created lexicon
	lexicon = develop_lexicon()   

	# the feature set
	features = []

	# the Sxx and classification set is added to features
	# array by calling the sample handling function 
	# with the lexicon passed as an argument
	features += sample_handling(lexicon)
	random.shuffle(features)
	features = np.array(features)

	# determined test size is calculated
	testing_size = int(test_size*len(features))

	# training and test data is assigned to each corresponding
	# variable
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])
	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x,train_y,test_x,test_y,lexicon



if __name__ == '__main__':

	# prompts the user for the path of the data folder
	path = input("Enter the FULL path to the folder in which all of data is located follow by a \." + "\n"
             + "Make sure that there are only text files in this folder: ")

	# the training and test data are taken with the lexicon
	train_x,train_y,test_x,test_y,lexicon = create_feature_sets_and_labels()
	
	# and a pickle of the data is created for the model
	with open('voice_data_set.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y,lexicon],f)
