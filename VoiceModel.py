import tensorflow as tf
import pickle
import numpy as np
import os
train_x,train_y,test_x,test_y,lexicon = pickle.load( open( "voice_data_set.pickle", "rb" ) )

n_nodes_hl1 = 1500
n_nodes_hl2 = 1500
n_nodes_hl3 = 1500
n_nodes_hl4 = 1500

# number of outputs
n_classes = len(lexicon)

# batch sizes 
batch_size = 100

# number of passes forward and back
hm_epochs = 3

x = tf.placeholder('float')
y = tf.placeholder('float')

# four hidden layers of the NN
hidden_1_layer = {'=':n_nodes_hl1,
                  'weight':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl1]))}

hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl2]))}

hidden_3_layer = {'f_fum':n_nodes_hl3,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl3]))}

hidden_4_layer = {'f_fum':n_nodes_hl4,
                  'weight':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                  'bias':tf.Variable(tf.random_normal([n_nodes_hl4]))}

# output layer of the NN
output_layer = {'f_fum':None,
                'weight':tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                'bias':tf.Variable(tf.random_normal([n_classes])),}


# Nothing changes
def neural_network_model(data):

	# performs the dot product to each layer with the weights of the layer
	# and the matrix is added to the bias
	l1 = tf.add(tf.matmul(data,hidden_1_layer['weight']), hidden_1_layer['bias'])

	# the resulting matrix is then passed through the "ReLu" activation function
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1,hidden_2_layer['weight']), hidden_2_layer['bias'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2,hidden_3_layer['weight']), hidden_3_layer['bias'])
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3,hidden_4_layer['weight']), hidden_4_layer['bias'])
	l4 = tf.nn.relu(l4)

	output = tf.matmul(l4,output_layer['weight']) + output_layer['bias']

	return output

# 
def train_neural_network(x):

	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y) )
	optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		File_writer = tf.summary.FileWriter('C:\\Users\\cej17\\Downloads\\Cpp-Ai_Data_ML-master\\Cpp-Ai_Data_ML-master\\Data\\graph',sess.graph)

		for epoch in range(hm_epochs):
			epoch_loss = 0
			i=0
			while i < len(train_x):
				start = i
				end = i+batch_size
				batch_x = np.array(train_x[start:end])
				batch_y = np.array(train_y[start:end])

				_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
				                                              y: batch_y})
				epoch_loss += c
				i+=batch_size 				
			print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		# empty array for the test data input
		test = []
		path = os.getcwd()
		# the file containing the test data
		path = path + "\\PomonaMat.txt"
		test = np.loadtxt(path)
		test = test.flatten()
		test = np.array(test)

		# the test data is input for evaluation using the model
		classification = sess.run(tf.argmax(prediction.eval(feed_dict={x:[test]}),1))
		print((tf.argmax(prediction.eval(feed_dict={x:[test]}))))
		print(classification.item())

		# the array output of the classification array is tested with the lexicon
		print(lexicon[classification.item()])
		print('Accuracy:',(accuracy.eval({x:test_x, y:test_y})))
  
train_neural_network(x)
