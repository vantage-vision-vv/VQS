import tensorflow as tf
import numpy as np

num_verbs = 	#V
num_nouns = 	#N
time_steps = 	#T
no_of_roles = 	#R


hidden_state_dim =   #H 
input_feature_dim =  #I


class GGNN():
	def __init__(self):
		self.graph = tf.Graph()
		self.sess = tf.Session(graph=self.graph)
		self.placeholders = {}
		self.weights = {}
		self.role_nodes = {}
		self.biases = {}
		self.embeddings = {}
		self.transform = {}
			
	def One_hot_encoder(total , pos):
		tmp = np.zeros((total,))
		tmp[pos] = 1
		return tf.convert_to_tensor(tmp)
	
	def Rnn():
		X_total = tf.get_variable("X_total_temp",shape=[hidden_state_dim,1])
		for i in range(no_of_roles):
			X_total = tf.add(tf.matmul(self.weights['message'],self.role_nodes[i]),X_total)
		X_total = tf.add(tf.matmul(self.weights['message'],self.verb_node),X_total)
		X_total = tf.add(self.biases['message'],X_total)
		for i in range(no_of_roles):
			X_temp = tf.subtract(X_total,tf.matmul(self.weights['message'],self.role_nodes[i]))
			Z_t = tf.nn.relu()
		
		
							
		

	def model(self,v,r,verb_encoding):
		self.placeholders['verb'] = v
		self.placeholders['noun'] = n
		self.verb_node = tf.nn.relu(tf.matmul(self.transform['verb'],self.placeholder['verb']))
		for i in range(no_of_roles):
			self.role_nodes[i] = tf.nn.relu(tf.multiply(tf.multiply(tf.matmul(self.transform['noun'],self.placeholder['verb']),tf.matmul(self.embeddings['noun'],One_hot_encoder(no_of_roles,i))),tf.matmul(self.embeddings['noun'],verb_encoding)))
		for time in range(time_steps):	# Try scan
			Rnn()
		

		
		
	def initialize(self):
		self.placeholders['verb'] = tf.placeholders("float",[input_feature_dim]) # I x 1
		self.placeholders['noun'] = tf.placeholders("float",[no_of_rolesinput_feature_dim]) # R x I		
		for i in range(no_of_roles):
			var_name = hidden_state_ + str(i)
			self.role_nodes[i] = tf.get_variable(var_name,shape_name=[hidden_state_dim])	# H
		self.verb_node = tf.get_variable(var_name,shape_name=[hidden_state_dim]		# H
		self.embeddings['verb'] = tf.get_variable("verb_embedding",shape=[hidden_state_dim,num_verbs],initializer=tf.contrib.layers.xavier_initializer()) # H x V
		self.embeddings['noun'] = tf.get_variable("noun_embedding",shape=[hidden_state_dim,num_nouns],initializer=tf.contrib.layers.xavier_initializer()) # H x N
		self.transform['verb'] = tf.get_variable("verb_transform",shape=[hidden_state_dim,input_feature_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x I
		self.transform['noun'] = tf.get_variable("noun_transform",shape=[hidden_state_dim,input_feature_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x I
		self.weights['message'] = tf.get_variable("message",shape=[hidden_state_dim,hidden_state_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x H
		self.weights['W_z'] = tf.get_variable("W_z",shape=[hidden_state_dim,hidden_state_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x H
		self.weights['W_r'] = tf.get_variable("W_r",shape=[hidden_state_dim,hidden_state_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x H
		self.weights['W_h'] = tf.get_variable("W_h",shape=[hidden_state_dim,hidden_state_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x H
		self.weights['U_z'] = tf.get_variable("U_z",shape=[hidden_state_dim,hidden_state_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x H
		self.weights['U_r'] = tf.get_variable("U_r",shape=[hidden_state_dim,hidden_state_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x H
		self.weights['U_h'] = tf.get_variable("U_h",shape=[hidden_state_dim,hidden_state_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x H
		self.biases['B_z'] = tf.get_variable("B_z",shape=[hidden_state_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x 1
		self.biases['B_r'] = tf.get_variable("B_r",shape=[hidden_state_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x 1
		self.biases['B_h'] = tf.get_variable("B_h",shape=[hidden_state_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x 1
		self.biases['message'] = tf.get_variable("message",shape=[hidden_state_dim],initializer=tf.contrib.layers.xavier_initializer()) # H x 1








		
	
	def train(self):
	
	def save_model(self):
