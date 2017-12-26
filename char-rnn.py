
import tensorflow as tf
import numpy as np
from textloader import TextLoader
from tensorflow.python.ops.rnn_cell import BasicLSTMCell, MultiRNNCell
from tensorflow.python.ops.rnn_cell import RNNCell
from tensorflow.contrib.legacy_seq2seq import sequence_loss, rnn_decoder
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.contrib.rnn.ops import gen_gru_ops

class mygru(RNNCell):
	def __init__( self, size_in ):
		self.size = size_in

	@property
	def state_size(self):
		return self.size

	@property
	def output_size(self):
		return self.size

	def __call__(self, inputs, state, scope=None):
		with vs.variable_scope(scope or type(self).__name__):
			input_size = inputs.get_shape().with_rank(2)[1]
			cell_size = state.get_shape().with_rank(2)[1]

			b_c = vs.get_variable("b_c", [self.size], initializer = init_ops.constant_initializer(0.0))
			W_c = vs.get_variable("W_c", [input_size + self.size, self.size])

			b_ru = vs.get_variable("b_ru", [self.size * 2], initializer = init_ops.constant_initializer(1.0))
			W_ru = vs.get_variable("W_ru", [input_size + self.size, self.size * 2])

			_, _, _, h = gen_gru_ops.gru_block_cell(x = inputs, h_prev = state, b_c = b_c, w_c = W_c, b_ru = b_ru, w_ru = W_ru)
			return h, h

#
# -------------------------------------------
#
# Global variables

batch_size = 50
sequence_length = 50

data_loader = TextLoader( ".", batch_size, sequence_length )

vocab_size = data_loader.vocab_size	# dimension of one-hot encodings
state_dim = 128

num_layers = 2

tf.reset_default_graph()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# define placeholders for our inputs.
# in_ph is assumed to be [batch_size,sequence_length]
# targ_ph is assumed to be [batch_size,sequence_length]

in_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='inputs' )
targ_ph = tf.placeholder( tf.int32, [ batch_size, sequence_length ], name='targets' )
in_onehot = tf.one_hot( in_ph, vocab_size, name="input_onehot" )

inputs = tf.split( in_onehot, sequence_length, axis=1 )
inputs = [ tf.squeeze(input_, [1]) for input_ in inputs ]
targets = tf.split( targ_ph, sequence_length, axis=1 )

# at this point, inputs is a list of length sequence_length
# each element of inputs is [batch_size,vocab_size]

# targets is a list of length sequence_length
# each element of targets is a 1D vector of length batch_size

with tf.name_scope("rnn_base") as scope:
	# lstm1 = BasicLSTMCell(state_dim)
	# lstm2 = BasicLSTMCell(state_dim)
	lstm1 = mygru(state_dim)
	lstm2 = mygru(state_dim)
	rnn = MultiRNNCell([lstm1, lstm2])
	initial_state = rnn.zero_state(batch_size, tf.float32)

with tf.name_scope("decoder") as scope:
	outputs, final_state = rnn_decoder(inputs, initial_state, rnn)
	W = tf.Variable(tf.random_normal([state_dim, vocab_size], stddev = 0.02))
	b = tf.Variable(tf.random_normal([vocab_size], stddev = 0.01))
	logits = [tf.matmul(output, W) + [b] for output in outputs]
	loss_w = [1.0 for i in range(sequence_length)]
	loss = sequence_loss(logits=logits, targets=targets, weights=loss_w)
	optim = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

with tf.name_scope("sampler") as scope:
	s_in_ph = tf.placeholder(tf.int32, [1], name='s_inputs')
	s_in_onehot = tf.one_hot(s_in_ph, vocab_size, name="s_input_onehot")
	s_inputs = tf.split(s_in_onehot, 1, axis=1)
	s_initial_state = rnn.zero_state(1, tf.float32)
	s_outputs, s_final_state = rnn_decoder(s_inputs, s_initial_state, rnn)
	s_logits = [tf.matmul(s_output, W) + [b] for s_output in s_outputs]

#
# ==================================================================
# ==================================================================
# ==================================================================
#

# with tf.name_scope("gru") as scope:
# 	#TODO use Hadamard product instead of *
# 	#TODO what it h_{t-1} ?
# 	#TODO what are the Ws?
# 	#TODO what are the Us?
# 	#TODO what are the bs?
# 	#TODO how to use this GRU?
# 	update_gate = tf.sigmoid(W_z * input_vector + U_z * h + b_z, name="update_gate")
# 	reset_gate = tf.sigmoid(W_r * input_vector + U_r * h + b_r, name="reset_gate")
# 	output_vec = update_gate * h + (1 - update_gate) * tf.tanh(W_h * input_vector + U_h * (reset_gate * h) + b_h)

def sample( num=200, prime='ab' ):

	# prime the pump

	# generate an initial state. this will be a list of states, one for
	# each layer in the multicell.
	s_state = sess.run( s_initial_state )

	# for each character, feed it into the sampler graph and
	# update the state.
	for char in prime[:-1]:
		x = np.ravel( data_loader.vocab[char] ).astype('int32')
		feed = { s_in_ph:x }
		for i, s in enumerate( s_initial_state ):
			feed[s] = s_state[i]
		s_state = sess.run( s_final_state, feed_dict=feed )

	# now we have a primed state vector; we need to start sampling.
	ret = prime
	char = prime[-1]
	for n in range(num):
		x = np.ravel( data_loader.vocab[char] ).astype('int32')

		# plug the most recent character in...
		feed = { s_in_ph:x }
		for i, s in enumerate( s_initial_state ):
			feed[s] = s_state[i]
		ops = [s_logits]
		ops.extend( list(s_final_state) )

		retval = sess.run( ops, feed_dict=feed )

		s_probsv = retval[0]
		s_state = retval[1:]

		# ...and get a vector of probabilities out!

		# now sample (or pick the argmax)
		sample = np.argmax( s_probsv[0] )
		# sample = np.random.choice( vocab_size, p=s_probsv[0] )

		pred = data_loader.chars[sample]
		ret += pred
		char = pred

	return ret

#
# ==================================================================
# ==================================================================
# ==================================================================
#

sess = tf.Session()
sess.run( tf.global_variables_initializer() )
summary_writer = tf.summary.FileWriter( "./tf_logs", graph=sess.graph )

lts = []

print "FOUND %d BATCHES" % data_loader.num_batches

for j in range(1000):

	state = sess.run( initial_state )
	data_loader.reset_batch_pointer()

	for i in range( data_loader.num_batches ):

		x,y = data_loader.next_batch()

		# we have to feed in the individual states of the MultiRNN cell
		feed = { in_ph: x, targ_ph: y }
		for k, s in enumerate( initial_state ):
			feed[s] = state[k]

		ops = [optim,loss]
		ops.extend( list(final_state) )

		# retval will have at least 3 entries:
		# 0 is None (triggered by the optim op)
		# 1 is the loss
		# 2+ are the new final states of the MultiRNN cell
		retval = sess.run( ops, feed_dict=feed )

		lt = retval[1]
		state = retval[2:]

		if i%1000==0:
			print "%d %d\t%.4f" % ( j, i, lt )
			lts.append( lt )

	print sample( num=60, prime="The " )
	# print sample( num=60, prime="And " )
	# print sample( num=60, prime="ababab" )
	# print sample( num=60, prime="foo ba" )
	# print sample( num=60, prime="abcdab" )

summary_writer.close()

#
# ==================================================================
# ==================================================================
# ==================================================================
#

#import matplotlib
#import matplotlib.pyplot as plt
#plt.plot( lts )
#plt.show()
