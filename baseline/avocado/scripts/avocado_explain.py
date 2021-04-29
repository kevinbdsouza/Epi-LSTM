from __future__ import division

import os    
os.environ['THEANO_FLAGS'] = "device=cuda{}".format(os.environ['SGE_HGR_cuda'])    
import theano
import numpy

from avocado import *
from avocado.io import *
from avocado.models import *

import keras
import keras.backend as K
from keras.layers import Input, Dense, Flatten, concatenate
from keras.models import Model, Sequential
import time

import sys, numpy
numpy.random.seed(0)

import numpy as np
from time import sleep
import time
from tqdm import tqdm

attr_dir = '/net/noble/vol5/user/jmschr/proj/avocado/attributions'

'''
Integrated gradients approximates Shapley values by integrating partial
gradients with respect to input features from reference input to the
actual input. The following class implements the paper "Axiomatic attribution
for deep neuron networks". Based off the implementation by Naozumi Hiranuma
at https://github.com/hiranumn/IntegratedGradients.
'''
class integrated_gradients:
	# train_fns: Keras train_fns that you wish to explain.
	# outchannels: In case the train_fns are multi tasking, you can specify which output you want explain .
	def __init__(self, model, samples, step_sizes, outchannels=[], verbose=1):
	
		#get backend info (either tensorflow or theano)
		self.backend = K.backend()
		self.samples = samples
		self.step_sizes = step_sizes
		
		#load train_fns supports keras.Model and keras.Sequential
		if isinstance(model, Sequential):
			self.model = model.model
		elif isinstance(model, Model):
			self.model = model
		else:
			print("Invalid input train_fns")
			return -1
		
		#load input tensors
		self.input_tensors = []
		for i in self.model.inputs:
			self.input_tensors.append(i)
		# The learning phase flag is a bool tensor (0 = test, 1 = train)
		# to be passed as input to any Keras function that uses 
		# a different behavior at train time and test time.
		self.input_tensors.append(K.learning_phase())
		
		#If outputchanels are specified, use it.
		#Otherwise evalueate all outputs.
		self.outchannels = outchannels
		if len(self.outchannels) == 0: 
			if verbose: print("Evaluated output channel (0-based index): All")
			if K.backend() == "tensorflow":
				self.outchannels = range(self.model.output.shape[1]._value)
			elif K.backend() == "theano":
				self.outchannels = range(self.model.output._keras_shape[1])
		else:
			if verbose: 
				print("Evaluated output channels (0-based index):")
				print(','.join([str(i) for i in self.outchannels]))
				
		#Build gradient functions for desired output channels.
		self.get_gradients = {}
		if verbose: print("Building gradient functions")
		
		# Evaluate over all requested channels.
		for c in self.outchannels:
			# Get tensor that calculates gradient
			if K.backend() == "tensorflow":
				gradients = self.model.optimizer.get_gradients(self.model.output[:, c], self.model.input)
			if K.backend() == "theano":
				gradients = self.model.optimizer.get_gradients(self.model.output[:, c].sum(), self.model.input)
				
			# Build computational graph that computes the tensors given inputs
			self.get_gradients[c] = K.function(inputs=self.input_tensors, outputs=gradients)
			
			# This takes a lot of time for a big train_fns with many tasks.
			# So lets print the progress.
			if verbose:
				sys.stdout.write('\r')
				sys.stdout.write("Progress: "+str(int((c+1)*1.0/len(self.outchannels)*1000)*1.0/10)+"%")
				sys.stdout.flush()
		# Done
		if verbose: print("\nDone.")
			
				
	'''
	Input: sample to explain, channel to explain
	Optional inputs:
		- reference: reference values (defaulted to 0s).
		- steps: # steps from reference values to the actual sample (defualted to 50).
	Output: list of numpy arrays to integrated over.
	'''
	def explain(self, X, num_steps=50):
		n, d = X.shape
		k = 0
		step_size = 1.0 / num_steps


		for i in range(n):
			self.step_sizes[i, 288:] = X[i] * step_size

			for j in range(num_steps):
				self.samples[k, 288:] = X[i] * (j+1) * step_size
				#self.samples[k] = X[i] * (j+1) * step_size
				k += 1

		_input = [self.samples, 0]
		tic = time.time()
		gradients = self.get_gradients[0](_input)

		explanations = []
		for i in range(n):
			start, end = i * num_steps, (i+1) * num_steps
			explanation = gradients[start:end].sum(axis=0) * self.step_sizes[i]
			explanations.append(explanation)

		return numpy.array(explanations)


	def gradients(self, X):
		gradients = self.get_gradients[0]([X, 0])
		return gradients



def AvocadoNN():
	x = Input(shape=(110+32+256,), name='X')
	x_in = x

	for i in range(2):
		x = Dense(2048, activation='relu')(x)

	y = Dense(1)(x)

	model = Model(input=x_in, output=y)
	model.compile(optimizer='adam', loss='mse', metrics=['mse'])
	return model

chrom, idx = sys.argv[1:]
n, d = chromosome_lengths[int(chrom)-1], 32+256+110

batch_size, num_steps = 20000, 10

model = AvocadoNN()
model.load_weights('/net/noble/vol5/user/jmschr/proj/avocado/models/full_model/havocado_chr{}.h5'.format(chrom), by_name=True)

celltype_embedding = numpy.load("../1_12_2018_Full_Model/celltype_embedding.npy")
assay_embedding = numpy.load("../1_12_2018_Full_Model/assay_embedding.npy")
genome_embedding = numpy.load("../1_12_2018_Full_Model/genome_embedding_chr20.npy")

all_tracks = training_set + test_set + validation_set
all_tracks = all_tracks[int(idx)::50]

X = numpy.empty((n, d))
X[:,288:] = genome_embedding

samples = numpy.empty((batch_size * num_steps, 32+256+110), dtype='float32')
step_sizes = numpy.empty((batch_size, 32+256+110), dtype='float32')

for idx, (celltype, assay) in enumerate(sorted(all_tracks)):
	celltype_idx = celltypes.index(celltype)
	assay_idx = assays.index(assay)

	X[:, :32] = celltype_embedding[celltype_idx]
	X[:, 32:288] = assay_embedding[assay_idx]

	for i in range(batch_size):
		step_sizes[i, :288] = X[0, :288] * 1.0 / num_steps
		for j in range(num_steps):
			samples[i*num_steps + j, :288] = X[0, :288] * (1.0 * (j+1) / (num_steps)) 

	ig = integrated_gradients(model, samples, step_sizes)

	track = numpy.empty((X.shape[0], 5), dtype='float32')
	tic1 = time.time()
	for i in range(X.shape[0] // batch_size + 1):
		start = i * batch_size
		end = (i+1) * batch_size

		tic = time.time()
		attribution = ig.explain(X[start:end, 288:], num_steps=num_steps)
		toc = time.time() - tic
		track[start:end, 0] = attribution[:, :32].sum(axis=1)
		track[start:end, 1] = attribution[:, 32:288].sum(axis=1)
		track[start:end, 2] = attribution[:, 288:313].sum(axis=1)
		track[start:end, 3] = attribution[:, 313:353].sum(axis=1)
		track[start:end, 4] = attribution[:, 353:].sum(axis=1)


	print "[{} chr{}; {:4}:1014] Total: {:4.4}s".format(celltype, chrom, idx, time.time() - tic1)

	track = track.astype('float32')
	numpy.savez_compressed('{}/{}.{}.chr{}.exps.npz'.format(attr_dir, celltype, assay, chrom), track)
