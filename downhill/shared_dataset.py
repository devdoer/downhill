#coding=utf8
'''This module contains class SharedDatset for theano shared data.
'''
import climate
import theano
import numpy as np
from .dataset import Dataset

logging = climate.get_logger(__name__)

class ChunksReader(object):
	'''Base class for chunks reader, this class should implement `next_chunks_of_inputs` interface.
	'''
	def next_chunk_of_inputs(self):
		'''Yield next chunk to be processed

		Yields
		------
		inputs : next chunk inputs to be processed
		'''
		raise NotImplementedError



class SharedDataset(Dataset):
	'''This class handle inputs stored in shared theano variables.
	'''
	_count = 0
	def __init__(self, chunks_reader, name=None, batch_size=32, axis=0):
		name = name or 'SharedDataset{}'.format(SharedDataset._count)
		SharedDataset._count += 1
		self._axis = axis
		self._batch_size = batch_size
		self._chunks_reader = chunks_reader
		self._chunks_count = 0
		self._shareds = None

	def make_shareds(self, input_variables):
		'''Create shared variables from inputs
		'''
		def make_value(ndim, dtype):
			VALUES = [0, (10,), (10,10), (10,10,10)]
			if ndim == 0:
				return np.cast[dtype](0)
			else:
				return np.zeros(VALUES[ndim], dtype=dtype)	
		shareds = [ theano.shared(make_value(x.ndim, x.dtype), 
						name='shared_{}'.format(x.name), borrow=True) for x in input_variables]
		self._shareds = shareds

	def get_shareds(self):
		return self._shareds

	# def get_batch_size(self):
		# return self._batch_size	

	def __iter__(self):
		'''
		partition  partition_size  index batch_size

		Returns
		-------

		'''
		batch_size = self._batch_size
		chunks_reader = self._chunks_reader
		shareds  = self._shareds
		axis = self._axis

		assert(shareds != None ), "Shared theano variable is not created!"
		for inputs in chunks_reader.next_chunk_of_inputs():
			if not inputs:
				logging.info('ends of chunks, chunks count is [%d]', self._chunks_count)
				break
			self._chunks_count += 1

			if not isinstance(inputs, (tuple, list)):
				inputs = (inputs, )
			L = inputs[0].shape[axis]
			assert( all(L== x.shape[axis] for x in inputs) ), 'shapes do not match along axis {}: {}'. \
								format( axis, '; '.join(str(x.shape) for x in chunk))
			assert(len(inputs) == len(shareds)), 'inputs length:{} do not match shareded variable \
				                size:{}'.format(len(inputs), len(shareds))

			for shared, x in zip(shareds, inputs):
				shared.set_value(x)

			for index in xrange(0, L, batch_size):
				end_index = min(L, index+batch_size)
				yield (index, end_index)



