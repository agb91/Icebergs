from import_all import *
import random

class Gene:


	#( self, batch_size, h_flip, v_flip, free_levels, momentum,	dropout, l1, l2, steps_per_epoch )

	def __init__( self, batch_size, h_flip, v_flip, free_levels, momentum,
		dropout, l1, l2, steps_per_epoch ):
		self.batch_size = batch_size
		self.h_flip = h_flip
		self.v_flip = v_flip
		self.free_levels = free_levels
		self.momentum = momentum
		self.dropout = dropout
		self.l1 = l1
		self.l2 = l2
		self.steps_per_epoch = steps_per_epoch
		self.level = None

	def toStr( self ):
		print( "gene: \n " +
			"\n batch size: " + str( self.batch_size ) +
			"\n h_flip: " + str( self.h_flip ) +
			"\n v_flip: " + str( self.v_flip ) +
			"\n free_levels: " + str( self.free_levels ) +
			"\n momentum: " + str( self.momentum ) +
			"\n dropout: " + str( self.dropout ) +
			"\n l1: " + str( self.l1 ) +
			"\n l2: " + str( self.l2 ) +
			"\n steps_per_epoch: " + str( self.steps_per_epoch )
			 )

	def setFitnessLevel( self, l ):
		self.level = l	

