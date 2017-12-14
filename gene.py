from import_all import *
import random

class Gene:


	def __init__( self, lr,	dropout1, dropout2, l1, l2 ):

		self.lr = lr
		self.dropout1 = dropout1
		self.dropout2 = dropout2
		self.l1 = l1
		self.l2 = l2
		self.level = None

	def toStr( self ):
		print( "gene: \n " +
			"\n lr: " + str( self.lr ) +
			"\n dropout1: " + str( self.dropout1 ) +
			"\n dropout2: " + str( self.dropout2 ) +
			"\n l1: " + str( self.l1 ) +
			"\n l2: " + str( self.l2 ) +
		
			+ "\n result level: " + str( self.level )
			 )

	def setFitnessLevel( self, l ):
		self.level = l	

