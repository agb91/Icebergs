from import_all import *
import random

class Gene:

	# in the Gene of this genetic algorithm there are the configurations of a neural network
	# the main aim of this application is to find the best combination of this configs
	def __init__( self, lr,	dropout, l1 ):
		#dropout 0.2 is a reasonable value
		self.lr = lr
		self.dropout = dropout
		self.l1 = l1
		self.level = None


	def toStr( self ):
		print( "gene: \n " +
			"\n lr: " + str( self.lr ) +
			"\n dropout: " + str( self.dropout ) +
			"\n l1: " + str( self.l1 ) +
			"\n result level: " + str( self.level )	 )

	#the level who show us how good is this solution	
	def setFitnessLevel( self, l ):
		nparray = np.asarray(l)
		nparray = sorted(nparray, reverse=False)
		#print( "\nfitness array: " + str( nparray ) )	
		self.level = nparray[0]
		#print("setted level: " + str(self.level) )


