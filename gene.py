from import_all import *
import random

class Gene:

	#dropdown was 0.2
	def __init__( self, lr,	dropout, l1 ):

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

	def setFitnessLevel( self, l ):
		nparray = np.asarray(l)
		nparray = sorted(nparray, reverse=False)
		#print( "\nfitness array: " + str( nparray ) )	
		self.level = nparray[0]
		#print("setted level: " + str(self.level) )


