from import_all import *

class Datas:

	#easy object, just contains some informations 
	def __init__( self, X_train, y_train, X_test ):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
	
		