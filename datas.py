from import_all import *

class Datas:

	def __init__( self, X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid ):
		self.X_train = X_train
		self.X_valid = X_valid
		self.X_angle_train = X_angle_train
		self.X_angle_valid = X_angle_valid
		self.y_train = y_train
		self.y_valid = y_valid

		