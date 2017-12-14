from import_all import *
from gene import Gene

class ModelFactory:

	#lr,	dropout1, dropout2, l1, l2 
	def __init__( self ):
		'''
		self.lr = gene.lr
		self.dropout1 = gene.dropout1
		self.dropout1 = gene.dropout2
		self.l1 = gene.l1
		self.l2 = gene.l2
		'''
		# Define the image transformations here
		self.gen = ImageDataGenerator(horizontal_flip = True,
			vertical_flip = True,
			width_shift_range = 0.,
			height_shift_range = 0.,
			channel_shift_range=0,
			zoom_range = 0.2,
			rotation_range = 10)


	# Here is the function that merges our two generators
	# We use the exact same generator with the same random seed for both the y and angle arrays
	def gen_flow_for_two_inputs( self, X1, X2, y):
		genX1 = self.gen.flow(X1,y,  batch_size=self.batch_size,seed=55)
		genX2 = self.gen.flow(X1,X2, batch_size=self.batch_size,seed=55)
		while True:
			X1i = genX1.next()
			X2i = genX2.next()
			#Assert arrays are equal - this was for peace of mind, but slows down training
			#np.testing.assert_array_equal(X1i[0],X2i[0])
			yield [X1i[0], X2i[1]], X1i[1]


	# Finally create generator
	def get_callbacks( self, filepath, patience=2):
		es = EarlyStopping('val_loss', patience=10, mode="min")
		msave = ModelCheckpoint(filepath, save_best_only=True)
		return [es, msave]

	def getModel( self, gene ):

		model = keras.models.Sequential()

		model.add(keras.layers.convolutional.Conv2D(64, kernel_size=(3,3), 
			input_shape=(75,75,3)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(3,3), 
			strides=(2,2)))
		model.add(keras.layers.Dropout( gene.dropout1 ))

		model.add(keras.layers.convolutional.Conv2D(128, 
			kernel_size=(3, 3)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), 
			strides=(2, 2)))
		model.add(keras.layers.Dropout( gene.dropout1 ))

		model.add(keras.layers.convolutional.Conv2D(128, kernel_size=(3, 3)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), 
			strides=(2, 2)))
		model.add(keras.layers.Dropout( gene.dropout2 ))

		model.add(keras.layers.convolutional.Conv2D(64, kernel_size=(3, 3)))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(keras.layers.convolutional.MaxPooling2D(pool_size=(2, 2), 
			strides=(2, 2)))
		model.add(keras.layers.Dropout( gene.dropout2 ))

		model.add(keras.layers.Flatten())

		model.add(keras.layers.Dense( gene.l1 ))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(keras.layers.Dropout(0.2))

		model.add(keras.layers.Dense( gene.l2 ))
		model.add(Activation('relu'))
		model.add(BatchNormalization())
		model.add(keras.layers.Dropout(0.2))


		model.add(keras.layers.Dense(1))
		model.add(Activation('sigmoid'))

		mypotim=Adam(lr= gene.lr , decay=0.0)
		model.compile(loss='binary_crossentropy', optimizer = mypotim, metrics=['accuracy'])

		return model

	#mode 0 = genetic, mode 1 = serious
	def run( self, datas, model, mode):   
		file_path = "input/aug_model_weights.hdf5"

		batch_size = 64

		early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 0, mode= 'min')
		reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 7, verbose =1, 
			epsilon = 1e-4, mode='min', min_lr = 0.0001)
		model_filepath=file_path
		checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
		callbacks_list = [early_stopping, checkpoint]


		if( mode == 0):
			this_X_train = data.X_train[ 0 : 100]
			this_Y_train = data.y_train[0 : 100]
			history = model.fit(this_X_train, this_Y_train, batch_size = batch_size, epochs =4, 
				verbose =1, validation_split = 0.2, callbacks=callbacks_list)
			
			return history.history['val_loss']
				
		if( mode == 1 ):
			history = model.fit(datas.X_train, datas.y_train, batch_size = batch_size, 
				epochs = 20,	verbose =1, validation_split = 0.1, callbacks=callbacks_list)
			model_json = model.to_json()
			with open("input/model.json", "w") as json_file:
				json_file.write(model_json)
			return history.history['val_loss']



			