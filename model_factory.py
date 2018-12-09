from import_all import *
from gene import Gene

class ModelFactory:

	#lr,dropout1, dropout2, l1, l2 
	def __init__( self ):
		self.file_path = ".model_weights.hdf5"

		# thanks Keras
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

	#create a convolutional neural network, using the information present in the gene
	#why are we using only some information in the gene and some other are cabled?
	#because this is demo!
	def getModel( self, gene ):

		#Building the model
	    gmodel=Sequential()
	    #Conv Layer 1
	    gmodel.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(55, 55, 3)))
	    gmodel.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding = 'same'))
	    gmodel.add(Dropout(0.2))

	    #print("-------------------------- layer 1 --------------------------")
	    #for layer in gmodel.layers:
	    #	print(layer.output_shape)
	    
	    #Conv Layer 2
	    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
	    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding = 'same'))
	    gmodel.add(Dropout( gene.dropout ))

	    #print("-------------------------- layer 2 --------------------------")
	    #for layer in gmodel.layers:
	    #	print(layer.output_shape)


	    #Conv Layer 3
	    gmodel.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
	    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	    gmodel.add(Dropout( gene.dropout ))

	    #print("-------------------------- layer 3 --------------------------")
	    #for layer in gmodel.layers:
	    #	print(layer.output_shape)

	    	
	    #Conv Layer 4
	    gmodel.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
	    gmodel.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
	    gmodel.add(Dropout( gene.dropout ))

	    #print("-------------------------- layer 4 --------------------------")
	    #for layer in gmodel.layers:
	    #	print(layer.output_shape)
	    
	    #Flatten the data for upcoming dense layers
	    gmodel.add(Flatten())

	    #Dense Layers
	    gmodel.add(Dense( gene.l1 ))
	    gmodel.add(Activation('relu'))
	    gmodel.add(Dropout(gene.dropout))#was 0.2

	    #Dense Layer 2
	    gmodel.add(Dense( int(gene.l1 / 2) ))
	    gmodel.add(Activation('relu'))
	    gmodel.add(Dropout( gene.dropout ))

	    #Sigmoid Layer
	    gmodel.add(Dense(1))
	    gmodel.add(Activation('sigmoid'))

	    mypotim=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	    gmodel.compile(loss='binary_crossentropy',
	                  optimizer=mypotim,
	                  metrics=['accuracy'])

	    #gmodel.summary()

	    return gmodel

	#mode 0 = genetic, mode 1 = cabled
	#train the neural network, and use and evalue it
	'''
		Keras can do it for me!:
			is_Y_train = datas.y_train[0 : 350]
			history = model.fit(this_X_train, this_Y_train, batch_size = 25, epochs =5, 
				verbose =1, validation_split = 0.5, callbacks=callbacks_list)
			
			return history.history['val_loss']
	'''
	def run( self, datas, model, mode):   
		

		batch_size = 25

		early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 0, mode= 'min')
		reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 7, verbose =1, 
			epsilon = 1e-4, mode='min', min_lr = 0.0001)
		model_filepath=self.file_path
		checkpoint = ModelCheckpoint(model_filepath, monitor='val_loss', verbose=1,
			save_best_only=True)
		callbacks_list = [early_stopping, checkpoint]

		#genetical
		#this is a demo se I use only some images
		if( mode == 0):
			this_X_train = datas.X_train[ 0 : 350]
			this_Y_train = datas.y_train[0 : 350]
			history = model.fit(this_X_train, this_Y_train, batch_size = 25, epochs =5, 
				verbose =1, validation_split = 0.5, callbacks=callbacks_list)
			
			return history.history['val_loss']
				
		#cabled		
		if( mode == 1 ):
			history = model.fit(datas.X_train, datas.y_train, batch_size = batch_size, 
				epochs = 10, verbose =1, validation_split = 0.5, callbacks=callbacks_list)
			return history.history['val_loss']

	def x01( self, x ):
		if( x < 0.5 ):
			return 0.0
		else:
			return 1.0			

	def evaluate( self, X_test, model, test ):
		model.load_weights( filepath=self.file_path )
		predicted_test = model.predict_proba(X_test)
		submission = pd.DataFrame()
		submission['id']=test['id']
		results = predicted_test.reshape((predicted_test.shape[0]))
		#results = np.array( results )
		#results = np.where(results > 0.5, 1, 0)
		#np.array([list(i*0.01) if i>0 else list(i) for i in arr])
		#print ( results.shape )
		#print ( results )
		submission['is_iceberg'] = results
		submission.to_csv('result/sub.csv', index=False)	

			