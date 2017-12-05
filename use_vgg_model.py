from import_all import *
from gene import Gene

class UseVggModel:



	#def __init__( self, batch_size, h_flip, v_flip, free_levels, momentum,
	#	dropout, l1, l2, steps_per_epoch ):
	def __init__( self, gene ):
		self.steps_per_epoch = gene.steps_per_epoch
		self.batch_size = gene.batch_size
		self.momentum = gene.momentum
		self.dropout = gene.dropout
		self.free_levels = gene.free_levels
		self.l1 = gene.l1
		self.l2 = gene.l2
		# Define the image transformations here
		self.gen = ImageDataGenerator(horizontal_flip = gene.h_flip,
	                         vertical_flip = gene.v_flip,
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

	def gen_flow_for_two_inputs_val( self, X1, X2, y):
		val_batch_size = 30
		genX1 = self.gen.flow(X1,y,  batch_size=val_batch_size, seed=5 )
		genX2 = self.gen.flow(X1,X2, batch_size=val_batch_size, seed=5 )
		while True:
			X1i = genX1.next()
			X2i = genX2.next()
			yield [X1i[0], X2i[1]], X1i[1]        

	# Finally create generator
	def get_callbacks( self, filepath, patience=2):
	    es = EarlyStopping('val_loss', patience=10, mode="min")
	    msave = ModelCheckpoint(filepath, save_best_only=True)
	    return [es, msave]

	def getVggAngleModel( self , datas):
		X_train = datas.X_train
		input_2 = Input(shape=[1], name="angle")
		angle_layer = Dense(1, )(input_2)
		base_model = VGG16(weights=None, include_top=False, 
	                 input_shape=X_train.shape[1:], classes=1)
		
		free_levels = self.free_levels
		untrainable_n = len(base_model.layers) - free_levels
		#print( "not trainable levels: " + str( untrainable_n ) )
		for layer in base_model.layers[:untrainable_n]:
			layer.trainable = False

		#base_model.summary()	

		base_model.load_weights('input/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
		x = base_model.get_layer('block5_pool').output
		x = GlobalMaxPooling2D()(x)
		merge_one = concatenate([x, angle_layer])
		merge_one = Dense(self.l1, activation='relu', name='fc2')(merge_one)
		merge_one = Dropout(self.dropout)(merge_one)
		merge_one = Dense(self.l2, activation='relu', name='fc3')(merge_one)
		merge_one = Dropout(self.dropout)(merge_one)
		predictions = Dense(1, activation='sigmoid')(merge_one)

		model = Model(input=[base_model.input, input_2], output=predictions)
		sgd = SGD(lr=1e-3, decay=1e-6, momentum=self.momentum, nesterov=True)
		model.compile(loss='binary_crossentropy',
	                  optimizer=sgd,
	                  metrics=['accuracy'])
		return model

	def run( self, datas, model):   
	    file_path = "input/aug_model_weights.hdf5"
	    callbacks = self.get_callbacks(filepath=file_path, patience=5)
	    gen_flow = self.gen_flow_for_two_inputs(datas.X_train, datas.X_angle_train, datas.y_train)
	    gen_flow_val = self.gen_flow_for_two_inputs_val(datas.X_valid, datas.X_angle_valid, datas.y_valid)
	    galaxyModel= self.getVggAngleModel( datas )
	    galaxyModel.fit_generator(
	        gen_flow,
	        steps_per_epoch=self.steps_per_epoch,
	        epochs=2,
	        verbose=1,
	        callbacks=callbacks,
	        validation_data = gen_flow_val,
	        validation_steps = 15
	        )
	    
	    #Getting the Best Model
	    galaxyModel.load_weights(filepath=file_path)
	    #Getting Training Score
	    score = galaxyModel.evaluate([datas.X_valid,datas.X_angle_valid], datas.y_valid, verbose=0)
	    print('Valid loss:', score[0])
	    print('Valid accuracy:', score[1])
	    return score[0]



	def evaluate( self, images, labels):
	    score = model.evaluate(images, labels, batch_size=50)
	    return score