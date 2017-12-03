from import_all import *

class UseVggModel:


	def __init__( self):
		self.batch_size=64
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

	def getVggAngleModel( self , images):
		X_train = images
		input_2 = Input(shape=[1], name="angle")
		angle_layer = Dense(1, )(input_2)
		base_model = VGG16(weights=None, include_top=False, 
	                 input_shape=X_train.shape[1:], classes=1)
		base_model.load_weights('input/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
		x = base_model.get_layer('block5_pool').output
		x = GlobalMaxPooling2D()(x)
		merge_one = concatenate([x, angle_layer])
		merge_one = Dense(512, activation='relu', name='fc2')(merge_one)
		merge_one = Dropout(0.3)(merge_one)
		merge_one = Dense(512, activation='relu', name='fc3')(merge_one)
		merge_one = Dropout(0.3)(merge_one)
		predictions = Dense(1, activation='sigmoid')(merge_one)

		model = Model(input=[base_model.input, input_2], output=predictions)
		sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(loss='binary_crossentropy',
	                  optimizer=sgd,
	                  metrics=['accuracy'])
		return model


	def generator( self, features, labels, batch_size):
	    batch_features = np.zeros((batch_size, 75, 75, 3))
	    batch_labels = np.zeros((batch_size,1))
	    while True:
	        for i in range(batch_size):
	            index= random.randint(0, (batch_size - 1 ) )
	            batch_features[i] = features[index]
	            batch_labels[i] = labels[index]
	    yield batch_features, batch_labels

	def run( self, images, angles, labels, model):   
	    file_path = "input/aug_model_weights.hdf5"
	    callbacks = self.get_callbacks(filepath=file_path, patience=5)
	    gen_flow = self.gen_flow_for_two_inputs(images, angles, labels)
	    galaxyModel= self.getVggAngleModel(images)
	    galaxyModel.fit_generator(
	        gen_flow,
	        steps_per_epoch=14,
	        epochs=100,
	        verbose=1,
	        callbacks=callbacks)
	    
	    #Getting the Best Model
	    galaxyModel.load_weights(filepath=file_path)
	    #Getting Training Score
	    score = galaxyModel.evaluate([images,angles], labels, verbose=0)
	    print('Train loss:', score[0])
	    print('Train accuracy:', score[1])




	def evaluate( self, images, labels):
	    score = model.evaluate(images, labels, batch_size=50)
	    return score