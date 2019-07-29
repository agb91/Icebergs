from import_all import *

class ManageImages:

	#I've a 2 channel image, quite unusual, I want a 3 channel image
	#so I create a third channel based on the first two and I normalize them all
	def transform ( self, df ):
	    images = []
	    for i, row in df.iterrows():
	        band_1 = np.array(row['band_1']).reshape(75,75)
	        band_1 = band_1[10: - 10, 10: - 10]

	        band_2 = np.array(row['band_2']).reshape(75,75)
	        band_2 = band_2[10: - 10, 10: - 10]

	        band_3 = band_1 / band_2
	        
	        band_1_norm = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
	        band_2_norm = (band_2 - band_2. mean()) / (band_2.max() - band_2.min())
	        band_3_norm = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
	        
	        this_image = np.dstack((band_1_norm, band_2_norm, band_3_norm))

	        #print( "in method: " + str( this_image.shape ) )
	        images.append( this_image )

	    return np.array(images)

	#we can have more images just flipping, rotating, mirroring the original ones
	#more images implies less overfitting
	def augment( self, images ):
	    image_mirror_lr = []
	    image_mirror_ud = []
	    image_rotate = []
	    for i in range(0,images.shape[0]):
	        band_1 = images[i,:,:,0]
	        band_2 = images[i,:,:,1]
	        band_3 = images[i,:,:,2]
	            
	        # mirror left-right
	        band_1_mirror_lr = np.flip(band_1, 0)
	        band_2_mirror_lr = np.flip(band_2, 0)
	        band_3_mirror_lr = np.flip(band_3, 0)
	        image_mirror_lr.append(np.dstack((band_1_mirror_lr, band_2_mirror_lr, band_3_mirror_lr)))
	        
	        # mirror up-down
	        band_1_mirror_ud = np.flip(band_1, 1)
	        band_2_mirror_ud = np.flip(band_2, 1)
	        band_3_mirror_ud = np.flip(band_3, 1)
	        image_mirror_ud.append(np.dstack((band_1_mirror_ud, band_2_mirror_ud, band_3_mirror_ud)))
	        
	        #rotate 
	        band_1_rotate = rot(band_1, 30, reshape=False)
	        band_2_rotate = rot(band_2, 30, reshape=False)
	        band_3_rotate = rot(band_3, 30, reshape=False)
	        image_rotate.append(np.dstack((band_1_rotate, band_2_rotate, band_3_rotate)))
	        
	    mirrorlr = np.array(image_mirror_lr)
	    mirrorud = np.array(image_mirror_ud)
	    rotated = np.array(image_rotate)
	    images = np.concatenate((images, mirrorlr, mirrorud, rotated))
	    return images


	# need to transform raw 2-channel radar images to 3-channle normalized images
	# and to augment their number (through mirroring, rotating and so on)   
	# x are features, y are labels. is_iceberg is the label column
	def create_dataset( self, train ):

		train_X = self.transform( train )
		train_y = np.array(train ['is_iceberg'])

		indx_tr = np.where(train.inc_angle > 0)
		#print (indx_tr[0].shape)

		train_y = train_y[indx_tr[0]]
		train_X = train_X[indx_tr[0], ...]

		train_X = self.augment( train_X )
		train_y = np.concatenate((train_y,train_y, train_y, train_y))

		##print ("in method: " + str( train_X.shape ) )
		##print ("in method: " + str( train_y.shape ) )

		return train_y, train_X    

	# the test images don't have label so there is no need to manage test_y	
	def create_dataset_test( self, test ):

		test_X = self.transform( test )
		#test_y = np.array(test ['is_iceberg'])

		#print ("in method: " + str( test_X.shape ) )
		
		return test_X    	

    