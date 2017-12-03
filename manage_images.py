from import_all import *

class ManageImages:

	def resizer(  self, images , r, c ):
	    type(images)
	    scale = lambda x: resize( x , (r,c) )
	    images = scale(images)
	    return images

	# Translate data to an image format
	def color_composite( self, data):
	    rgb_arrays = []
	    for i, row in data.iterrows():
	        band_1 = np.array(row['band_1']).reshape(75, 75)
	        #band_1 = resize( band_1, (28, 28) )
	        band_2 = np.array(row['band_2']).reshape(75, 75)
	        #band_2 = resize( band_2, (28, 28) )
	         #angle = float( math.ceil(row['inc_angle']) ) / 100.0
	        #band_3 = np.full((75, 75), angle)
	       
	        band_3 = (band_1 / band_2)

	        r = (band_1 + abs(band_1.min())) / np.max((band_1 + abs(band_1.min())))
	        g = (band_2 + abs(band_2.min())) / np.max((band_2 + abs(band_2.min())))
	        b = (band_3 + abs(band_3.min())) / np.max((band_3 + abs(band_3.min())))

	        rgb = np.dstack((r, g, b))
	        rgb_arrays.append(rgb)
	    return np.array(rgb_arrays)

	def hvhh( self, data):
	    band_1V = []
	    band_2V = []
	    for i, row in data.iterrows():
	        band_1 = np.array(row['band_1']).reshape(75, 75)
	        #band_1 = resize( band_1, (28, 28) )
	        band_1V.append(band_1)
	        band_2 = np.array(row['band_2']).reshape(75, 75)
	        #band_2 = resize( band_2, (28, 28) )
	        band_2V.append(band_2)
	    return band_1V ,band_2V    

	def denoise( self, X, weight, multichannel):
	    return np.asarray([denoise_tv_chambolle(item, weight=weight, multichannel=multichannel) for item in X])

	def smooth( self, X, sigma):
	    return np.asarray([gaussian(item, sigma=sigma) for item in X])

	def grayscale( self, X):
	    return np.asarray([rgb2gray(item) for item in X])

	def augment( self, images):
	    image_mirror_lr = []
	    image_mirror_ud = []
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
	        
	    mirrorlr = np.array(image_mirror_lr)
	    mirrorud = np.array(image_mirror_ud)
	    images = np.concatenate((images, mirrorlr, mirrorud))
	    return images



	def create_dataset( self, frame, labeled, smooth_rgb=0.2, smooth_gray=0.5,
	                   weight_rgb=0.05, weight_gray=0.05):
	    band_1, band_2 = None,None
	    images = self.color_composite(frame)

	    images = self.smooth( self.denoise(images, weight_rgb, True), smooth_rgb)
	    print('images done, they are:' + str( len(images) )  )

	    X_angle = np.array(frame.inc_angle)
	    print( 'angles done, they are:' + str( len(X_angle) )  )
	    
	    y = np.array(frame["is_iceberg"])
	    print( 'labels done, they are:' + str( len(y) )  )

	    return y, X_angle, band_1, band_2, images    

    