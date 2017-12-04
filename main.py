from import_all import *

from manage_images import ManageImages
from use_vgg_model import UseVggModel





    
train = pd.read_json("data/train.json")

print( train.columns.values )
#print( "len:  " + str( len(train) ) )
#print( train.describe()  )

#train = train[ 0 : 50]


train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)


manage_images = ManageImages()  
y, X_angle, band1, band2, images = manage_images.create_dataset(train, True)

X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(
    images , X_angle, y, random_state=123, train_size=0.67)


print( "len:  " + str( len(X_train) ) )
print("\n shape:")
print( X_train[0].shape )

#print( "------------param:  |" + str(X_train.shape[1:]) + "|" )

#examples: mentum = 0.9,dropout = 0.3, l1 = 512, l2 = 512,
use_vgg_model = UseVggModel( 60, True, True, 6, 0.9,
		0.3, 256, 128 , 20)
model = use_vgg_model.getVggAngleModel( X_train )
print("yeah we have a model")

use_vgg_model.run( X_train, X_angle_train, y_train, model )
#print( evaluate( X_train, y_train ) )


print( "Finished" )