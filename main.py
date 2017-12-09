from import_all import *

from manage_images import ManageImages
from use_vgg_model import UseVggModel
from gene import Gene
from datas import Datas




    
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

datas = Datas( X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid )


print( "len:  " + str( len(X_train) ) )
print("\n shape:")
print( X_train[0].shape )

#print( "------------param:  |" + str(X_train.shape[1:]) + "|" )

#examples: mentum = 0.9,dropout = 0.3, l1 = 512, l2 = 512,

gene = Gene( 6, True, True, 6, 0.9,
		0.3, 256, 128 , 2 )
use_vgg_model = UseVggModel( gene )
model = use_vgg_model.getVggAngleModel( datas )
print("yeah we have a model")

use_vgg_model.run( datas, model )











creator = GeneCreator()
breeder = Breeder()

print( "\n\n\n########################## TRY! ##########################")
generation = breeder.getFirstGeneration( population )
generation = breeder.run( generation, datas )

for i in range ( 0 , nGenerations ):
	print( "\n\n\n########################## GENERATION: " + str(i) + " ##########################")
    generation = breeder.getNewGeneration(generation , population)
    generation = breeder.run( generation )
    #print( "gen lenght: " + str(len(generation)) )
    best = breeder.takeBest( generation )
    #best.toStr()
    tot = 0
    
    print("we reach a error of: " + str( best.level) )

print( "\n\n\n########################## IN THE END ##########################")

print( "Finished" )