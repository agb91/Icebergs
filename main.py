from import_all import *

from manage_images import ManageImages
from model_factory import ModelFactory
from gene import Gene
from datas import Datas
from breeder import Breeder
from geneCreator import GeneCreator



    
train = pd.read_json("data/train.json")

print( train.columns.values )
#print( "len:  " + str( len(train) ) )
#print( train.describe()  )

#train = train[ 0 : 65]


train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)


manage_images = ManageImages()  
y, X_angle, band1, band2, images = manage_images.create_dataset(train, True)

#for the moment here I don't need a validation set (maybe later I'll do)
X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = train_test_split(
    images , X_angle, y, random_state=123, train_size=0.99)

datas = Datas( X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid )


print( "len:  " + str( len(X_train) ) )
print("\n shape:")
print( X_train[0].shape )

#print( "------------param:  |" + str(X_train.shape[1:]) + "|" )

'''
#lr, dropout1, dropout2, l1, l2 
gene = Gene( 0.01, 0.2,	0.3, 512, 256 )
model_factory = ModelFactory( )
model = model_factory.getModel( gene )
print("yeah we have a model")

model_factory.run( datas, model, 1 )
'''



population = 5
nGenerations = 4


creator = GeneCreator()
breeder = Breeder()

print( "\n\n\n########################## TRY! ##########################")
generation = breeder.getFirstGeneration( population )
generation = breeder.run( generation, datas )

for i in range ( 0 , nGenerations ):
	print( "\n\n\n########################## GENERATION: " + str(i) + " ##########################")
	generation = breeder.getNewGeneration(generation , population)
	#print("genelen before run: " +  str( len(generation) ) )
	generation = breeder.run( generation, datas )
	#print("genelen after run: " +  str( len(generation) ) )
	best = breeder.takeBest( generation )
	print("we reach a error of: " + str( best.level) )
	gene.toStr()

print( "\n\n\n########################## RE-RUN THE BEST: ##########################")

model_factory = ModelFactory( )
model = model_factory.getModel( best )
print("yeah we have a model")

model_factory.run( datas, model, 1 )


print( "Finished" )