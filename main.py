from import_all import *

from manage_images import ManageImages
from model_factory import ModelFactory
from gene import Gene
from datas import Datas
from breeder import Breeder
from geneCreator import GeneCreator

'''
What does the whole system do?


This demo project analyzes some labeled radar-based images (taken from public Statoil/C-CORE's
database) that can contain both ships and icebergs, the goal of the project is to create a classifier able
to distinguish these two situations and to guess what does an image contain.
The images are in json format and each point of the image is a two-dimensional vector (this kind of
radar images has 2 layers: HH, that means transmit/receive horizontally and HV that means transmit
horizontally and receive vertically).
The system operates some data wrangling, for example, creating a third channel for the images
(because working with 2 layer images is quite unusual, the program is able to create a third layer
based on the existing ones) or cropping the images (the part of the radar images with the ship or the
iceberg is more interesting than the background, that is simply sea).
After having prepared the data the system uses refined images to train a convolutional neural network,
with some dense layers on the top able to obtain a classification.

In this project there is an implementation of a genetic algorithm able to automatically configure the
topology of the neural network (each gene has inside him some infos about the topology
and some parameters.)
This is a demo, so around the code, somo point are intentionally simplified 
(e.g. each gene has a reduced amount of informations, the other ones are cabled)

The entire project is based on Python3 and Keras library.
'''



#datasets taken from https://www.kaggle.com/, use Pandas to read it   
train = pd.read_json("data/train.json")
test = pd.read_json("data/test.json")

print( "train column values: " + str( train.columns.values ) )
print( "test column values: " + str( test.columns.values ) )

#print( "len:  " + str( len(train) ) )
#print( train.describe()  )

#train = train[ 0 : 65]

#fix empty values in dataset (just to be sure cause the dataset is pretty clean)
test.inc_angle = test.inc_angle.replace('na', 0)
test.inc_angle = test.inc_angle.astype(float).fillna(0.0)
train.inc_angle = train.inc_angle.replace('na', 0)
train.inc_angle = train.inc_angle.astype(float).fillna(0.0)

#some data mungling in order to have features and label to train, and features to test
manage_images = ManageImages()  
y_train, X_train = manage_images.create_dataset( train )
X_test = manage_images.create_dataset_test( test )

#Datas object is a convenient way to manage features and label to train, and features to test
datas = Datas( X_train = X_train, y_train = y_train, X_test = X_test )


#some informations to debug if needed..
#print( "len:  " + str( len(X_train) ) )
#print("\n shape: " + str(X_train[0].shape ) )
#plt.imshow( X_train[2] )
#plt.show()
#print( "------------param:  |" + str(X_train.shape[1:]) + "|" )


#a good settings for debug if needed 
#gene = Gene( 0.001, 0.2, 512 )
#model_factory = ModelFactory( )
#model = model_factory.getModel( gene )
#print("yeah we have a model")

#model_factory.run( datas, model, 1 )



#good compromise taken into account my reduced computational power
population = 6
nGenerations = 6


creator = GeneCreator()
breeder = Breeder()

print( "\n\n\n########################## START! ##########################")
#first generation is (almost) completely random (random between a range)
generation = breeder.getFirstGeneration( population )
#let's try and evalue the first generation
generation = breeder.run( generation, datas )

for i in range ( 0 , nGenerations ):
	print( "\n\n\n########################## GENERATION: " + str(i) + " ##########################")
	#create a new generation
	generation = breeder.getNewGeneration(generation , population)
	#try and evalue it
	generation = breeder.run( generation, datas )
	#take the best one of this generation and pring how good is it
	best = breeder.takeBest( generation )
	print("we reach a error of: " + str( best.level) )
	best.toStr()

print( "\n\n\n########################## RE-RUN THE BEST: ##########################")

model_factory = ModelFactory( )
model = model_factory.getModel( best )
print("yeah we have a model")

model_factory.run( datas, model, 1 )

predicted_test=model_factory.evaluate( X_test, model, test )

print( "Finished" )