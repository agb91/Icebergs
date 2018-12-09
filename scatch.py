from import_all import *


train = pd.read_json("data/train.json")
test = pd.read_json("data/test.json")

#print( "train column values: " + str( train.columns.values ) )
#print( "test column values: " + str( test.columns.values ) )

#print( "len:  " + str( len(train) ) )
#print( train.describe()  )

#train = train[ 0 : 65]
print( train.isnull().sum().sum() )

#print( test.inc_angle )