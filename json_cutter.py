from  __future__ import division
import math
# Data reading and visualization
import pandas as pd




train = pd.read_json("test.json")

print(type(train))

train = train[0:200]

print(len(train))

train.to_json("test.json")
