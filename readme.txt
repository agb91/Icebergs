Icebergs - Machine learning

This demo project analyzes some labeled radar-based images (taken from public Statoil/C-CORE's database) that can contain both ships and icebergs, the goal of the project is to create a classifier able to distinguish these two situations and to guess what does an image contain.

The images are in json format and each point of the image is a two-dimensional vector (this kind of radar images has 2 layers: HH, that means transmit/receive horizontally and HV that means transmit horizontally and receive vertically). 
The system operates some data wrangling, for example, creating a third channel for the images (because working with 2 layer images is quite unusual, the program is able to create a third layer based on the existing ones) or cropping the images (the part of the radar images with the ship or the iceberg is more interesting than the background, that is simply sea).
After having prepared the data the system uses refined images to train a convolutional neural network, with some dense layers on the top able to obtain a classification.
In this project there is an implementation of a genetic algorithm able to automatically configure the topology of the neural network, but at the moment, even though its result is quite good, the best results are obtained by organizing the topology of the neural network by hand.  

The entire project is based on Python3 and Keras library. 
