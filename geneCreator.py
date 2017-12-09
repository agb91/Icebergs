from  __future__ import division
from gene import Gene
import random

class GeneCreator:
	
	def randomMomentum(self):
		result = (random.random() / 2) + 0.5  # I wanna something in 0.5 - 1
		return ( result )

	def randomDropout(self):
		result = (random.random() / 2)  # I wanna something in 0 - 0.5
		return ( result )	

	def randomBool(self):
		result = random.randint(0,1)
		if( result == 0 ):
			return True
		else:
			return False

	def randomFree(self):
		return ( random.randint(0,2) )

	def randomBatch( self ):
		return ( random.randint(20,60) )

	def randomLs(self):
		return ( random.randint(10,360) )			

	def randomSteps(self):
		return ( random.randint(10,40) )			

	def randomCreate(self):
		batch_size = self.randomBatch()
		h_flip = self.randomBool()
		v_flip = self.randomBool()
		free_levels = self.randomFree()
		momentum = self.randomMomentum()
		dropout = self.randomDropout()
		l1 = self.randomLs()
		l2 = self.randomLs()
		steps_per_epoch = self.randomSteps()
		gene = Gene(  batch_size, h_flip, v_flip, free_levels, momentum,
			dropout, l1, l2, steps_per_epoch )
		return gene 	
	