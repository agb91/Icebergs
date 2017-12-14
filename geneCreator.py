from  __future__ import division
from gene import Gene
import random

class GeneCreator:
	

	def randomDropout(self):
		result = (random.random() / 2)  # I wanna something in 0 - 0.5
		return ( result )	

	def randomLs(self):
		return ( random.randint(360,600) )			

	def randomLr(self):
		result = ( random.random() / 65 )
		return result

	def randomCreate(self):
		dropout1 = self.randomDropout()
		dropout2 = self.randomDropout()
		l1 = self.randomLs()
		l2 = self.randomLs()
		lr = self.randomLr()
		gene = Gene(  lr, dropout1, dropout2, l1, l2 )
		return gene 	
	