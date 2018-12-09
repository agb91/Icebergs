from  __future__ import division
from gene import Gene
import random

class GeneCreator:
	
	# all the useful random generator fo settings, because we need to start from a random point!
	# note: they generates only realistic values 
	# (which values are realistically needed to generate a good solution is my personal 
	# opinion based on my experience)

	def randomDropout(self):
		result = (random.random() / 2)  # I wanna something in 0 - 0.5
		return ( result )	

	def randomLs1(self):
		return ( random.randint(360,800) )	

	def randomLs2(self):
		return ( random.randint(180,300) )				

	def randomLr(self):
		result = ( random.random() / 65 )
		return result

	def randomCreate(self):
		dropout = self.randomDropout()
		#dropout2 = self.randomDropout()
		l1 = self.randomLs1()
		#l2 = self.randomLs2()
		lr = self.randomLr()
		gene = Gene(  lr, dropout, l1 )
		return gene 	
	