from  __future__ import division

from gene import Gene
from geneCreator import GeneCreator
from manage_images import ManageImages
from use_vgg_model import UseVggModel
from gene import Gene
from datas import Datas
from geneCreator import GeneCreator
from import_all import *

class Breeder:

	def getNewGeneration( self, old, n):
		geneCreator = GeneCreator()
		newGeneration = list()
		strongestN = 3
		if(n<3):
			strongestN = n
		reprods = math.ceil(n/2.5)
		randomAdds = 2 
		goods = self.takeGoods( old , strongestN )
		reproducible = self.takeGoods(old, reprods )
		#I want to maintain old goods in my genetic pools
		for i in range( 0, len(goods) ):
			newGeneration.append(goods[i])
		#I want some sons generated by goods
		for i in range( 0 , (n - strongestN - randomAdds ) ):
			son = self.getSon( reproducible )
			newGeneration.append(son)
		#I want also some randoms new borns
		for i in range( 0, randomAdds ):
			newGeneration.append( geneCreator.randomCreate() )
		
		return newGeneration

	def getSon( self, parents ):


		self.steps_per_epoch = steps_per_epoch

		bsi = random.randint(0, (len(parents) - 1 ) )
		bs = parents[bsi].batch_size 

		hfi = random.randint(0, (len(parents) - 1 ) )
		hf = parents[hfi].h_flip 

		vfi = random.randint(0, (len(parents) - 1 ) )
		vf = parents[vfi].v_flip 
		
		fri = random.randint(0, (len(parents) - 1 ) )
		fr = parents[fri].free_levels 

		moi = random.randint(0, (len(parents) - 1 ) )
		mo = parents[moi].momentum 

		dri = random.randint(0, (len(parents) - 1 ) )
		dr = parents[dri].dropout 

		l1i = random.randint(0, (len(parents) - 1 ) )
		l1 = parents[l1i].l1

		l2i = random.randint(0, (len(parents) - 1 ) )
		l2 = parents[l2i].l2 

		sti = random.randint(0, (len(parents) - 1 ) )
		st = parents[sti].steps_per_epoch 


		son = Gene( cbt, ss, mcw, md, ne, lr, way , n_neighbors)
		
		return son	

	def run(self, generation, datas):
		runnedGeneration = list()
		


		X_train, X_valid, X_angle_train, X_angle_valid, y_train, y_valid = datas.X_train, datas.X_valid, datas.X_angle_train, datas.X_angle_valid, datas.y_train, datas.y_valid

		
		for i in range( 0 , len(generation)):
			
			thisGene = generation[i]
			use_vgg_model = UseVggModel( thisGene )
			model = use_vgg_model.getVggAngleModel( datas )
			#print("yeah we have a model")

			result = use_vgg_model.run( datas, model )
			#print( evaluate( X_train, y_train ) )

			thisGene.setFitnessLevel( result ) 
			runnedGeneration.append(thisGene)

		return runnedGeneration	

	def getFirstGeneration( self, n ):
		genes = list()
		creator = GeneCreator()
		for i in range( 0 , n):
			g = creator.randomCreate()
			genes.append(g)
		return genes

	def orderGenes( self , genes ):
		result = []
		genesSet = set(genes)
		genes = list( genesSet ) # no doubles!
		for i in range( 0, len(genes) ):
			print( "before: " + str(genes[i].level) )		
		
		result = sorted(genes, key=lambda x: x.level, reverse=False)
		for i in range( 0, len(result) ):
			print( "after: " + str(result[i].level) )		
		
		return result

	def takeGoods( self, genes, n ):
		goods = []

		for i in range(0, len(genes) ):
			g = genes[i]
			goods.append(g)
			goods = self.orderGenes( goods )
			if( len( goods ) > n):
				goods = goods[ 0 : n ]

		for i in range( 0, len(goods) ):
			print( "goods: " + str( goods[i].level ) )		
		return goods		    

	def takeBest( self, genes ):

		minLevel = 999 #level of error
		bestGene = None

		for i in range(0, len(genes) ):
			g = genes[i]
			if( g.level < minLevel ):
				bestGene = g
				minLevel = g.level

		return bestGene		