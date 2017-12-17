#Latest updated : Dec 11, 2017

"""
We need to extract all npy files to current folder including

boughtlist.npy
mapping.npy
model.npy
tree.npy

npy files can be founded in data.rar. Extract them to the same folder where instacart_predicting.py is located.

Or we can generate them by executing instacart_preprocessing.py automatically.

"""

import numpy as np
import time
import math
import csv

_NUM_USERS=15631
_NUM_PRODUCTS=49688

#Loading data getting from computation in instacart_preprocessing.py
t1 = int(round(time.time() * 1000))
boughtlist=np.load('boughtlist.npy')
prob_tree=np.load('tree.npy') 
model=np.load('model.npy') 
train_to_user=np.load('mapping.npy')
t2 = int(round(time.time() * 1000))
print("Time usage for loading data (second) : "+str((t2-t1)/1000))


#Predicting products that wil be bought
t1 = int(round(time.time() * 1000))
train_to_user=train_to_user.item() #Convert numpy to usable dict
predictlist={} #Data structure collecting product_id that may be bought next time in each order_id

k=2196797 #user_id test case in case of debugging

for key in range(1,6): #Looping for the 1st to 5th documents

	cases=[] #Variable collecting predicted products list that will be used for writing csv files

	with open('./dataset/order_ids_'+str(key)+'.csv', "r" ,encoding="utf8") as f: 
		next(f)
		for line in f:
			line = line.split(",")
			order_id=int(line[0])
		
			mapped=train_to_user[order_id] # Find user_id from givern order_id

			myorders=boughtlist[mapped-1] # Retrieve information of all user_id buying orders
			buyinglist=[]
			volumelist=[]

			for idx,product_id in enumerate(myorders):
				buyinglist.append(product_id)	# Collect all product_id bought by user_id
				volumelist.append(myorders[product_id])	#Collect number of product_id bought by user_id

			mean_volume=np.mean(volumelist)	#Find average number of buying
			variance_volume=np.var(volumelist)	#Find variance of buying
		
			for idx,product_id in enumerate(buyinglist):
				table=model[product_id-1]	#Get the table from model to consider product_id buying

				if len(table)!=0:	# Check if we can find this table for product_id

					count=0
				
					for idx2,referred_product_id in enumerate(buyinglist):

						current_volume=myorders[referred_product_id]	# Get number of referred_product_id buying

						if ((referred_product_id-1) in prob_tree[product_id-1] and (prob_tree[product_id-1][referred_product_id-1][1]<=1 or (prob_tree[product_id-1][referred_product_id-1][0]==2 and prob_tree[product_id-1][referred_product_id-1][1]<=3))):
							continue

						"""
						In this above step, we will ignore too low number of referred_product_id buying that should not significantly affect product_id buying due to extremely low number of cases
						"""

						pvalue=mean_volume+0.68*math.sqrt(variance_volume) #75% Confidence Interval

						if current_volume>pvalue and myorders[product_id]>mean_volume:	#Conditions for checking confidence in term of referred_product_id and product_id buying

							hl=0
							if math.fabs(volumelist[idx2]-table[0][0])>math.fabs(volumelist[idx2]-table[0][1]):	#Check if it is in Low-buying group or High-buying group
								hl=1

							if (referred_product_id-1) in table[1][hl]:	#Check if referred_product_id is in the list that affect buying product_id

								count+=1

								if count>=2 and not order_id in predictlist:
									predictlist[order_id]=[]
								if count>=2:
									predictlist[order_id].append(str(product_id))	# Add this product_id as the product that is believed to be bought next time

								"""
								We need at least two products (of referred_product_id) from user_id that affect product_id buying so we can confirm this strong relationship
								"""

						if count>=2:	#If we already add product_id to our list, we do not need to consider other referred_product_id anymore
							break

			if order_id in predictlist:
				lt=' '.join(list(set(predictlist[order_id])))
			else:
				lt='None'

			cases.append([str(order_id),lt])

	with open('./submissions/submission_'+str(key)+'.csv', 'w') as f:

		writer = csv.writer(f, lineterminator = '\n')
		writer.writerow(['order_id','products'])
		for item in cases:
			writer.writerow(item)

t2 = int(round(time.time() * 1000))
print("Time usage for predicting data (second) : "+str((t2-t1)/1000))

"""
For debugging only
print(len(boughtlist[train_to_user[k]-1]))
print(len(predictlist[k]))
print(boughtlist[train_to_user[k]-1])
print(predictlist[k])
"""
print("Finish predicting!")