#Latest updated : Dec 17, 2017

"""
We need to extract all csv files to folder named 'dataset' that is in the same location as instacart_preprocessing.py including

products.csv
order_products__prior.csv
order_products__train.csv
orders.csv

CSV files can be downloaded from https://www.instacart.com/datasets/grocery-shopping-2017. Put them all to the same folder where instacart_preprocessing.py is located.
"""

from openpyxl import load_workbook
import csv
import numpy as np
import time
import math

_NUM_USERS=15631
_NUM_PRODUCTS=49688

file  = open('./dataset/products.csv', "r" ,encoding="utf8")
read = csv.reader(file)

products=[{} for _ in range(_NUM_PRODUCTS)] #Variable that collect information of each product_id

for product_id in range(_NUM_PRODUCTS) :
	products[product_id]['freq']=0 # Define how much this product is bought for each users


orders=[[] for _ in range(35000000)] #Variable that collect information of each order_id

#read orders data (prior) to list
t1 = int(round(time.time() * 1000))
with open('./dataset/order_products__prior.csv', "r" ,encoding="utf8") as f: 
	next(f)
	for line in f:
		
		line = line.split(",")
		order_id=int(line[0])
		product_id=line[1]
		seq=int(line[2])
		
		orders[order_id].append(product_id)
	
#read orders data (train) to list
with open('./dataset/order_products__train.csv', "r" ,encoding="utf8") as f: 
	next(f)
	for line in f:

		line = line.split(",")
		order_id=int(line[0])
		product_id=line[1]
		seq=int(line[2])
		
		orders[order_id].append(product_id)

t2 = int(round(time.time() * 1000))

print("Time usage for loading prior data (second) : "+str((t2-t1)/1000))

#read users data (train) to list	
users=[{} for _ in range(_NUM_USERS)]
boughtlist=[{} for _ in range(_NUM_USERS)]
trainlist=[[] for _ in range(_NUM_USERS)]
train_to_user={}
total_users=0

t1 = int(round(time.time() * 1000))

with open('./dataset/orders.csv', "r" ,encoding="utf8") as f: 
	next(f)
	temp_dow=0.0
	temp_hod=0.0
	temp_dsp=0.0
	temp_user=1
	for line in f:
		line = line.split(",")
		order_id=int(line[0])
		user_id=int(line[1])
		eval=line[2]
		order_number=float(line[3])
		dow=float(line[4])
		hod=float(line[5])
		dsp=0

		if user_id!=temp_user:
			users[temp_user]['dow']=float(temp_dow)
			users[temp_user]['hod']=float(temp_hod)
			users[temp_user]['dsp']=float(temp_dsp)
			
			temp_dow=0.0
			temp_hod=0.0
			temp_dsp=0.0
			temp_user=user_id

		temp_dow=float(temp_dow*(order_number-1)+dow)/(order_number)
		temp_hod=float(temp_hod*(order_number-1)+hod)/(order_number)
		temp_dsp=float(temp_dsp*(order_number-1)+dsp)/(order_number)

		if eval=="prior":
			for item in orders[order_id]: # Count how many this product is bought by this user
				product_id=int(item)
				if not product_id in boughtlist[user_id-1]:
					boughtlist[user_id-1][product_id]=0
				boughtlist[user_id-1][product_id]+=1


				products[product_id-1]['freq']+=1
		else:
			for item in orders[order_id]: # Collect the latest product bought by this user
				product_id=int(item)
				trainlist[user_id-1].append(product_id)
				train_to_user[int(order_id)]=int(user_id) #map the data which user_id buys order_id (Given order_id, we can map back to user_id who buy this order)
			total_users+=1

t2 = int(round(time.time() * 1000))
print("Time usage for processing data (second): "+str((t2-t1)/1000))

#read test cases to list <-- Not used
"""
t1 = int(round(time.time() * 1000))

with open('./dataset/test_set.csv', "r" ,encoding="utf8") as f: 
	next(f)
	for line in f:

		line = line.split(",")
		order_id=int(line[0])

		typelist[train_to_user[order_id]-1]=1 # This is out sample

t2 = int(round(time.time() * 1000))
print("Time usage for classifying in-sample and out-sample (second): "+str((t2-t1)/1000))
"""

#Build forest tree to construct product buying structure
t1 = int(round(time.time() * 1000))

prob_tree=[{} for _ in range(_NUM_PRODUCTS)]

"""
prob_tree is a data structure that collects information of how latest buying is affected by previous buying in term of product_id

prob_tree[A][B] keeps the information of latest product A buying and it will be linked to the recently buying of product B by collecting number of cases which product B causes buying of product A and cumulative buying of product B

For example, prob_tree[14][42]=[9,80] implies that buying product number 14 in the future is affected by recent buying of product number 42 for 9 cases and the cummulative product number 42 buying is 80 

"""

for user_id in range(_NUM_USERS):

	for idx,target_id in enumerate(trainlist[user_id]):

		for idx,product_id in enumerate(boughtlist[user_id]): # number of product_id --(affect buying)--> target_id

			if not int(product_id-1) in prob_tree[int(target_id)-1]:
				prob_tree[int(target_id)-1][int(product_id)-1]=[0,0]

			val=boughtlist[user_id][int(product_id)]
			prob_tree[int(target_id)-1][int(product_id)-1][0]+=1
			prob_tree[int(target_id)-1][int(product_id)-1][1]+=val

			if not 'max' in prob_tree[int(target_id)-1]:
				prob_tree[int(target_id)-1]['max']=0
			if val>prob_tree[int(target_id)-1]['max']:	#Assign the maximum number of recently buying in any products that affect buying target_id product, we will use this information of the next step
				prob_tree[int(target_id)-1]['max']=val
			
			
t2 = int(round(time.time() * 1000))
print("Time usage for tree constructing (second): "+str((t2-t1)/1000))

def kmean(k,data,id): #K-mean clustering algorithm
	centroid=[0]*k

	for i in range(k):
		centroid[i]=data['max']*i/(k+1)

	del data['max']

	z1=[]
	
	for count in range(10):

		temp_centroid=[0]*k
		temp_product=[]

		for i in range(k):
			temp_product.append([])

		for idx,product_id in enumerate(data):
			min_dis=2**10
			min_ind=-1
			for i in range(k):
				dis=math.fabs(data[product_id][1]/data[product_id][0]-centroid[i]) #Find the distance between average number of buying with the centroid
				#dis=math.fabs(data[product_id][1]-centroid[i])
				if dis<min_dis:
					min_dis=dis
					min_ind=i

			temp_product[min_ind].append(product_id)
			temp_centroid[min_ind]+=data[product_id][1]/data[product_id][0]
			#temp_centroid[min_ind]+=data[product_id][1]
		
		for i in range(k):
			if len(temp_product[i])!=0:
				temp_centroid[i]=temp_centroid[i]/len(temp_product[i])

		centroid=temp_centroid

	return [centroid,temp_product]

t1 = int(round(time.time() * 1000))

model=[{} for _ in range(_NUM_PRODUCTS)]

for product_id in range(_NUM_PRODUCTS):
	if bool(prob_tree[product_id]):
		model[product_id]=kmean(2,prob_tree[product_id],product_id) 

"""
We will divide the information of buying product_id into two groups using k-mean clustering. We will classify these two groups as high-buying group and low-buying group.
High-buying group collects list of products with relative high number of buying that affect buying product_id comparing to Low-buying group.
Low-buying group collects list of products with relative low number of buying that affect buying product_id comparing to High-buying group.

the structure of model (variable) will be like this

model=[[Average number of buying in Low-buying group,Average number of buying in High-buying group],[Low-buying group,High-buying group]]

For example, model[13]=[[1.7,5.6],[[16,21,63,132,157],[48,68]]] implies that buying product number 13 is affected by
low number of buying product number 16,21,63,132,157 with average number of buying 1.7 and is affected by
high number of buying product number 48,68 with average number of buying 5.6

From this model, we may implies that if any user buy product number 48 around 5.6 times recently. there will be probability that he/she will buy product number 13 next time.

"""

t2 = int(round(time.time() * 1000))
print("Time usage for model constructing (second): "+str((t2-t1)/1000))

#Save data for prediction step
np.save('boughtlist.npy', boughtlist) #File collecting prior set for every users
np.save('tree.npy', prob_tree) #File containing data structure for predicting
np.save('model.npy', model) #File containing data structure for predicting
np.save('mapping.npy', train_to_user) #File containing data structure for predicting

print("Finish preprocessing!")