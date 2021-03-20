import numpy as np
from PIL import Image
import json
import cStringIO
import urllib

# This script solves two issues in the dataset that remain after process.py
# 1. Some images are 'Not Found'
# 2. Some review duplicates exist in AMAZON_FASHION.json
# Users that have any reviews with one of these problems are tossed out entirely to avoid complicated logic

# Import the dataset
dataset_name = 'product_only.npz'
dataset = np.load(dataset_name)

user_train = dataset['user_train'].item()
user_validation = dataset['user_validation'].item()
user_test = dataset['user_test'].item()
Item = dataset['Item'].item()
usernum = dataset['usernum']
itemnum = dataset['itemnum']

# Discard users that have an issue with one of their reviews
user_train_new = {}
user_test_new = {}
user_validation_new = {}
count = 0
for i in range(usernum):
    skip = False

    # Check for duplicates
    prod_ids = [rev['productid'] for rev in user_train[i]]
    prod_ids.append(user_test[i][0]['productid'])
    prod_ids.append(user_validation[i][0]['productid'])

    if any(prod_ids.count(x) > 1 for x in prod_ids):
        skip = True

    # Check for broken links
    for review in user_train[i]:
        j = review['productid']
        if Item[j]['imgs'] == 'Not Found':
            skip = True
    
    j = user_test[i][0]['productid']
    if Item[j]['imgs'] == 'Not Found':
        skip = True

    j = user_validation[i][0]['productid']
    if Item[j]['imgs'] == 'Not Found':
        skip = True

    # Add users without issues to the new sets
    if not skip:
        user_train_new[count] = user_train[i]
        user_test_new[count] = user_test[i]
        user_validation_new[count] = user_validation[i]
        count += 1
    
# Discard items whose images are 'Not Found'
Item_new = {}
old_to_new_id = {}
count = 0
for i in range(itemnum):
    if not (Item[i]['imgs'] == 'Not Found'):
        Item_new[count] = Item[i]
        old_to_new_id[i] = count
        count += 1

# Fix the productid's of the reviews
for i in range(len(user_train_new)):
    for review in user_train_new[i]:
        review['productid'] = old_to_new_id[review['productid']]
    
    user_test_new[i][0]['productid'] = old_to_new_id[user_test_new[i][0]['productid']] 

    user_validation_new[i][0]['productid'] = old_to_new_id[user_validation_new[i][0]['productid']] 

# Save new dataset
np.savez(dataset_name[0:-4] + str("_final.npz"),
            user_train = user_train_new,
            user_validation = user_validation_new,
            user_test = user_test_new,
            Item = Item_new,
            usernum = len(user_train_new),
            itemnum = len(Item_new))

print("Number of users lost: " + str(usernum - len(user_train_new)))
print("Number of items lost: " + str(itemnum - len(Item_new)))
print("New number of users: " + str(len(user_train_new)))
print("New number of items: " + str(len(Item_new)))
