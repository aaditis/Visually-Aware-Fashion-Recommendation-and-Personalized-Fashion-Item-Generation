import numpy as np
from PIL import Image
import json
import cStringIO
import urllib

def remove_item_reviews(asins, user_dict):
    for user in user_dict:
        delete_from_list = []
        for review in user_dict[user]:
            if review['asin'] in asins:
                delete_from_list.append(review)
        for review in delete_from_list:
            user_dict[user].remove(review)


def main():
    # Pre-processing parameters
    use_user_images = False
    npz_name = 'product_only.npz'

    # Each line of the datasets is a json object, convert them line by line
    user_reviews = [json.loads(line) for line in open('AMAZON_FASHION.json', 'r')]
    item_data = [json.loads(line) for line in open('meta_AMAZON_FASHION.json', 'r')]

    # Create a dictionary of user reviews grouped by user, and item image urls grouped by item
    user_dict = {}
    for review in user_reviews:
        if review['reviewerID'] in user_dict:
            user_dict[review['reviewerID']].append(review)
        else:
            user_dict[review['reviewerID']] = [review]

    # Create new dictionary with only users that have five or more reviews
    user_dict_five = {}
    for user in user_dict:
        if len(user_dict[user]) >= 5:
            user_dict_five[user] = user_dict[user]

    # Group item image urls by item, using only items reviewed by the reviewers in user_dict_five
    item_image_urls = {}
    for user in user_dict_five:
        for review in user_dict_five[user]:
            if not (review['asin'] in item_image_urls):
                item_image_urls[review['asin']] = []
            if 'image' in review and use_user_images:
                item_image_urls[review['asin']].extend(review['image'])

    # Find the product image urls for the items reviewed by the super users
    item_image_urls_product_index = {} # Index of the first product image in the image list
    for item in item_data:
        if item['asin'] in item_image_urls:
            if len(item['image']) > 0:
                item_image_urls_product_index[item['asin']] = len(item_image_urls[item['asin']])
            else:
                item_image_urls_product_index[item['asin']] = 0
            item_image_urls[item['asin']].extend(item['image'])

    # If an item does not have any images, remove all reviews of the item and discard it
    items_wo_images = []
    for item in item_image_urls:
        if len(item_image_urls[item]) == 0:
            items_wo_images.append(item)
    remove_item_reviews(items_wo_images, user_dict_five)

    # If a user no longer has five reviews, remove them from the list
    delete_from_users = []
    for user in user_dict_five:
        if len(user_dict_five[user]) < 5:
            delete_from_users.append(user)
    for user in delete_from_users:
        del user_dict_five[user]

    # Construct the sets for the .npz file. Treat each image as a separate product.
    product_id = 0
    user_train = {}
    user_validation = {}
    user_test = {}
    Item = {}
    item_asin_to_product_ids = {}
    tot_review_count = 0
    num_users = float(len(user_dict_five))
    for user_count, user in enumerate(user_dict_five):
        print(100 * (user_count / num_users))
        user_train[user_count] = []
        for review_count, review in enumerate(user_dict_five[user]):
            tot_review_count += 1
            asin = review['asin']
            if not (asin in item_asin_to_product_ids):
                item_asin_to_product_ids[asin] = []
                for image_url in item_image_urls[asin]:
                    item_asin_to_product_ids[asin].append(product_id)
                    Item[product_id] = {}
                    Item[product_id]['asin'] = asin
                    Item[product_id]['imgurl'] = image_url
                    Item[product_id]['imgs'] = urllib.urlopen(image_url).read()
                    #Item[product_id]['imgs'] = 'testrun'
                    product_id +=1
            if review_count == 0:
                user_review = {}
                user_review['asin'] = asin
                product_index = item_image_urls_product_index[asin] # Use the product image if possible in the validation set
                user_review['productid'] = item_asin_to_product_ids[asin][product_index]
                user_validation[user_count] = [user_review]
            elif review_count == 1:
                user_review = {}
                user_review['asin'] = asin
                product_index = item_image_urls_product_index[asin] # Use the product image if possible in the test set
                user_review['productid'] = item_asin_to_product_ids[asin][product_index]
                user_test[user_count] = [user_review]
            else:
                user_review_list = []
                for prod_id in item_asin_to_product_ids[asin]:
                    user_review = {}
                    user_review['asin'] = asin
                    user_review['productid'] = prod_id
                    user_review_list.append(user_review)
                user_train[user_count].extend(user_review_list)
    usernum = len(user_train)
    itemnum = len(Item)

    # Construct npz file
    np.savez(npz_name,
             user_train = user_train,
             user_validation = user_validation,
             user_test = user_test,
             Item = Item,
             usernum = usernum,
             itemnum = itemnum)

    # Compute user submitted image statistics
    num_img_rev = 0
    tot_user_img_num = 0
    for user in user_dict_five:
        for review in user_dict_five[user]:
            if 'image' in review and use_user_images:
                num_img_rev += 1
                tot_user_img_num += len(review['image'])
    
    # Print statistics
    print("OLD DATASET STATISTICS (Amazon Fashion):")
    print("Number of unique reviewers: " + str(64583))
    print("Number of reviewed items: " + str(234892))
    print("Number of super users: " + str(45184))
    print("Number of items reviewed by super users: " + str(166270))

    print("NEW DATASET STATISTICS (Amazon Review Data 2018, Amazon Fashion):")
    print("Total number of user reviews: " + str(len(user_reviews)))
    print("Total number of reviewed items: " + str(len(item_data)))
    print("Number of unique reviewers: " + str(len(user_dict.keys())))
    print("Number of super users (reviewers with five or more reviews on items with at least one image): " + str(len(user_dict_five.keys())))
    print("Number of items reviewed by super users: " + str(tot_review_count))
    print("Number of super user reviews including images: " + str(num_img_rev))
    print("Total number of images submitted in super user reviews: " + str(tot_user_img_num))
    print("Total number of item images: " + str(len(Item)))

if __name__ == "__main__":
    main()