#
# Read google local "jsons"
#

import pandas as pd
import ast
from tqdm import tqdm

# Downloaded from http://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local
places_file = "data/places.clean.json"
reviews_file = "data/reviews.clean.json"
users_file = "data/users.clean.json"

places_lines = sum(1 for line in open(places_file))

# Read places
with open(places_file, "r") as f:
    a = []
    for i in tqdm(range(places_lines)):
        s = f.readline()
        j = ast.literal_eval(s)
        gps = j['gps']
        address = j['address'] if j['address'] is not None else []
        hours = j['hours']
        phone = j['phone']
        l_ = [j['name'],
              j['gPlusPlaceId'],
              j['price'],
              hours is not None,
              phone is not None,
              gps[0] if gps is not None else None,
              gps[1] if gps is not None else None,
              address[0] if len(address) > 0 else None,
              address[1] if len(address) > 1 else None,
              address[2] if len(address) > 2 else None,
              j['closed']]
        a.append(l_)

places = pd.DataFrame(a)
places.columns = ['name', 'gPlusPlaceId', 'price', 'has_hours', 'has_phone',
                  'lat', 'long', 'adress1', 'address2', 'address3', 'closed']

user_lines = sum(1 for line in open(users_file))

# Read users
with open(users_file, "r") as f:
    a = []
    for i in tqdm(range(user_lines)):
        s = f.readline()
        j = ast.literal_eval(s)
        education = j['education'] if j['education'] is not None else []
        jobs = j['jobs'] if j['jobs'] is not None else []
        l_ = [j['gPlusUserId'],
              j['userName'],
              education[0] if len(education) > 0 else None,
              education[1] if len(education) > 1 else None,
              jobs[0] if len(jobs) > 0 else None,
              jobs[1] if len(jobs) > 1 else None]
        a.append(l_)

users = pd.DataFrame(a)
users.columns = ['gPlusUserId', 'name', 'education0', 'education1', 'jobs0', 'jobs1']
# Read reviews

reviews_lines = sum(1 for line in open(reviews_file))

with open(reviews_file, "r") as f:
    a = []
    for i in tqdm(range(reviews_lines)):
        s = f.readline()
        j = ast.literal_eval(s)
        cats = j['categories'] if j['categories'] is not None else []
        l_ = [j['rating'],
              j['gPlusPlaceId'],
              j['unixReviewTime'],
              j['gPlusUserId'],
              cats[0] if len(cats) > 0 else None,
              cats[1] if len(cats) > 1 else None]
        a.append(l_)

reviews = pd.DataFrame(a)
reviews.columns = ['rating', 'gPlusPlaceId', 'unixReviewTime', 'gPlusUserId',
                   'category0', 'category1']


places.to_pickle("data/places.pkl")
users.to_pickle("data/users.pkl")
reviews.to_pickle("data/reviews.pkl")
