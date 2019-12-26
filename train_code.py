import pandas as pd
from warp import fit, prepare_data

# Get positive reviews
reviews = pd.read_pickle("data/reviews.pkl").query("rating > 2")

user_feature_cols = ["gPlusUserId"]
item_feature_cols = ["gPlusPlaceId"]

item_data, user_data, item_dataset, item_map, user_map = \
    prepare_data(reviews[user_feature_cols + item_feature_cols],
                 user_feature_cols,
                 item_feature_cols)

res = fit(user_data,
          item_data,
          item_dataset,
          batch_size=100000)
