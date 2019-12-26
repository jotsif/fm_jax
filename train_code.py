import pandas as pd
import numpy as onp
from warp import fit, prepare_data, predict

# Get positive reviews
reviews = pd.read_pickle("data/reviews.pkl").query("rating > 2")

user_feature_cols = ["gPlusUserId"]
item_feature_cols = ["gPlusPlaceId"]

item_data, user_data, item_dataset, item_map, user_map = \
    prepare_data(reviews[user_feature_cols + item_feature_cols],
                 user_feature_cols,
                 item_feature_cols)

params = fit(user_data,
             item_data,
             item_dataset,
             batch_size=100000)

users_subsample = onp.random.permutation(onp.unique(user_data))[:10000]

preds = predict(params, users_subsample, item_dataset)

# u_id = users_subsample[0]
# interactions = item_data[user_data == u_id]
# itx = onp.zeros_like(item_dataset)
# itx[interactions] = 1

# auc = calc_auc(preds[0], itx)
