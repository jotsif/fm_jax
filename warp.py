import jax.numpy as np
from jax import grad, jit, vmap
from jax import lax
from jax.experimental import optimizers
# import time
import itertools
# Current convention is to import original numpy as "onp"
import numpy as onp
from tqdm import tqdm
import numpy.random as npr


def make_embedding_map(matrix):
    """
    Return a map from vector categories to integers
    """
    _map = dict([(k, v) for v, k in enumerate(onp.unique(matrix))])
    return _map


@jit
def compute_representation(feature_values, feature_embeddings, feature_bias):
    """
    Compute vector representation of current row id using the feature values,
    the embeddings and the bias as a weighted sum a la LightFM.
    feature_values is a key value dict
    """
    def get_embedding(key):
        return feature_embeddings[key]

    embs = vmap(get_embedding)(feature_values)

    return np.sum(embs, axis=0)


def get_negative_item(params, user_repr, score_pos,
                      item_dataset, max_samples):
    """Sample a negative item ranked higher than the positive item"""
    n_items = item_dataset.shape[0]

    def resample(state):
        _, _, sampled = state
        sampled = sampled + 1
        neg_item = npr.randint(0, n_items)
        neg_repr = compute_representation(item_dataset[neg_item],
                                          params[ITEM_FEATURE_EMBEDDING_IDX],
                                          params[ITEM_BIAS_IDX])
        return (neg_item, np.dot(user_repr, neg_repr), sampled)

    def cond(state):
        (_, score_neg, sampled) = state
        return (score_neg < score_pos - 1) & (sampled < max_samples)

    (neg_item, _, sampled) = lax.while_loop(cond, resample,
                                            (0, score_pos + 1, 0))
    return neg_item, sampled


def warp(params,
         item_dataset,
         user_data,
         item_data,
         max_samples=10):
    """
    Calculate rank constant and score error for a given (u, i)-pair
    """
    n_items = item_dataset.shape[0]
    user_repr = compute_representation(user_data,
                                       params[USER_FEATURE_EMBEDDING_IDX],
                                       params[USER_BIAS_IDX])
    pos_repr = compute_representation(item_data,
                                      params[ITEM_FEATURE_EMBEDDING_IDX],
                                      params[ITEM_BIAS_IDX])
    # Bias?
    score_pos = np.dot(user_repr, pos_repr)
    neg_item, sampled = get_negative_item(lax.stop_gradient(params),
                                          lax.stop_gradient(user_repr),
                                          lax.stop_gradient(score_pos),
                                          item_dataset,
                                          max_samples)
    # Recalculate negative representation since
    # while_loop is not differentiable yet in jax

    def loss():
        neg_repr = compute_representation(item_dataset[neg_item],
                                          params[ITEM_FEATURE_EMBEDDING_IDX],
                                          params[ITEM_BIAS_IDX])
        score_neg = np.dot(user_repr, neg_repr)
        # Run constant with original numpy?
        L = np.floor((n_items - 1)/sampled)
        constant = np.log(np.where(1.0 > L, 1.0, L))
        loss = score_neg - score_pos + 1
        return constant * loss

    loss = np.where(sampled < max_samples, loss(), 0)
    return loss


# TODO: Is there a quick nicer solution?
USER_FEATURE_EMBEDDING_IDX = 0
ITEM_FEATURE_EMBEDDING_IDX = 1
USER_BIAS_IDX = 2
ITEM_BIAS_IDX = 3


def initial_params(n_user_features, n_item_features, z):
    """
    TODO: change initialisation distribution?
    """
    return [
        onp.random.randn(n_user_features, z),  # user feature embeddings
        onp.random.randn(n_item_features, z),  # item feature embeddings
        onp.random.randn(n_user_features, 1),  # user feature bias
        onp.random.randn(n_item_features, 1),  # item feature bias
    ]


def loss(params, item_dataset,
         user_data, item_data):
    """Convenience function for loss"""
    res = vmap(warp, in_axes=(None, None, 0, 0))(
        params, item_dataset,
        user_data, item_data)
    return np.mean(res)


def data_stream(user_data, item_data,
                num_batches, batch_size):
    rng = npr.RandomState(0)
    num_train = user_data.shape[0]
    while True:
        perm = rng.permutation(num_train)
        for i in range(num_batches):
            batch_idx = perm[i * batch_size:(i + 1) * batch_size]
            yield user_data[batch_idx], item_data[batch_idx]


def prepare_data(interactions, user_feature_cols, item_feature_cols):
    """
    Prepare matrices for training
    """
    user_df = interactions[user_feature_cols]
    item_df = interactions[item_feature_cols]
    item_dataset = item_df.drop_duplicates()
    item_map = make_embedding_map(item_dataset)
    user_map = make_embedding_map(user_df)
    # Remap dataframe categories to integers
    item_data = item_df.apply(lambda r: r.apply(lambda x: item_map[x]))
    item_dataset = item_dataset.apply(lambda r: r.apply(lambda x: item_map[x]))
    user_data = user_df.apply(lambda r: r.apply(lambda x: user_map[x]))
    return item_data.to_numpy(), user_data.to_numpy(), \
        item_dataset.to_numpy(), item_map, user_map


def fit(user_data, item_data, item_dataset,
        num_epochs=1, step_size=0.001, batch_size=100, z=50):
    user_data = np.array(user_data)
    item_data = np.array(item_data)
    item_dataset = np.array(item_dataset)
    """
    user_feature is a list of column names for the user features
    item_feature is a list of column names for the item features
    """
    if user_data.shape[0] != item_data.shape[0]:
        raise ValueError("User and item data not of the same shape")

    opt_init, opt_update, get_params = optimizers.adam(step_size)

    num_batches = int(item_data.shape[0]/batch_size)
    # take two matrices as argument instead?
    batches = data_stream(user_data, item_data,
                          num_batches, batch_size)
    itercount = itertools.count()
    opt_state = opt_init(initial_params(np.max(user_data) + 1,
                                        np.max(item_data) + 1,
                                        z))

    def update(i, opt_state, batch):
        params = get_params(opt_state)
        user_data, item_data = batch
        return opt_update(i,
                          grad(loss)(params, item_dataset,
                                     user_data, item_data),
                          opt_state)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
#        start_time = time.time()
        for _ in tqdm(range(num_batches)):
            opt_state = update(next(itercount), opt_state, next(batches))
    params = get_params(opt_state)
    return params
