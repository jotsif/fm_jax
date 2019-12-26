import jax.numpy as np
from jax import grad, jit, vmap
from jax import lax, random
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


def compute_representation(feature_values, feature_embeddings, feature_bias):
    """
    Compute vector representation of current row id using the feature values,
    the embeddings and the bias as a weighted sum a la LightFM.
    feature_values is a key value dict
    """
    def get_embedding(key):
        feature_vector = feature_embeddings[key]
        return np.append(feature_vector, feature_bias[key])

    embs = vmap(get_embedding)(feature_values)

    return np.sum(embs, axis=0)


def calc_score(user_repr, item_repr, z):
    return np.dot(user_repr[:z], item_repr[:z]) + user_repr[z] + item_repr[z]


def get_negative_item(params, user_repr, score_pos,
                      item_dataset, interactions, max_samples, z, key):
    """Sample a negative item ranked higher than the positive item"""
    n_items = item_dataset.shape[0]

    def resample(state):
        _, _, sampled, key = state
        sampled = sampled + 1
        key, key_ = random.split(key)
        neg_item = random.randint(key_, (1, ), 0, n_items)
        # TODO: should not sample positive item
        neg_repr = compute_representation(item_dataset[neg_item],
                                          params[ITEM_FEATURE_EMBEDDING_IDX],
                                          params[ITEM_BIAS_IDX])
        return (neg_item, calc_score(user_repr, neg_repr, z), sampled, key)

    def cond(state):
        _, score_neg, sampled, _ = state
        return (score_neg < score_pos - 1) & (sampled < max_samples)

    (neg_item, _, sampled, _) = lax.while_loop(cond, resample,
                                               (np.array([0]), score_pos - 2,
                                                0, key))
    return neg_item, sampled


# @partial(jit, static_argnums=(1, 2, 3))
def warp(params,
         z,
         max_samples,
         item_dataset,
         interactions,
         user_data,
         item_data,
         key):
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

    score_pos = calc_score(user_repr, pos_repr, z)

    neg_item, sampled = get_negative_item(lax.stop_gradient(params),
                                          lax.stop_gradient(user_repr),
                                          lax.stop_gradient(score_pos),
                                          item_dataset,
                                          interactions,
                                          max_samples,
                                          z,
                                          key)
#    neg_item = 0
#    sampled = 1
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

    # If we can't find a negative example skip this positive pair
    loss = np.where(sampled < max_samples, loss(), 0)
    return loss


# TODO: Is there a quick nicer solution?
USER_FEATURE_EMBEDDING_IDX = 0
ITEM_FEATURE_EMBEDDING_IDX = 1
USER_BIAS_IDX = 2
ITEM_BIAS_IDX = 3


def initial_params(n_user_features, n_item_features, z):
    """
    Using same initisialisation as lightfm
    """
    return [
        np.array(onp.random.rand(n_user_features, z) - 0.5)/z,  # user feature embeddings
        np.array(onp.random.rand(n_item_features, z) - 0.5)/z,  # item feature embeddings
        np.zeros(n_user_features),  # user feature bias
        np.zeros(n_item_features),  # item feature bias
    ]


def loss(params, z, max_samples, item_dataset, interactions,
         user_data, item_data, key):
    """Convenience function for loss"""
    batch_size = user_data.shape[0]
    keys = random.split(key, batch_size)
    res = vmap(warp, in_axes=(None, None, None, None, None, 0, 0, 0))(
        params, z, max_samples, item_dataset, interactions,
        user_data, item_data, keys)
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
        num_epochs=1, step_size=0.1, batch_size=100, z=50,
        max_samples=10, seed=0):
    """
    user_feature is a list of column names for the user features
    item_feature is a list of column names for the item features
    """
    n_users = user_data.max() + 1
    n_items = item_data.max() + 1
    user_data = np.array(user_data)
    item_data = np.array(item_data)
    item_dataset = np.array(item_dataset)
    if user_data.shape[0] != item_data.shape[0]:
        raise ValueError("User and item data not of the same shape")

    opt_init, opt_update, get_params = optimizers.adam(step_size)

    num_batches = int(item_data.shape[0]/batch_size)
    batches = data_stream(user_data, item_data,
                          num_batches, batch_size)
    itercount = itertools.count()
    opt_state = opt_init(initial_params(n_users,
                                        n_items,
                                        z))
    key = random.PRNGKey(seed)
    interactions = 0

    @jit
    def update(i, opt_state, batch, key):
        params = get_params(opt_state)
        user_data, item_data = batch
        grad_loss = grad(loss)(params, z, max_samples,
                               item_dataset, interactions,
                               user_data, item_data, key)
        return opt_update(i,
                          grad_loss,
                          opt_state)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for _ in tqdm(range(num_batches)):
            key, key_ = random.split(key)
            opt_state = update(next(itercount), opt_state,
                               next(batches), key_)
    params = get_params(opt_state)
    return params


@jit
def calc_auc(users, predictions, interactions):
    """
    Calculate the AUC for all users given their predictions
    and item interactions.
    """
    n_items = predictions.shape[1]

    def per_user(preds, interacts):
        n_pos = interacts.sum()
        pos = np.flip(np.argsort(preds))
        ranks = interacts[pos].nonzero()[0]
        r1 = ranks.sum()
        u1 = r1 - n_pos * (n_pos + 1)/2
        return u1/(n_pos * (n_items - n_pos))
    return vmap(per_user)(users, predictions)


@jit
def predict(params, users, items):
    z = params[USER_FEATURE_EMBEDDING_IDX].shape[1]
    users = np.array(users)
    items = np.array(items)

    def score_user(params, user):
        user_repr = compute_representation(
            np.array([user]),  # should not have to do this
            params[USER_FEATURE_EMBEDDING_IDX],
            params[USER_BIAS_IDX])

        def score_item(item):
            item_repr = compute_representation(
                item,
                params[ITEM_FEATURE_EMBEDDING_IDX],
                params[ITEM_BIAS_IDX])
            score = calc_score(user_repr, item_repr, z)
            return score
        return vmap(score_item)(items)
    score = vmap(score_user, in_axes=(None, 0))(params, users)
    return score
