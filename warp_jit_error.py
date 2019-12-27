from jax import random
from warp import loss, warp, initial_params
import jax.numpy as np
from jax import jit

z = 50

params = initial_params(10, 10, z)

key = random.PRNGKey(0)


a = jit(warp, static_argnums=(1, 2, 3, 4))(
    params, z, 10, np.array([[1], [2], [3], [4]]),
    None, np.array([[1]]), np.array([[3]]), key)

print(f"This works {a}")

# but this doesn't
jit(loss, static_argnums=(1, 2, 3, 4))(
    params, z, 10, np.array([[1], [2], [3], [4]]),
    None, np.array([[1]]), np.array([[3]]), key)
