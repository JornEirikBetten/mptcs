# JAX 
import chex 
import distrax
import haiku as hk
import jax
import jax.numpy as jnp



"""
----------------------------------------------------
                FORWARD FUNCTION
----------------------------------------------------
"""
# FORWARD FUNCTION 
class ActorCritic(hk.Module):
    def __init__(self, num_actions, activation="tanh"):
        super().__init__()
        self.num_actions = num_actions
        self.activation = activation
        assert activation in ["relu", "tanh"]

    def __call__(self, x):
        x = x.astype(jnp.float32)
        if self.activation == "relu":
            activation = jax.nn.relu
        else:
            activation = jax.nn.tanh
        x = hk.Conv2D(32, kernel_shape=2)(x)
        x = jax.nn.relu(x)
        x = hk.avg_pool(x, window_shape=(2, 2), strides=(2, 2), padding="VALID")
        x = hk.Flatten()(x)
        x = hk.Linear(64)(x)
        x = jax.nn.relu(x)
        actor_mean = hk.Linear(64)(x)
        actor_mean = activation(actor_mean)
        actor_mean = hk.Linear(64)(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = hk.Linear(self.num_actions)(actor_mean)

        critic = hk.Linear(64)(x)
        critic = activation(critic)
        critic = hk.Linear(64)(critic)
        critic = activation(critic)
        critic = hk.Linear(1)(critic)

        return actor_mean, jnp.squeeze(critic, axis=-1)

def make_forward_fn(num_actions): 
    def forward_fn(x, is_eval=False, flag=None):
        net = ActorCritic(num_actions, activation="relu")
        logits, value = net(x)
        return logits, value
    return forward_fn

