# %%
import os
import sys
import traceback

import haiku as hk
import jax
import pgx
import pgx.minatar
from omegaconf import OmegaConf
from pydantic import BaseModel

import ga_utils
from pats import forward_fns, param_loader
from pats import simulators as runners 

ENVIRONMENTS = (
    "minatar-asterix",
    "minatar-breakout",
    "minatar-freeway",
    "minatar-seaquest",
    "minatar-space_invaders",  # TODO: Fix doesn't terminate
)


class GATestMutatorsConfig(BaseModel):
    seed: int = 10
    num_policies: int = 10
    num_test_steps: int = 3

    # GA Setup
    pop_size: int = 15
    num_iterations: int = 1

    class Config:
        extra = "forbid"


args = GATestMutatorsConfig(**OmegaConf.to_object(OmegaConf.from_cli()))
print(args, file=sys.stderr)

# PATHS
path_to_eval_data = os.getcwd() + "/evaluation_data/"
path_to_trained_policies = os.getcwd() + "/trained_policies/"

for env_name in ENVIRONMENTS:
    # ENVIRONMENT FUNCTIONS
    env = pgx.make(env_name)
    n_actions = env.num_actions
    print(f"{env_name}: {n_actions} actions")
    env_step = jax.jit(jax.vmap(env.step))
    env_init = jax.jit(jax.vmap(env.init))

    # Forward function
    forward_fn = forward_fns.make_forward_fn(env.num_actions)
    forward = hk.without_apply_rng(hk.transform(forward_fn))

    run = runners.build_evaluate_function(forward, env, args.num_test_steps)
    batched_run = jax.vmap(run, in_axes=(0, None, None))
    jitted_batched_run = jax.jit(batched_run)

    params_stacked, params_list, _, _ = param_loader.load_params(
        args.num_policies,
        env_name,
        path_to_eval_data,
        path_to_trained_policies,
        load_rashomon=True,
    )

    rng = jax.random.PRNGKey(args.seed)
    rng, rng_init = jax.random.split(rng)
    # It's very important to make the initial state with vmap, otherwise the batching dimension is missing
    reference_state = env_init(jax.random.split(rng_init, 1))

    mutators = ga_utils.MUTATORS[env_name]
    observe_fn = ga_utils.OBSERVE_FN[env_name]
    mutation_strength: float = 0.5  # TODO: Make this a parameter, default too high
    # TODO Figure out good default mutation strength
    for it in range(args.num_iterations):
        initial_states = env_init(jax.random.split(rng_init, args.pop_size))

        rng, rng_mut, rng_init, rng_exec = jax.random.split(rng, 4)

        for field, mutate_func in mutators.items():
            print(f"Mutating {field}")
            try:
                new_values = jax.vmap(
                    lambda s, k: mutate_func(s, k, strength=mutation_strength)
                )(initial_states, jax.random.split(rng_mut, args.pop_size))
                # print(new_values)
            except Exception as e:
                print(f"Error mutating {field}: {e}")
                print(traceback.format_exc())
                continue
            # break
            new_states = initial_states.replace(**{field: new_values})
            new_observation = jax.vmap(observe_fn)(new_states)
            mutated_states = new_states.replace(observation=new_observation)
            trajs = jitted_batched_run(params_stacked, mutated_states, rng_exec)

            if trajs.state.terminated.any():
                print(
                    f"{env_name}: {field} can cause termination within {args.num_test_steps} steps"
                )
