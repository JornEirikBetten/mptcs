import os 
import jax 
import jax.numpy as jnp 

import pandas as pd 


"""
----------------------------------------------------
                LOAD PARAMETERS
----------------------------------------------------
"""

def load_parameters(policy_path):
    parameters = jnp.load(
        policy_path,
        allow_pickle=True,
    )
    return parameters

def load_stacked_params(policy_indices, env_name, path_to_policies):
    policy_paths = [
        path_to_policies + f"{env_name}/ppo/seed=1234-policy-{p}.pkl"
        for p in policy_indices
    ]
    params_list = [load_parameters(p) for p in policy_paths]
    params_stacked = jax.tree.map(lambda *args: jnp.stack(args), *params_list)
    return params_stacked, params_list 

    
def load_params(num_policies, env_name, path_to_eval_data, path_to_policies, load_rashomon=True): 
    poleval = pd.read_csv(path_to_eval_data + f"/{env_name}/evaluation_of_policies.csv")
    
    assert num_policies*2 <= len(poleval), "num_policies must be less than or equal to half of the number of policies in the evaluation data"

    # TODO Evaluation should use more than num_policies policies

    # Prepare policy parameters
    if load_rashomon:
        # We just take the top-k policies
        # We take every second nlargest policy, the others go into the test set
        top_policies = poleval.nlargest(num_policies * 2, 'rewards_mean')
        policy_indices = top_policies.iloc[::2]['policy_index'].tolist()
        test_policy_indices = top_policies.iloc[1::2]['policy_index'].tolist()
    else:
        policy_indices = jnp.arange(1, num_policies + 1)
        test_policy_indices = jnp.arange(num_policies + 1, num_policies + num_policies + 1)
        
    params_stacked, params_list = load_stacked_params(policy_indices, env_name, path_to_policies)
    test_params_stacked, test_params_list = load_stacked_params(test_policy_indices, env_name, path_to_policies)
    return params_stacked, params_list, test_params_stacked, test_params_list


def load_params_with_fixed_num_test_policies(num_policies, num_test_policies, env_name, path_to_eval_data, path_to_policies, load_rashomon=True):
    poleval = pd.read_csv(path_to_eval_data + f"/{env_name}/evaluation_of_policies.csv")
    # Prepare policy parameters
    if load_rashomon:
        # We just take the top-k policies
        # We take every second nlargest policy, the others go into the test set
        top_policies = poleval.nlargest(num_policies + num_test_policies, 'rewards_mean')
        top_policies = top_policies['policy_index'].tolist()
        policy_indices = [] 
        test_policy_indices = []
        for i in range(num_policies): 
            policy_indices.append(top_policies.pop(0))
            test_policy_indices.append(top_policies.pop(0))
        
        test_policy_indices = test_policy_indices + top_policies
        assert len(policy_indices) == num_policies
        assert len(test_policy_indices) == num_test_policies
    else:
        policy_indices = jnp.arange(1, num_policies + 1)
        test_policy_indices = jnp.arange(num_policies + 1, num_policies + num_policies + 1)
    
    print(f"Loading {len(policy_indices)} policies for the search set.")
    print(f"Loading {len(test_policy_indices)} test policies for the evaluation set.")
    params_stacked, params_list = load_stacked_params(policy_indices, env_name, path_to_policies)
    test_params_stacked, test_params_list = load_stacked_params(test_policy_indices, env_name, path_to_policies)
    return params_stacked, params_list, test_params_stacked, test_params_list
    
