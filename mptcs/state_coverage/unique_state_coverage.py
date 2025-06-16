import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import jax
from minimal_pats.state_coverage.minatar_state_coverages import _map_var_to_trajectory_field
import math


def get_unique_state_coverage(trajectory, environment_state_space) -> Dict[str, float]:
    """
    Calculate the coverage of unique full states visited in the trajectory.
    
    This differs from get_state_coverage in that it considers the combination of
    all state variables together, instead of individually. It represents how many
    unique full states (combinations of variable values) were visited compared to
    the total number of possible states in the environment.
    
    Args:
        trajectory: A Pgx.State object containing the full trajectory with shape 
                    (executed_policies, time_step, num_initial_states, object.shape)
        environment_state_space: An environment-specific state coverage object
                                 that provides possible values for each state variable
    
    Returns:
        A dictionary with metrics including the percentage of 
        unique states visited relative to the possible environment state space
    """
    complete_state_space = environment_state_space.get_complete_state_space()
    coverage_metrics = {}
    
    # Calculate total possible state space size
    total_possible_states = _calculate_state_space_size(complete_state_space)
    coverage_metrics["total_possible_states"] = total_possible_states
    
    # Get trajectory dimensions
    num_policies = trajectory._step_count.shape[0]
    num_timesteps = trajectory._step_count.shape[1]
    num_init_states = trajectory._step_count.shape[2]
    
    # Set to store unique state fingerprints (as tuples)
    unique_states = set()
    total_states_visited = 0
    
    # Iterate through all states in the trajectory
    for p in range(num_policies):
        for t in range(num_timesteps):
            for s in range(num_init_states):
                # Skip terminated states
                if hasattr(trajectory, '_terminal') and trajectory._terminal[p, t, s]:
                    continue
                
                # Create a state fingerprint as a tuple of values
                state_fingerprint = _create_state_fingerprint(trajectory, p, t, s, complete_state_space)
                
                if state_fingerprint:
                    unique_states.add(state_fingerprint)
                    total_states_visited += 1
    
    # Calculate metrics
    coverage_metrics["unique_states_count"] = len(unique_states)
    coverage_metrics["total_states_visited"] = total_states_visited
    
    if total_states_visited > 0:
        coverage_metrics["unique_visited_percentage"] = (len(unique_states) / total_states_visited) * 100.0
    else:
        coverage_metrics["unique_visited_percentage"] = 0.0
    
    # Calculate the percentage of possible states that were visited
    if total_possible_states > 0:
        coverage_metrics["unique_states_coverage"] = (len(unique_states) / total_possible_states) * 100.0
    else:
        coverage_metrics["unique_states_coverage"] = 0.0
    
    return coverage_metrics


def _calculate_state_space_size(complete_state_space):
    """
    Calculate the total number of possible states in the environment.
    
    Args:
        complete_state_space: The complete state space dictionary
        
    Returns:
        The total number of possible states, or an estimate if the exact number is too large
    """
    # Start with 1 (for the product)
    total_states = 1
    
    # For each state variable, multiply by the number of possible values
    for var_name, var_info in complete_state_space.items():
        field_info = _map_var_to_trajectory_field(var_name)
        if field_info is None:
            continue
            
        field_name, _ = field_info
        
        # Handle different field types
        if field_name == "_entities":
            # Entities typically have positions in a grid
            # Assuming a grid size from the var_info if available
            if "grid_size" in var_info:
                grid_height, grid_width = var_info["grid_size"]
                # For each entity, it can be in any grid position or absent
                entity_states = (grid_height * grid_width) + 1  # +1 for absent
                # If we know max_entities, use that, otherwise assume 1
                max_entities = var_info.get("max_entities", 1)
                # Multiply by the number of possible states for each entity
                total_states *= entity_states ** max_entities
            else:
                # Fallback: assume 10x10 grid and 1 entity if no specific info
                total_states *= 101  # 100 positions + 1 for absent
        
        elif field_name in ["_brick_map", "_alien_map", "_f_bullet_map", "_e_bullet_map"]:
            # Boolean maps have 2^(height*width) possible states
            if "grid_size" in var_info:
                grid_height, grid_width = var_info["grid_size"]
                # Limit the calculation to avoid overflow
                if grid_height * grid_width <= 20:  # Only calculate exactly for small grids
                    total_states *= 2 ** (grid_height * grid_width)
                else:
                    # For larger grids, use a more conservative estimate
                    # We'll assume typical games don't use all possible map configurations
                    # and estimate based on observed density
                    density = var_info.get("typical_density", 0.1)  # Default 10% filled
                    avg_active_cells = int(grid_height * grid_width * density)
                    # Binomial coefficient: ways to choose active cells
                    if avg_active_cells > 0:
                        combinations = math.comb(grid_height * grid_width, avg_active_cells)
                        total_states *= min(combinations, 1_000_000)  # Cap to avoid overflow
                    else:
                        total_states *= grid_height * grid_width  # Linear approximation
            else:
                # Fallback: assume a small 5x5 grid if no specific info
                total_states *= 2 ** 25
        
        elif field_name in ["_f_bullets", "_e_fish", "_e_subs", "_divers"]:
            # Similar to entities, but typically bullets/objects have simpler state
            if "grid_size" in var_info:
                grid_height, grid_width = var_info["grid_size"]
                max_items = var_info.get("max_items", 5)  # Default max 5 bullets/entities
                # Each item can be in any position or absent
                item_states = (grid_height * grid_width) + 1
                
                # Limit calculation to avoid overflow
                if max_items <= 3:
                    total_states *= item_states ** max_items
                else:
                    # For many possible items, use a more conservative estimate
                    total_states *= (item_states ** 3) * max_items  # Approximate
            else:
                # Fallback: assume 10x10 grid and max 3 items
                total_states *= 101 ** 3  # (100 positions + 1 for absent)^3
        
        else:
            # For scalar variables, use the number of possible values if known
            if "values" in var_info and isinstance(var_info["values"], list):
                num_values = len(var_info["values"])
                total_states *= num_values
            # If we have min/max, calculate range
            elif "min" in var_info and "max" in var_info:
                min_val = var_info["min"]
                max_val = var_info["max"]
                # Check if discrete or continuous
                if isinstance(min_val, int) and isinstance(max_val, int):
                    num_values = max_val - min_val + 1
                    total_states *= num_values
                else:
                    # For continuous variables, use a discretization
                    discretization = var_info.get("discretization", 10)
                    total_states *= discretization
            else:
                # Default: assume 10 possible values if no information
                total_states *= 10
    
    # Cap to avoid overflow or unrealistic numbers
    MAX_STATES = 1e15  # Cap at a trillion trillion states
    return min(total_states, MAX_STATES)


def _create_state_fingerprint(trajectory, policy_idx, timestep_idx, init_state_idx, complete_state_space):
    """
    Create a fingerprint (tuple) for a specific state in the trajectory.
    
    Args:
        trajectory: The trajectory object
        policy_idx: Index of the policy
        timestep_idx: Index of the timestep
        init_state_idx: Index of the initial state
        complete_state_space: The complete state space dictionary
        
    Returns:
        A hashable fingerprint (tuple) representing the state
    """
    # List to store state components
    state_components = []
    
    # Extract values for each state variable
    for var_name in complete_state_space.keys():
        field_info = _map_var_to_trajectory_field(var_name)
        if field_info is None:
            continue
            
        field_name, _ = field_info
        
        if hasattr(trajectory, field_name):
            field_value = getattr(trajectory, field_name)
            
            # Handle different types of fields
            if field_name == "_entities":
                # Extract entities for this specific state
                if len(field_value.shape) >= 4:  # Ensure we have enough dimensions
                    entities = field_value[policy_idx, timestep_idx, init_state_idx]
                    # Get active entities (those with x,y < sentinel value)
                    active_entities = []
                    for entity in entities:
                        if entity[0] < 99 and entity[1] < 99:  # Not sentinel values
                            active_entities.append(tuple(map(int, entity.tolist())))
                    # Sort to ensure consistent order
                    active_entities.sort()
                    state_components.append(("entities", tuple(active_entities)))
                
            elif field_name in ["_brick_map", "_alien_map", "_f_bullet_map", "_e_bullet_map"]:
                # Handle boolean maps
                if len(field_value.shape) >= 3:  # Ensure we have enough dimensions
                    bool_map = field_value[policy_idx, timestep_idx, init_state_idx]
                    # Convert to coordinates of True values
                    true_coords = []
                    for y in range(bool_map.shape[0]):
                        for x in range(bool_map.shape[1]):
                            if bool_map[y, x]:
                                true_coords.append((y, x))
                    state_components.append((field_name, tuple(true_coords)))
                
            elif field_name in ["_f_bullets", "_e_fish", "_e_subs", "_divers"]:
                # Handle bullet/entity arrays
                if len(field_value.shape) >= 4:  # Ensure we have enough dimensions
                    bullets = field_value[policy_idx, timestep_idx, init_state_idx]
                    # Get active bullets/entities
                    active_elements = []
                    for bullet in bullets:
                        if bullet[0] >= 0 and bullet[1] >= 0:  # Not sentinel
                            active_elements.append(tuple(map(int, bullet.tolist())))
                    # Sort for consistency
                    active_elements.sort()
                    state_components.append((field_name, tuple(active_elements)))
                
            else:
                # For simple scalar fields
                if len(field_value.shape) >= 3:  # Ensure we have enough dimensions
                    val = field_value[policy_idx, timestep_idx, init_state_idx]
                    if isinstance(val, jnp.ndarray):
                        # Handle non-scalar arrays
                        if val.size == 1:
                            # If it's size 1, convert to scalar
                            val = val.item()
                        else:
                            # If it's a multi-element array, convert to tuple
                            val = tuple(val.flatten().tolist())
                    state_components.append((field_name, val))
    
    # Sort components by field name for consistency
    state_components.sort(key=lambda x: x[0])
    
    # Create a hashable tuple from the components
    return tuple(state_components) 


"""
Asterix: 
    agent: 100 (10x10 grid)
    gold_and_enemies: 8! * 10 * 2 (8 rows which can be simultaneously occupied, 10 columns, 2 possible orientations)
    SIZE: 100 * 8! * 10 * 2 = 2.4e8
    
Breakout: 
    agent: 10 (pad is only on the bottom)
    ball: 100 (10x10 grid) * 4 (4 possible directions)
    bricks: 30! (10x10 grid)
    SIZE: 10 * 100 * 4 * 30! = 1e34 which is too large  -> 1e10
    
Seaquest: 
    sub_agent: 90 * 2 (2 possible orientations)
    fish: ~ 80 * 2 * 2 (2 possible orientations, 2 possible fish)
    diver: ~ 80 * 2 * 2 (2 possible orientations, 2 possible divers)
    sub_enemy: ~ 80 * 2 * 2 (2 possible orientations, 2 possible sub_enemies)
    friendly_bullet: ~ 80 * 2 * 2 (2 possible orientations, 2 possible friendly_bullets)
    enemy_bullet: ~ 80 * 2 * 2 (2 possible orientations, 2 possible enemy_bullets)
    oxygen: 10 
    diver_gauge: 10 
    SIZE: 90 * 2 * 2 * 80 * 2 * 2 * 80 * 2 * 2 * 80 * 2 * 2 * 10 * 10 = 1e10
    
SpaceInvaders: 
    player: 10 (10x10 grid)
    alien_map: (4 rows, 6 columns) -> 24! = 6.2e21
    alien_direction: 2 (left, right)
    alien_move_timer: 11
    alien_shot_timer: 11
    alien_move_interval: 10
    ramp_index: 9
    SIZE: dominated by alien_map = 1e21 is too large -> 1e10
    
    
"""

SIMPLIFIED_STATE_SPACE_SIZES = { 
    "minatar-asterix": 1e8, 
    "minatar-breakout": 1e10, 
    "minatar-seaquest": 1e10, 
    "minatar-space_invaders": 1e10, 
}