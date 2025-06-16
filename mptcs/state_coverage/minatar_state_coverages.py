import jax.numpy as jnp 
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import jax


def get_state_coverage(trajectory, environment_state_space) -> Dict[str, float]:
    """
    Calculate the coverage of state space visited in the trajectory.
    
    Args:
        trajectory: A Pgx.State object containing the full trajectory with shape 
                    (executed_policies, time_step, num_initial_states, object.shape)
        environment_state_space: An environment-specific state coverage object
                                 that provides possible values for each state variable
    
    Returns:
        A dictionary mapping state variable names to their coverage percentages
    """
    complete_state_space = environment_state_space.get_complete_state_space()
    coverage_metrics = {}
    
    # Extract trajectory attributes based on environment
    for var_name, possible_values in complete_state_space.items():
        # For each state variable, determine which field in the trajectory corresponds to it
        field_info = _map_var_to_trajectory_field(var_name)
        
        if field_info is None:
            continue  # Skip if we can't map this variable to the trajectory
            
        field_name, extract_method = field_info
        
        try:
            # Extract values using the specialized extraction method
            unique_observed = extract_method(trajectory, field_name, var_name)
            
            if unique_observed is not None:
                # Calculate coverage
                num_unique = len(unique_observed)
                num_possible = len(possible_values)
                coverage = (num_unique / num_possible) * 100.0
                coverage_metrics[var_name] = float(coverage)
        except Exception as e:
            # Log the error but continue processing other variables
            print(f"Error processing variable {var_name}: {e}")
    
    # Calculate overall coverage (average of individual coverages)
    if coverage_metrics:
        coverage_metrics["overall"] = sum(coverage_metrics.values()) / len(coverage_metrics)
    
    return coverage_metrics


def _extract_direct_field(trajectory, field_name, var_name):
    """Extract values from a simple field in the trajectory."""
    if hasattr(trajectory, field_name):
        values = getattr(trajectory, field_name)
        # Flatten all dimensions
        flat_values = jnp.reshape(values, (-1,))
        # Get unique values
        return jnp.unique(flat_values)
    return None


def _extract_entity_field(trajectory, field_name, var_name):
    """Extract entity-specific values from the _entities field."""
    if hasattr(trajectory, field_name):
        entities = getattr(trajectory, field_name)
        # Entity structure is [x, y, direction, is_gold]
        if var_name == "entity_x":
            idx = 0
        elif var_name == "entity_y":
            idx = 1
        elif var_name == "entity_direction":
            idx = 2
        elif var_name == "entity_type":
            idx = 3
        else:
            return None
            
        # Extract the specific component from entities
        # entities shape is (policies, timesteps, init_states, num_entities, 4)
        # Reshape to get all values of the specific component
        values = entities[..., idx]
        flat_values = jnp.reshape(values, (-1,))
        # Filter out sentinel values (e.g., 99)
        valid_values = flat_values[flat_values < 99]  # Assuming 99 is the sentinel value
        return jnp.unique(valid_values)
    return None


def _extract_map_coordinates(trajectory, field_name, var_name):
    """Extract x,y coordinates from boolean maps like brick_map, alien_map, etc."""
    if hasattr(trajectory, field_name):
        bool_map = getattr(trajectory, field_name)
        # Map structure is typically a 2D or 3D boolean array
        
        # Determine which dimension to extract
        if var_name.endswith("_x"):
            axis = 1  # X is typically the second dimension
        elif var_name.endswith("_y"):
            axis = 0  # Y is typically the first dimension
        else:
            return None
            
        # For each map in the trajectory, find coordinates where True
        # This is more complex and would require iterating through the maps
        # For simplicity, we'll just return all possible coordinates
        if axis == 1:  # X coordinates
            return jnp.arange(bool_map.shape[-1])  # Assuming last dim is X
        else:  # Y coordinates
            return jnp.arange(bool_map.shape[-2])  # Assuming second-to-last dim is Y
    return None


def _extract_bullets_position(trajectory, field_name, var_name):
    """Extract bullet positions from bullet arrays."""
    if hasattr(trajectory, field_name):
        bullets = getattr(trajectory, field_name)
        # Bullet structure is typically [x, y, direction]
        
        if var_name.endswith("_x"):
            idx = 0  # X is first component
        elif var_name.endswith("_y"):
            idx = 1  # Y is second component
        elif var_name.endswith("_dir"):
            idx = 2  # Direction is third component
        else:
            return None
            
        # Extract the specific component from bullets
        values = bullets[..., idx]
        flat_values = jnp.reshape(values, (-1,))
        # Filter out sentinel values (e.g., -1)
        valid_values = flat_values[flat_values >= 0]  # Assuming -1 is the sentinel value
        return jnp.unique(valid_values)
    return None


def _map_var_to_trajectory_field(var_name: str) -> Optional[Tuple[str, callable]]:
    """
    Maps a state space variable name to the corresponding field in the Pgx.State trajectory
    and provides an appropriate extraction method.
    
    Args:
        var_name: The variable name from the state space
    
    Returns:
        A tuple of (field_name, extraction_method) or None if no mapping exists
    """
    # Map variable names to (field_name, extraction_function)
    # This specialized mapping handles the different data structures in the trajectory
    mapping = {
        # Asterix mappings
        "player_x": ("_player_x", _extract_direct_field),
        "player_y": ("_player_y", _extract_direct_field),
        "entity_x": ("_entities", _extract_entity_field),
        "entity_y": ("_entities", _extract_entity_field),
        "entity_direction": ("_entities", _extract_entity_field),
        "entity_type": ("_entities", _extract_entity_field),
        "shot_timer": ("_shot_timer", _extract_direct_field),
        "spawn_speed": ("_spawn_speed", _extract_direct_field),
        "spawn_timer": ("_spawn_timer", _extract_direct_field),
        "move_speed": ("_move_speed", _extract_direct_field),
        "move_timer": ("_move_timer", _extract_direct_field),
        "ramp_timer": ("_ramp_timer", _extract_direct_field),
        "ramp_index": ("_ramp_index", _extract_direct_field),
        
        # Breakout mappings
        "ball_x": ("_ball_x", _extract_direct_field),
        "ball_y": ("_ball_y", _extract_direct_field),
        "ball_dir": ("_ball_dir", _extract_direct_field),
        "pos": ("_pos", _extract_direct_field),
        "brick_x": ("_brick_map", _extract_map_coordinates),
        "brick_y": ("_brick_map", _extract_map_coordinates),
        "strike": ("_strike", _extract_direct_field),
        "last_x": ("_last_x", _extract_direct_field),
        "last_y": ("_last_y", _extract_direct_field),
        
        # Seaquest mappings
        "oxygen": ("_oxygen", _extract_direct_field),
        "diver_count": ("_diver_count", _extract_direct_field),
        "sub_x": ("_sub_x", _extract_direct_field),
        "sub_y": ("_sub_y", _extract_direct_field),
        "sub_or": ("_sub_or", _extract_direct_field),
        "bullet_x": ("_f_bullets", _extract_bullets_position),
        "bullet_y": ("_f_bullets", _extract_bullets_position),
        "bullet_dir": ("_f_bullets", _extract_bullets_position),
        "fish_x": ("_e_fish", _extract_bullets_position),
        "fish_y": ("_e_fish", _extract_bullets_position),
        "fish_dir": ("_e_fish", _extract_bullets_position),
        "fish_color": ("_e_fish", _extract_bullets_position),
        "esub_x": ("_e_subs", _extract_bullets_position),
        "esub_y": ("_e_subs", _extract_bullets_position),
        "esub_dir": ("_e_subs", _extract_bullets_position),
        "esub_type": ("_e_subs", _extract_bullets_position),
        "esub_firing": ("_e_subs", _extract_bullets_position),
        "diver_x": ("_divers", _extract_bullets_position),
        "diver_y": ("_divers", _extract_bullets_position),
        "diver_dir": ("_divers", _extract_bullets_position),
        "diver_state": ("_divers", _extract_bullets_position),
        "e_spawn_speed": ("_e_spawn_speed", _extract_direct_field),
        "e_spawn_timer": ("_e_spawn_timer", _extract_direct_field),
        "d_spawn_timer": ("_d_spawn_timer", _extract_direct_field),
        "move_speed": ("_move_speed", _extract_direct_field),
        "shot_timer": ("_shot_timer", _extract_direct_field),
        "surface": ("_surface", _extract_direct_field),
        
        # Space Invaders mappings
        # "pos": ("_pos", _extract_direct_field),  # Already mapped
        "f_bullet_x": ("_f_bullet_map", _extract_map_coordinates),
        "f_bullet_y": ("_f_bullet_map", _extract_map_coordinates),
        "e_bullet_x": ("_e_bullet_map", _extract_map_coordinates),
        "e_bullet_y": ("_e_bullet_map", _extract_map_coordinates),
        "alien_x": ("_alien_map", _extract_map_coordinates),
        "alien_y": ("_alien_map", _extract_map_coordinates),
        "alien_dir": ("_alien_dir", _extract_direct_field),
        "enemy_move_interval": ("_enemy_move_interval", _extract_direct_field),
        "alien_move_timer": ("_alien_move_timer", _extract_direct_field),
        "alien_shot_timer": ("_alien_shot_timer", _extract_direct_field),
        # "ramp_index": ("_ramp_index", _extract_direct_field),  # Already mapped
        # "shot_timer": ("_shot_timer", _extract_direct_field),  # Already mapped
    }
    
    return mapping.get(var_name)


class AsterixStateCoverage: 
    def __init__(self): 
        # Grid size is 10x10
        self.possible_x = jnp.arange(10)  # Player x position (0-9)
        self.possible_y = jnp.arange(10)  # Player y position (0-9)
        
        # Entities can be at positions 0-9, with left/right direction and gold/regular type
        self.entity_x_positions = jnp.array([0, 9])  # Entities spawn at x=0 or x=9
        self.entity_y_positions = jnp.arange(1, 10)  # Entity y positions (1-9)
        self.entity_directions = jnp.array([0, 1])  # 0: left-to-right, 1: right-to-left
        self.entity_types = jnp.array([0, 1])  # 0: regular, 1: gold
        
        # Timers and speeds
        self.shot_timer_range = jnp.arange(5)  # 0-4
        self.spawn_speed_range = jnp.arange(8, 1, -1)  # 8 down to 2
        self.spawn_timer_range = jnp.arange(9)  # 0-8
        self.move_speed_range = jnp.arange(4, 0, -1)  # 4 down to 1
        self.move_timer_range = jnp.arange(5)  # 0-4
        self.ramp_timer_range = jnp.arange(1000)  # 0-999
        self.ramp_index_range = jnp.arange(9)  # 0-8
    
    def get_complete_state_space(self):
        # Returns the complete state space as ranges for each variable
        return {
            "player_x": self.possible_x,
            "player_y": self.possible_y,
            "entity_x": self.entity_x_positions,
            "entity_y": self.entity_y_positions,
            "entity_direction": self.entity_directions,
            "entity_type": self.entity_types,
            "shot_timer": self.shot_timer_range,
            "spawn_speed": self.spawn_speed_range,
            "spawn_timer": self.spawn_timer_range,
            "move_speed": self.move_speed_range,
            "move_timer": self.move_timer_range,
            "ramp_timer": self.ramp_timer_range,
            "ramp_index": self.ramp_index_range
        }


class BreakoutStateCoverage:
    def __init__(self):
        # Grid size is 10x10
        self.ball_x_range = jnp.arange(10)  # Ball x position (0-9)
        self.ball_y_range = jnp.arange(10)  # Ball y position (0-9)
        
        # Ball directions: 0-3 corresponding to (up-right, up-left, down-right, down-left)
        self.ball_dir_range = jnp.arange(4)
        
        # Paddle position (pos)
        self.pos_range = jnp.arange(10)  # Paddle position (0-9)
        
        # Brick map is a 10x10 boolean array (represented here as ranges for each dimension)
        self.brick_x_range = jnp.arange(10)  # Brick x positions (0-9)
        self.brick_y_range = jnp.arange(10)  # Brick y positions (0-9)
        
        # Strike is a boolean indicating if ball hit paddle
        self.strike_range = jnp.array([0, 1])  # 0: False, 1: True
        
        # Last ball position
        self.last_x_range = jnp.arange(10)  # Last ball x position (0-9)
        self.last_y_range = jnp.arange(10)  # Last ball y position (0-9)
    
    def get_complete_state_space(self):
        # Returns the complete state space as ranges for each variable
        return {
            "ball_x": self.ball_x_range,
            "ball_y": self.ball_y_range,
            "ball_dir": self.ball_dir_range,
            "pos": self.pos_range,
            "brick_x": self.brick_x_range,
            "brick_y": self.brick_y_range,
            "strike": self.strike_range,
            "last_x": self.last_x_range,
            "last_y": self.last_y_range
        }


class SeaquestStateCoverage:
    def __init__(self):
        # Grid size is 10x10
        self.oxygen_range = jnp.arange(101)  # Oxygen level (0-100)
        self.diver_count_range = jnp.arange(6)  # Number of rescued divers (0-5)
        
        # Submarine position and orientation
        self.sub_x_range = jnp.arange(10)  # Submarine x position (0-9)
        self.sub_y_range = jnp.arange(10)  # Submarine y position (0-9)
        self.sub_or_range = jnp.array([0, 1])  # 0: facing left, 1: facing right
        
        # Bullets, fish, enemy subs, and divers have x, y positions and other properties
        # F_bullets have x, y, and direction (represented here as ranges)
        self.bullet_x_range = jnp.arange(10)  # Bullet x positions (0-9)
        self.bullet_y_range = jnp.arange(10)  # Bullet y positions (0-9)
        self.bullet_dir_range = jnp.array([0, 1])  # 0: left, 1: right
        
        # Enemy fish have x, y, direction, and color
        self.fish_x_range = jnp.arange(10)  # Fish x positions (0-9)
        self.fish_y_range = jnp.arange(10)  # Fish y positions (0-9)
        self.fish_dir_range = jnp.array([0, 1])  # 0: left, 1: right
        self.fish_color_range = jnp.arange(3)  # 0,1,2 for different fish colors
        
        # Enemy subs have x, y, direction, type, and firing status
        self.esub_x_range = jnp.arange(10)  # Enemy sub x positions (0-9)
        self.esub_y_range = jnp.arange(10)  # Enemy sub y positions (0-9)
        self.esub_dir_range = jnp.array([0, 1])  # 0: left, 1: right
        self.esub_type_range = jnp.arange(3)  # 0,1,2 for different sub types
        self.esub_firing_range = jnp.array([0, 1])  # 0: not firing, 1: firing
        
        # Divers have x, y, direction, and state
        self.diver_x_range = jnp.arange(10)  # Diver x positions (0-9)
        self.diver_y_range = jnp.arange(10)  # Diver y positions (0-9)
        self.diver_dir_range = jnp.array([0, 1])  # 0: left, 1: right
        self.diver_state_range = jnp.array([0, 1])  # 0: swimming, 1: rescued
        
        # Timers and speeds
        self.e_spawn_speed_range = jnp.arange(50, 5, -5)  # 50 down to 10, step -5
        self.e_spawn_timer_range = jnp.arange(51)  # 0-50
        self.d_spawn_timer_range = jnp.arange(101)  # 0-100
        self.move_speed_range = jnp.arange(5, 0, -1)  # 5 down to 1
        self.ramp_index_range = jnp.arange(10)  # 0-9
        self.shot_timer_range = jnp.arange(5)  # 0-4
        
        # Surface indicator
        self.surface_range = jnp.array([0, 1])  # 0: underwater, 1: at surface
    
    def get_complete_state_space(self):
        # Returns the complete state space as ranges for each variable
        return {
            "oxygen": self.oxygen_range,
            "diver_count": self.diver_count_range,
            "sub_x": self.sub_x_range,
            "sub_y": self.sub_y_range,
            "sub_or": self.sub_or_range,
            "bullet_x": self.bullet_x_range,
            "bullet_y": self.bullet_y_range,
            "bullet_dir": self.bullet_dir_range,
            "fish_x": self.fish_x_range,
            "fish_y": self.fish_y_range,
            "fish_dir": self.fish_dir_range,
            "fish_color": self.fish_color_range,
            "esub_x": self.esub_x_range,
            "esub_y": self.esub_y_range,
            "esub_dir": self.esub_dir_range,
            "esub_type": self.esub_type_range,
            "esub_firing": self.esub_firing_range,
            "diver_x": self.diver_x_range,
            "diver_y": self.diver_y_range,
            "diver_dir": self.diver_dir_range,
            "diver_state": self.diver_state_range,
            "e_spawn_speed": self.e_spawn_speed_range,
            "e_spawn_timer": self.e_spawn_timer_range,
            "d_spawn_timer": self.d_spawn_timer_range,
            "move_speed": self.move_speed_range,
            "ramp_index": self.ramp_index_range,
            "shot_timer": self.shot_timer_range,
            "surface": self.surface_range
        }


class SpaceInvadersStateCoverage:
    def __init__(self):
        # Grid size is 10x10
        self.pos_range = jnp.arange(10)  # Player position (0-9)
        
        # Bullet maps and alien map are 10x10 boolean arrays
        # (represented here as ranges for each dimension)
        self.f_bullet_x_range = jnp.arange(10)  # Friendly bullet x positions (0-9)
        self.f_bullet_y_range = jnp.arange(10)  # Friendly bullet y positions (0-9)
        
        self.e_bullet_x_range = jnp.arange(10)  # Enemy bullet x positions (0-9)
        self.e_bullet_y_range = jnp.arange(10)  # Enemy bullet y positions (0-9)
        
        self.alien_x_range = jnp.arange(10)  # Alien x positions (0-9)
        self.alien_y_range = jnp.arange(7)  # Alien y positions (0-6, they don't go to bottom)
        
        # Alien direction (1: right, -1: left)
        self.alien_dir_range = jnp.array([-1, 1])
        
        # Timers and intervals
        self.enemy_move_interval_range = jnp.arange(10, 0, -1)  # 10 down to 1
        self.alien_move_timer_range = jnp.arange(11)  # 0-10
        self.alien_shot_timer_range = jnp.arange(11)  # 0-10
        self.ramp_index_range = jnp.arange(9)  # 0-8
        self.shot_timer_range = jnp.arange(5)  # 0-4
    
    def get_complete_state_space(self):
        # Returns the complete state space as ranges for each variable
        return {
            "pos": self.pos_range,
            "f_bullet_x": self.f_bullet_x_range,
            "f_bullet_y": self.f_bullet_y_range,
            "e_bullet_x": self.e_bullet_x_range,
            "e_bullet_y": self.e_bullet_y_range,
            "alien_x": self.alien_x_range,
            "alien_y": self.alien_y_range,
            "alien_dir": self.alien_dir_range,
            "enemy_move_interval": self.enemy_move_interval_range,
            "alien_move_timer": self.alien_move_timer_range,
            "alien_shot_timer": self.alien_shot_timer_range,
            "ramp_index": self.ramp_index_range,
            "shot_timer": self.shot_timer_range
        }