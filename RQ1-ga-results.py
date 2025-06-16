import pandas as pd
import numpy as np

env_names = ["minatar-asterix", "minatar-breakout", "minatar-seaquest", "minatar-space_invaders"]
results_path = "results/ga-effectiveness/"

# Dictionary to store results for each environment
results = {}

for env_name in env_names:
    print(f"\nAnalyzing {env_name}:")
    print("-" * 50)
    
    # Dictionary to store aggregated data across experiments
    aggregated_data = {}
    
    # Load and aggregate data from three experiments
    for experiment in range(1, 4):
        csv_file_path = results_path + env_name + f"/experiment_{experiment}/run_data.csv"
        try:
            data = pd.read_csv(csv_file_path)
            
            # Initialize aggregated_data with the first dataset structure
            if experiment == 1:
                aggregated_data = {col: [] for col in data.columns}
                
            # Collect data from each experiment
            for col in data.columns:
                if experiment == 1:
                    aggregated_data[col] = [data[col].values]
                else:
                    aggregated_data[col].append(data[col].values)
                    
        except FileNotFoundError:
            print(f"Error: The file {csv_file_path} was not found.")
            continue
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            continue
    
    # Calculate statistics for each policy
    policies = ["single-policy", "multi-policy"]
    
    print(f"{'Policy':<8} {'Mean Failure Rate':<20} {'Std Dev':<15} {'Found Failures':<15} {'Not Confirmed':<15} {'% Not Confirmed':<15} {'Sim Steps':<15}")
    print("-" * 90)
    
    for policy in policies:
        # Calculate mean failure rate statistics
        failure_rate_col = f'{policy}_mean_failure_rate'
        if failure_rate_col in aggregated_data:
            failure_rates = np.array(aggregated_data[failure_rate_col])
            mean_failure_rates = np.mean(failure_rates, axis=0)
            std_failure_rates = np.std(failure_rates, axis=0)/np.sqrt(failure_rates.shape[0])
            #mean_failure_rate = np.mean(mean_failure_rates)
            #std_failure_rate = np.mean(std_failure_rates)
        else:
            mean_failure_rate = np.nan
            std_failure_rate = np.nan
            
        # Calculate found failures statistics
        found_failures_col = f'{policy}_found_failures'
        if found_failures_col in aggregated_data:
            found_failures = np.array(aggregated_data[found_failures_col])
            total_found = np.sum(found_failures, axis=1)
            mean_found = np.mean(total_found, axis=0)
            std_found = np.std(total_found, axis=0)/np.sqrt(total_found.shape[0])
        else:
            total_found = np.nan
            mean_found = np.nan
            std_found = np.nan
            
        # Calculate not confirmed solvable statistics
        not_confirmed_col = f'{policy}_not_confirmed_solvable'
        if not_confirmed_col in aggregated_data:
            not_confirmed = np.array(aggregated_data[not_confirmed_col])
            total_not_confirmed = np.sum(not_confirmed, axis=1)
            mean_not_confirmed = np.mean(total_not_confirmed, axis=0)
            std_not_confirmed = np.std(total_not_confirmed, axis=0)/np.sqrt(total_not_confirmed.shape[0])
            percent_not_confirmed = (np.sum(total_not_confirmed) / np.sum(total_found) * 100) if np.sum(total_found) > 0 else 0
        else:
            total_not_confirmed = np.nan
            mean_not_confirmed = np.nan
            std_not_confirmed = np.nan
            percent_not_confirmed = np.nan
            
        num_sim_steps_col = f'{policy}_num_sim_steps'
        if num_sim_steps_col in aggregated_data:
            num_sim_steps = np.array(aggregated_data[num_sim_steps_col])
            total_sim_steps = np.sum(num_sim_steps, axis=1)
            mean_sim_steps = np.mean(total_sim_steps, axis=0)
            std_sim_steps = np.std(total_sim_steps, axis=0)/np.sqrt(total_sim_steps.shape[0])
        else:
            total_sim_steps = np.nan
            mean_sim_steps = np.nan
            std_sim_steps = np.nan
        # Print results
        print(f"{policy:<8} {mean_failure_rates[-1]:>8.4f} ± {std_failure_rates[-1]:<8.4f} {mean_found:>8.0f} ± {std_found:<8.0f} {mean_not_confirmed:>8.0f} ± {std_not_confirmed:<8.0f} {100-percent_not_confirmed:>8.2f} ± {std_not_confirmed/mean_not_confirmed*100:<8.2f}% {mean_sim_steps:>8.0f} ± {std_sim_steps:<8.0f}")
    
    print("\n") 