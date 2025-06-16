import pandas as pd
import numpy as np
import os
from pathlib import Path

def calculate_mean_sem(values):
    """Calculate mean and standard error of the mean (SEM)"""
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(len(values))
    return mean, sem

def aggregate_results():
    """Aggregate results from all environments and experiments"""
    
    # Base path for mdpfuzz_candidate_generation results
    base_path = Path("results/mdpfuzz-effectiveness/")
    
    # Find all environments
    environments = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("minatar-")]
    
    # Dictionary to store all data
    all_data = {}
    
    print("=== MDP-Fuzz Candidate Generation Results ===\n")
    
    for env in environments:
        env_name = env.name
        print(f"Processing {env_name}...")
        
        # Find all experiment directories
        experiment_dirs = [d for d in env.iterdir() if d.is_dir() and d.name.startswith("experiment_")]
        
        # Collect data from all experiments for this environment
        env_data = []
        
        for exp_dir in experiment_dirs:
            data_file = exp_dir / "data.csv"
            if data_file.exists():
                df = pd.read_csv(data_file)
                if not df.empty:
                    env_data.append(df.iloc[0])  # Take the first (and likely only) row
        
        if env_data:
            # Convert to DataFrame for easier processing
            env_df = pd.DataFrame(env_data)
            all_data[env_name] = env_df
            print(f"  Found {len(env_data)} experiments")
        else:
            print(f"  No valid data found")
    
    # Calculate and display aggregated results
    print("\n=== AGGREGATED RESULTS (Mean ± SEM) ===\n")
    
    # Get all metric columns (excluding the index column)
    if all_data:
        sample_df = next(iter(all_data.values()))
        metrics = [col for col in sample_df.columns if col not in ['Unnamed: 0']]
        
        # Create summary for each metric
        summary_data = {}
        
        for metric in metrics:
            print(f"{metric}:")
            metric_values = {}
            
            for env_name, env_df in all_data.items():
                if metric in env_df.columns:
                    values = env_df[metric].dropna().values
                    if len(values) > 0:
                        mean, sem = calculate_mean_sem(values)
                        metric_values[env_name] = (mean, sem, len(values))
                        print(f"  {env_name}: {mean:.4f} ± {sem:.4f} (n={len(values)})")
            
            # Calculate overall aggregate across all environments
            if metric_values:
                all_values = []
                for env_name, env_df in all_data.items():
                    if metric in env_df.columns:
                        all_values.extend(env_df[metric].dropna().values)
                
                if all_values:
                    overall_mean, overall_sem = calculate_mean_sem(all_values)
                    print(f"  OVERALL: {overall_mean:.4f} ± {overall_sem:.4f} (n={len(all_values)})")
                    summary_data[metric] = {
                        'overall_mean': overall_mean,
                        'overall_sem': overall_sem,
                        'total_n': len(all_values),
                        'by_environment': metric_values
                    }
            
            print()
        
        # Create a summary table
        print("=== SUMMARY TABLE ===")
        print("Metric | Overall Mean ± SEM | Total N")
        print("-" * 50)
        
        for metric, data in summary_data.items():
            print(f"{metric:<30} | {data['overall_mean']:8.4f} ± {data['overall_sem']:6.4f} | {data['total_n']:3d}")
    
    else:
        print("No data found in any environment directories.")

if __name__ == "__main__":
    aggregate_results()
