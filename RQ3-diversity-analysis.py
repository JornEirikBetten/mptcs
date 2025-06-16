import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.lines import Line2D
import jax 
import jax.numpy as jnp

# Set the style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

path_to_data = os.getcwd() + "/results/diversity/results.csv"

df = pd.read_csv(path_to_data)

asterix_data = df[df["env_name"] == "minatar-asterix"]
breakout_data = df[df["env_name"] == "minatar-breakout"]
seaquest_data = df[df["env_name"] == "minatar-seaquest"]
space_invaders_data = df[df["env_name"] == "minatar-space_invaders"]
policy_numbers = [i+1 for i in range(20)]
policy_labels = [r"$\pi_{" + str(p) + "}$" for p in policy_numbers]

# Create a more balanced figure size and better spacing
fig, axs = plt.subplots(4, 2, figsize=(20, 14))
plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.95)  # Adjust top margin for legend

# Define colors for consistent use
topk_color = sns.color_palette("husl", 8)[0]
mptcs_color = sns.color_palette("husl", 8)[4]

# Create custom legend handles
legend_elements = [
    Line2D([0], [0], color=topk_color, lw=0, marker='s', markersize=15, 
           markerfacecolor=topk_color, alpha=0.8, markeredgecolor='black', markeredgewidth=0.5, label='Top-K'),
    Line2D([0], [0], color=mptcs_color, lw=0, marker='s', markersize=15, 
           markerfacecolor=mptcs_color, alpha=0.8, markeredgecolor='black', markeredgewidth=0.5, label='MPTCS')
]

# Add a single legend at the top of the figure
fig.legend(handles=legend_elements, loc='upper center', 
           fontsize=14, frameon=True, ncol=2, bbox_to_anchor=(0.5, 0.98))
titles = ["Asterix", "Breakout", "Seaquest", "Space Invaders"]
for idx, env_name in enumerate(["asterix", "breakout", "seaquest", "space_invaders"]):
    data = [asterix_data, breakout_data, seaquest_data, space_invaders_data][idx]
    print(env_name)

    topk_mean_failure_rate = np.array(data["top_k_mean_failure_rate"].values)
    topk_unique_observations = np.array(data["top_k_unique_observations"].values)
    mptcs_mean_failure_rate = np.array(data["mptcs_mean_failure_rate"].values)
    mptcs_unique_observations = np.array(data["mptcs_unique_observations"].values)
    mean_topk_failure_rate = np.mean(topk_mean_failure_rate)
    std_topk_failure_rate = np.std(topk_mean_failure_rate)/np.sqrt(len(topk_mean_failure_rate))
    mean_topk_unique_observations = np.mean(topk_unique_observations)
    std_topk_unique_observations = np.std(topk_unique_observations)/np.sqrt(len(topk_unique_observations))
    mean_mptcs_mean_failure_rate = np.mean(mptcs_mean_failure_rate)
    std_mptcs_mean_failure_rate = np.std(mptcs_mean_failure_rate)/np.sqrt(len(mptcs_mean_failure_rate))
    mean_mptcs_unique_observations = np.mean(mptcs_unique_observations)
    std_mptcs_unique_observations = np.std(mptcs_unique_observations)/np.sqrt(len(mptcs_unique_observations))
    
    # Number of test cases in suites
    num_test_cases_topk = np.array(data["top_k_num_test_cases"].values)
    num_test_cases_mptcs = np.array(data["mptcs_num_test_cases"].values)
    mean_num_test_cases_topk = np.mean(num_test_cases_topk)
    std_num_test_cases_topk = np.std(num_test_cases_topk)/np.sqrt(len(num_test_cases_topk))
    mean_num_test_cases_mptcs = np.mean(num_test_cases_mptcs)
    std_num_test_cases_mptcs = np.std(num_test_cases_mptcs)/np.sqrt(len(num_test_cases_mptcs))
    
    print(f"env_name: {env_name}")
    print(f"mean_topk_failure_rate: {mean_topk_failure_rate} ± {std_topk_failure_rate}")
    print(f"mean_topk_unique_observations: {mean_topk_unique_observations/mean_num_test_cases_topk} ± {std_topk_unique_observations/mean_num_test_cases_topk}")
    print(f"mean_mptcs_mean_failure_rate: {mean_mptcs_mean_failure_rate} ± {std_mptcs_mean_failure_rate}")
    print(f"mean_mptcs_unique_observations: {mean_mptcs_unique_observations/mean_num_test_cases_mptcs} ± {std_mptcs_unique_observations/mean_num_test_cases_mptcs}")
    avg_failures_topk = []
    failures_std_topk = []
    avg_failures_mptcs = []
    failures_std_mptcs = []
    all_failures_topk = np.zeros((len(policy_numbers), 10))
    all_failures_mptcs = np.zeros((len(policy_numbers), 10))
    for policy_number in policy_numbers:
        topk_failures = np.array(data[f"top_k_failures_{policy_number}_policies"].values)/mean_num_test_cases_topk
        avg_failures_topk.append(np.mean(topk_failures))
        failures_std_topk.append(np.std(topk_failures)/np.sqrt(len(topk_failures)))
        mptcs_failures = np.array(data[f"mptcs_failures_{policy_number}_policies"].values)/mean_num_test_cases_mptcs
        avg_failures_mptcs.append(np.mean(mptcs_failures))
        failures_std_mptcs.append(np.std(mptcs_failures)/np.sqrt(len(mptcs_failures)))
        all_failures_topk[policy_number-1, :] = topk_failures
        all_failures_mptcs[policy_number-1, :] = mptcs_failures
    
    def calculate_entropy(distribution): 
        normalized_distribution = distribution/jnp.sum(distribution)
        entropy = -jnp.sum(jnp.where(normalized_distribution > 0, normalized_distribution*np.log(normalized_distribution), 0))
        return entropy
    entropies_topk = np.zeros(10)
    for i in range(10): 
        passes = 1 - all_failures_topk[:, i]
        entropies_topk[i] = calculate_entropy(passes)
    avg_entropy_topk = np.mean(entropies_topk)
    std_entropy_topk = np.std(entropies_topk)/np.sqrt(len(entropies_topk))
    print(f"avg_entropy_topk: {avg_entropy_topk} ± {std_entropy_topk}")
    
    # Improved plotting for Top-K
    ax = axs[idx, 0]
    avg_passes_topk_high = 1 - np.array(avg_failures_topk) + np.array(failures_std_topk)
    avg_passes_topk_low = 1 - np.array(avg_failures_topk) - np.array(failures_std_topk) 
    avg_passes_topk = (avg_passes_topk_high + avg_passes_topk_low)/2
    normalized_avg_passes_topk = avg_passes_topk/np.sum(avg_passes_topk)
    normalized_avg_passes_topk_high = avg_passes_topk_high/np.sum(avg_passes_topk_high)
    normalized_avg_passes_topk_low = avg_passes_topk_low/np.sum(avg_passes_topk_low)
    # entropy_topk = -np.sum(np.where(normalized_avg_passes_topk > 0, normalized_avg_passes_topk*np.log(normalized_avg_passes_topk), 0))
    # entropy_topk_high = -np.sum(np.where(normalized_avg_passes_topk_high > 0, normalized_avg_passes_topk_high*np.log(normalized_avg_passes_topk_high), 0))
    # entropy_topk_low = -np.sum(np.where(normalized_avg_passes_topk_low > 0, normalized_avg_passes_topk_low*np.log(normalized_avg_passes_topk_low), 0))
    # print(f"entropy_topk: {entropy_topk} ± {abs(entropy_topk_high - entropy_topk_low)}")
    # #print(f"entropy_topk_high: {entropy_topk_high}")
    # #print(f"entropy_topk_low: {entropy_topk_low}")
    bars = ax.bar(policy_numbers, avg_passes_topk, yerr=failures_std_topk, 
             color=topk_color, alpha=0.8, 
             width=0.7, edgecolor='black', linewidth=0.5)
    
    # Add a subtle 3D effect to bars
    for bar in bars:
        bar.set_edgecolor('black')
        bar.set_linewidth(0.5)
    
    # Improved title and styling
    ax.set_title(f"{titles[idx]}", fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(policy_numbers)
    ax.set_xticklabels(policy_labels, rotation=45, fontsize=10)
    
    # Only add xlabel to bottom plots
    if idx == 3:  # Last row
        ax.set_xlabel("policies in evaluation ensemble", fontsize=12)
    else:
        ax.set_xlabel("")
        
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10)
    ax.set_ylabel("percentage of test cases passed", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set consistent y-axis limits for better comparison
    ax.set_ylim(0, 1.05)
    
    # Improved plotting for MPTCS
    ax = axs[idx, 1]
    avg_passes_mptcs = 1 - np.array(avg_failures_mptcs)
    normalized_avg_passes_mptcs = avg_passes_mptcs/np.sum(avg_passes_mptcs)
    normalized_std_passes_mptcs = failures_std_mptcs/np.sum(avg_passes_mptcs)
    avg_passes_mptcs_high = 1 - np.array(avg_failures_mptcs) + np.array(failures_std_mptcs)
    avg_passes_mptcs_low = 1 - np.array(avg_failures_mptcs) - np.array(failures_std_mptcs) 
    normalized_avg_passes_mptcs_high = avg_passes_mptcs_high/np.sum(avg_passes_mptcs_high)
    normalized_avg_passes_mptcs_low = avg_passes_mptcs_low/np.sum(avg_passes_mptcs_low)
    entropies_mptcs = np.zeros(10)
    for i in range(10): 
        passes = 1 - all_failures_mptcs[:, i]
        entropies_mptcs[i] = calculate_entropy(passes)
    avg_entropy_mptcs = np.mean(entropies_mptcs)
    std_entropy_mptcs = np.std(entropies_mptcs)/np.sqrt(len(entropies_mptcs))
    print(f"avg_entropy_mptcs: {avg_entropy_mptcs} ± {std_entropy_mptcs}")
    # entropy_mptcs = -np.sum(np.where(normalized_avg_passes_mptcs > 0, normalized_avg_passes_mptcs*np.log(normalized_avg_passes_mptcs), 0))
    # entropy_mptcs_high = -np.sum(np.where(normalized_avg_passes_mptcs_high > 0, normalized_avg_passes_mptcs_high*np.log(normalized_avg_passes_mptcs_high), 0))
    # entropy_mptcs_low = -np.sum(np.where(normalized_avg_passes_mptcs_low > 0, normalized_avg_passes_mptcs_low*np.log(normalized_avg_passes_mptcs_low), 0))
    # print(f"entropy_mptcs: {entropy_mptcs} ± {abs(entropy_mptcs_high - entropy_mptcs_low)}")
    # print(f"entropy_mptcs_high: {entropy_mptcs_high}")
    # print(f"entropy_mptcs_low: {entropy_mptcs_low}")
    
    bars = ax.bar(policy_numbers, avg_passes_mptcs, yerr=failures_std_mptcs, 
             color=mptcs_color, alpha=0.8, 
             width=0.7, edgecolor='black', linewidth=0.5)
    
    # Add a subtle 3D effect to bars
    for bar in bars:
        bar.set_edgecolor('black')
        bar.set_linewidth(0.5)
    
    ax.set_title(f"{titles[idx]}", fontsize=14, fontweight='bold', pad=10)
    ax.set_xticks(policy_numbers)
    ax.set_xticklabels(policy_labels, rotation=45, fontsize=10)
    
    # Only add xlabel to bottom plots
    if idx == 3:  # Last row
        ax.set_xlabel("Policies", fontsize=12)
    else:
        ax.set_xlabel("")
        
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=10)
    ax.set_ylabel("percentage of test cases passed", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set consistent y-axis limits
    ax.set_ylim(0, 1.05)

# Add descriptive text at the bottom
#fig.text(0.5, 0.01, 
#         "Top-K: Test cases selected by highest difficulty scores \nMPTCS: Multi-Policy Test Case Selection", 
#         ha='center', fontsize=12)

# Save with higher DPI for better quality
plt.savefig(os.getcwd() + "/results/diversity/diversity_analysis.pdf", 
            format="pdf", bbox_inches="tight", dpi=300)


