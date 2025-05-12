import jax.numpy as jnp 
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import colors 
import numpy as np 
import jax 

def make_ticks_from_limits(limits, num_ticks, num_decimals=4): 
    """
    Make ticks from limits and number of ticks. 
    """
    numerical_ticks = jnp.linspace(limits[0], limits[1], num_ticks)
    #numerical_ticks = jnp.arange(limits[0], limits[1], (limits[1] - limits[0])/num_ticks)
    ticks = [f"{tick:.{num_decimals}f}" for tick in numerical_ticks]
    return ticks



def plot_archive(archive, iteration, save_path, cbar_label, xlabel, ylabel, xtick_labels, ytick_labels, ticks, shape=(50, 50)): 
    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure()
    ax = plt.gca() 
    fitnesses = jnp.where(archive.fitnesses > 0, archive.fitnesses, 0).reshape(shape)
    im = ax.imshow(fitnesses, cmap="bone", origin="lower")
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")
    # Make spines thick and black
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color('black')
    plt.xticks(ticks, xtick_labels)
    plt.yticks(ticks, ytick_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return 0


def plot_test_case_trajectories(trajectories, save_path, action_mapping): 
    num_channels = trajectories.state.observation.shape[-1]
    num_runs = trajectories.state.observation.shape[0] 
    lengths = trajectories.state.observation.shape[1] 
    cmap = sns.color_palette("cubehelix", num_channels)
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(num_channels + 2)]
    norm = colors.BoundaryNorm(bounds, num_channels + 1)
    def visualize_state(observation, ax, timestep, action, terminated, policy_number): 
        if terminated: 
            numerical_state = jnp.zeros((observation.shape[-3],observation.shape[-2])) 
        else: 
            numerical_state = np.amax(observation * np.reshape(np.arange(observation.shape[-1]) + 1, (1, 1, -1)), 2) + 0.5 
        ax.imshow(numerical_state, cmap=cmap, norm=norm)
        ax.set_xticks([])
        ax.set_yticks([])
        # Make spines white
        for spine in ax.spines.values():
            spine.set_color('white')
        ax.text(4.5, 0.5, f"{action_mapping[action]}", ha="center", va="center", color="white", fontsize="x-small")
        if policy_number == 1: 
            ax.set_title(f"t={timestep}")
        if timestep == 0: 
            ax.set_ylabel(rf"$\pi_{{{policy_number}}}$", rotation=0, labelpad=10)
        return 0     

    fig, axs = plt.subplots(nrows=num_runs, ncols=lengths, figsize=(lengths + 1, num_runs + 1))
    for i in range(lengths): 
        for j in range(num_runs): 
            observation = trajectories.state.observation[j, i].squeeze(axis=0)
            action = int(trajectories.action[j, i].squeeze(axis=0))
            terminated = trajectories.state.terminated[j, i]
            visualize_state(observation, axs[j, i], i, action, terminated, j+1)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return 0 


def plot_test_case_trajectory(trajectory, save_path, action_mapping): 
    num_channels = trajectory.state.observation.shape[-1]
    lengths = trajectory.state.observation.shape[1] 
    cmap = sns.color_palette("cubehelix", num_channels)
    cmap.insert(0, (0, 0, 0))
    cmap = colors.ListedColormap(cmap)
    bounds = [i for i in range(num_channels + 2)]
    norm = colors.BoundaryNorm(bounds, num_channels + 1)
    def visualize_state(observation, ax, timestep, action, terminated, policy_number): 
        if terminated: 
            numerical_state = jnp.zeros((observation.shape[-3],observation.shape[-2])) 
        else: 
            numerical_state = np.amax(observation * np.reshape(np.arange(observation.shape[-1]) + 1, (1, 1, -1)), 2) + 0.5 
        ax.imshow(numerical_state, cmap=cmap, norm=norm)
        ax.set_xticks([])
        ax.set_yticks([])
        return 0     
    for i in range(trajectory.state.observation.shape[0]):
        save_path_ = save_path.replace(".pdf", f"_policy_{i+1}.pdf")
        subtraj = jax.tree.map(lambda x: x[i], trajectory)
        fig, axs = plt.subplots(nrows=1, ncols=lengths, figsize=(lengths + 1, 1))
        for i in range(lengths): 
            #for j in range(num_runs): 
            observation = subtraj.state.observation[i].squeeze(axis=0)
            action = int(subtraj.action[i].squeeze(axis=0))
            terminated = subtraj.state.terminated[i]
            visualize_state(observation, axs[i], i, action, terminated, 1)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.savefig(save_path_, format="pdf", bbox_inches="tight")
        plt.close(fig)
    return 0 