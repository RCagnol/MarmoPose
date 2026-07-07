import matplotlib.pyplot as plt
import numpy as np

def plot_heatmaps(points, dimensions, indices):
    bin_step = 30
    n_ticks = 4
    positions = np.mean(points[:,np.r_[indices]], axis = 1)
    bins_x = np.arange(0, dimensions[0] + bin_step, bin_step)
    bins_y = np.arange(0, dimensions[1] + bin_step, bin_step)
    bins_z = np.arange(0, dimensions[2] + bin_step, bin_step)
    fig, axs = plt.subplots(1,3)
    heatmap, _, _ = np.histogram2d(positions[:,1],positions[:,0],bins = (bins_y, bins_x))
    axs[0].imshow(heatmap[::-1,:], cmap='viridis', interpolation='nearest', aspect='equal')
    xticks = np.arange(0, heatmap.shape[1] + 1, heatmap.shape[1]/n_ticks)
    yticks = np.arange(0, heatmap.shape[0] + 1, heatmap.shape[0]/n_ticks)
    axs[0].set_xticks(xticks - 0.5, labels = (xticks*bin_step/10).astype(int))
    axs[0].set_yticks(heatmap.shape[0] - yticks - 0.5, labels = (yticks*bin_step/10).astype(int))
    # axs[0].set_title('y VS x heatmap')
    axs[0].set_xlabel('x position (cm)')
    axs[0].set_ylabel('y position (cm)')

    heatmap, _, _ = np.histogram2d(positions[:,2],positions[:,0],bins = (bins_z, bins_x))
    axs[1].imshow(heatmap[::-1,:], cmap='viridis', interpolation='nearest', aspect='equal')
    yticks = np.arange(0, heatmap.shape[0] + 1, heatmap.shape[0]/n_ticks)
    axs[1].set_xticks(xticks - 0.5, labels = (xticks*bin_step/10).astype(int))
    axs[1].set_yticks(heatmap.shape[0] - yticks - 0.5, labels = (yticks*bin_step/10).astype(int))
    # axs[1].set_title('z VS x heatmap')
    axs[1].set_xlabel('x position (cm)')
    axs[1].set_ylabel('z position (cm)')

    heatmap, _, _ = np.histogram2d(positions[:,2],positions[:,1],bins = (bins_z, bins_y))
    axs[2].imshow(heatmap[::-1,:], cmap='viridis', interpolation='nearest', aspect='equal')
    xticks = np.arange(0, heatmap.shape[1] + 1, heatmap.shape[1]/n_ticks)
    axs[2].set_xticks(xticks - 0.5, labels = (xticks*bin_step/10).astype(int))
    axs[2].set_yticks(heatmap.shape[0] - yticks - 0.5, labels = (yticks*bin_step/10).astype(int))
    # axs[2].set_title('z VS y heatmap')
    axs[2].set_xlabel('y position (cm)')
    axs[2].set_ylabel('z position (cm)')
    fig.tight_layout()
    return fig, axs

def plot_distribution_distance(points1, points2, indices,):
    positions1 = np.mean(points1[:,np.r_[indices]], axis = 1)
    positions2 = np.mean(points2[:,np.r_[indices]], axis = 1)
    distances = np.sqrt(np.einsum('ij,ij->i',positions1 - positions2, positions1 - positions2))
    counts, bins = np.histogram(distances,bins=np.arange(0,3001,100))
    fig, ax = plt.subplots()
    ax.stairs(counts/np.sum(counts), bins, fill = True)
    return fig, ax