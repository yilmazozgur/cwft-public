import numpy as np
import matplotlib.pyplot as plt

def run_ca_simulation(steps=100, N=141, feedback_type="global_parity"):
    """
    Runs a non-local elementary cellular automaton simulation based on Rule 30.
    
    Parameters:
    - steps: Total time steps to simulate.
    - N: Number of cells in the 1D ring (best if odd for center symmetry).
    - feedback_type: 'global_parity' or 'right_sector_loop'
    """
    # Initialize spacetime grid with a single active central cell
    grid = np.zeros((steps, N), dtype=int)
    grid[0, N // 2] = 1 
    
    # Run the Simulation
    for t in range(steps - 1):
        next_local = np.zeros(N, dtype=int)
        
        # Calculate local Rule 30 mechanics
        for i in range(N):
            left = grid[t, (i - 1) % N]
            center = grid[t, i]
            right = grid[t, (i + 1) % N]
            # Boolean formulation of Rule 30: left XOR (center OR right)
            next_local[i] = left ^ (center | right)
        
        # Apply the chosen non-local feedback matrix/constraint
        if feedback_type == "global_parity":
            # Parity of the entire universe
            G = np.sum(next_local) % 2
        elif feedback_type == "right_sector_loop":
            # Parity restricted exclusively to the rightmost third of the grid
            window_sum = np.sum(next_local[2 * N // 3 :])
            G = int(window_sum % 2)
        else:
            G = 0
            
        # Broadcast the non-local modifier uniformly across all grid nodes
        grid[t + 1, :] = next_local ^ G
        
    # Calculate Spatial Block Entropy (Size = 3) over time
    block_size = 3
    entropies = []
    for t in range(steps):
        blocks = []
        for i in range(N):
            block = tuple(grid[t, (i + j) % N] for j in range(block_size))
            blocks.append(block)
        
        # Find unique configurations and their probabilities
        unique, counts = np.unique(blocks, axis=0, return_counts=True)
        probs = counts / N
        # Compute Shannon Entropy (with a small epsilon to avoid log(0))
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        entropies.append(entropy)
        
    return grid, entropies

def plot_results():
    steps = 100
    N = 141
    
    # Generate data for both conditions
    grid_global, entropy_global = run_ca_simulation(steps, N, "global_parity")
    grid_sector, entropy_sector = run_ca_simulation(steps, N, "right_sector_loop")
    
    # Set up a 2x2 plotting grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top Left: Global Spacetime
    axes[0, 0].imshow(grid_global, cmap='binary', interpolation='nearest')
    axes[0, 0].set_title("Spacetime (Rule 30 + Global Parity Feedback)")
    axes[0, 0].set_xlabel("Spatial Grid (Cells)")
    axes[0, 0].set_ylabel("Time Steps")
    
    # Top Right: Global Entropy
    axes[0, 1].plot(range(steps), entropy_global, color='purple', lw=2, label='Entropy ($H_3$)')
    axes[0, 1].set_title("Global Feedback Entropy Over Time")
    axes[0, 1].set_xlabel("Time Steps")
    axes[0, 1].set_ylabel("Entropy (Bits)")
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend()
    
    # Bottom Left: Sector Spacetime
    axes[1, 0].imshow(grid_sector, cmap='binary', interpolation='nearest')
    axes[1, 1].get_shared_y_axes().join(axes[0, 1], axes[1, 1]) # Match entropy axes scales
    axes[1, 0].set_title("Spacetime (Rule 30 + Right-Sector Matrix Loop)")
    axes[1, 0].set_xlabel("Spatial Grid (Cells)")
    axes[1, 0].set_ylabel("Time Steps")
    
    # Bottom Right: Sector Entropy
    axes[1, 1].plot(range(steps), entropy_sector, color='teal', lw=2, label='Entropy ($H_3$)')
    axes[1, 1].set_title("Sector Feedback Entropy Over Time")
    axes[1, 1].set_xlabel("Time Steps")
    axes[1, 1].set_ylabel("Entropy (Bits)")
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ensure you have numpy and matplotlib installed: pip install numpy matplotlib
    plot_results()
