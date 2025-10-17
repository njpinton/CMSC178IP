"""
Core methods for Computer Vision and Deep Learning I.
Generates fundamental visualizations for neural networks and CV concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Get figures directory
figures_dir = Path(__file__).parent.parent / "figures"
figures_dir.mkdir(exist_ok=True)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def tanh(x):
    """Tanh activation function."""
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)

def generate_activation_functions():
    """Generate visualization of common activation functions."""
    x = np.linspace(-5, 5, 1000)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Sigmoid
    axes[0, 0].plot(x, sigmoid(x), linewidth=2.5, color='#2E86AB')
    axes[0, 0].set_title('Sigmoid Function', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('x', fontsize=12)
    axes[0, 0].set_ylabel('σ(x)', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
    axes[0, 0].text(0.05, 0.95, r'$\sigma(x) = \frac{1}{1 + e^{-x}}$',
                    transform=axes[0, 0].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ReLU
    axes[0, 1].plot(x, relu(x), linewidth=2.5, color='#A23B72')
    axes[0, 1].set_title('ReLU Function', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('x', fontsize=12)
    axes[0, 1].set_ylabel('ReLU(x)', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linewidth=0.5)
    axes[0, 1].text(0.05, 0.95, r'$ReLU(x) = max(0, x)$',
                    transform=axes[0, 1].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Tanh
    axes[1, 0].plot(x, tanh(x), linewidth=2.5, color='#F18F01')
    axes[1, 0].set_title('Tanh Function', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('x', fontsize=12)
    axes[1, 0].set_ylabel('tanh(x)', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
    axes[1, 0].text(0.05, 0.95, r'$tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$',
                    transform=axes[1, 0].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Leaky ReLU
    axes[1, 1].plot(x, leaky_relu(x), linewidth=2.5, color='#C73E1D')
    axes[1, 1].set_title('Leaky ReLU Function', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('x', fontsize=12)
    axes[1, 1].set_ylabel('Leaky ReLU(x)', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
    axes[1, 1].text(0.05, 0.95, r'$LeakyReLU(x) = max(x, 0.01x)$',
                    transform=axes[1, 1].transAxes, fontsize=11,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(figures_dir / "01_activation_functions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 01_activation_functions.png")

def generate_perceptron_diagram():
    """Generate a simple perceptron diagram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Input nodes
    input_y = [4.5, 3, 1.5]
    for i, y in enumerate(input_y):
        circle = Circle((1.5, y), 0.3, color='#2E86AB', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.5, y, f'$x_{i+1}$', fontsize=14, ha='center', va='center')

    # Output node
    circle = Circle((7, 3), 0.4, color='#A23B72', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(7, 3, 'Σ', fontsize=18, ha='center', va='center', color='white', fontweight='bold')

    # Activation
    circle = Circle((9, 3), 0.35, color='#F18F01', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(9, 3, 'σ', fontsize=16, ha='center', va='center', color='white', fontweight='bold')

    # Connections
    for i, y in enumerate(input_y):
        arrow = FancyArrowPatch((1.8, y), (6.6, 3),
                               arrowstyle='->', mutation_scale=20,
                               linewidth=2, color='gray')
        ax.add_patch(arrow)
        mid_x, mid_y = (1.8 + 6.6) / 2, (y + 3) / 2
        ax.text(mid_x, mid_y + 0.3, f'$w_{i+1}$', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Connection from sum to activation
    arrow = FancyArrowPatch((7.4, 3), (8.65, 3),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='black')
    ax.add_patch(arrow)

    # Output
    arrow = FancyArrowPatch((9.35, 3), (9.8, 3),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2.5, color='black')
    ax.add_patch(arrow)
    ax.text(10.2, 3, '$y$', fontsize=14, ha='center', va='center')

    # Bias
    circle = Circle((4, 5), 0.25, color='#C73E1D', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(4, 5, 'b', fontsize=12, ha='center', va='center', color='white', fontweight='bold')
    arrow = FancyArrowPatch((4, 4.75), (6.7, 3.3),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=1.5, color='gray', linestyle='dashed')
    ax.add_patch(arrow)

    # Title
    ax.text(5, 0.3, 'Single Perceptron Architecture', fontsize=16,
           ha='center', fontweight='bold')

    plt.savefig(figures_dir / "02_perceptron_diagram.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 02_perceptron_diagram.png")

def generate_neural_network_diagram():
    """Generate a multi-layer neural network diagram."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Layer positions
    layer_x = [2, 5, 8, 11]
    layer_neurons = [3, 4, 4, 2]
    layer_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    layer_labels = ['Input\nLayer', 'Hidden\nLayer 1', 'Hidden\nLayer 2', 'Output\nLayer']

    # Draw neurons
    neuron_positions = []
    for layer_idx, (x, n_neurons) in enumerate(zip(layer_x, layer_neurons)):
        positions = []
        start_y = 4 - (n_neurons - 1) * 0.8 / 2
        for i in range(n_neurons):
            y = start_y + i * 1.2
            circle = Circle((x, y), 0.25, color=layer_colors[layer_idx],
                          ec='black', linewidth=1.5, alpha=0.8)
            ax.add_patch(circle)
            positions.append((x, y))
        neuron_positions.append(positions)

        # Layer label
        ax.text(x, 0.5, layer_labels[layer_idx], fontsize=11,
               ha='center', fontweight='bold')

    # Draw connections
    for layer_idx in range(len(neuron_positions) - 1):
        for start_pos in neuron_positions[layer_idx]:
            for end_pos in neuron_positions[layer_idx + 1]:
                ax.plot([start_pos[0], end_pos[0]],
                       [start_pos[1], end_pos[1]],
                       'gray', linewidth=0.5, alpha=0.4)

    # Title
    ax.text(6.5, 7.5, 'Multi-Layer Perceptron (MLP) Architecture',
           fontsize=16, ha='center', fontweight='bold')

    plt.savefig(figures_dir / "03_mlp_architecture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 03_mlp_architecture.png")

def generate_loss_functions():
    """Generate visualization of common loss functions."""
    y_true = 0.7
    y_pred = np.linspace(0.01, 0.99, 1000)

    # Binary Cross-Entropy
    bce = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    # Mean Squared Error
    mse = (y_true - y_pred) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Binary Cross-Entropy
    axes[0].plot(y_pred, bce, linewidth=2.5, color='#2E86AB')
    axes[0].axvline(x=y_true, color='red', linestyle='--', linewidth=2, label='True Value')
    axes[0].set_title('Binary Cross-Entropy Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Predicted Value', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    axes[0].set_ylim([0, 5])

    # Mean Squared Error
    axes[1].plot(y_pred, mse, linewidth=2.5, color='#A23B72')
    axes[1].axvline(x=y_true, color='red', linestyle='--', linewidth=2, label='True Value')
    axes[1].set_title('Mean Squared Error Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Predicted Value', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(figures_dir / "04_loss_functions.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 04_loss_functions.png")

def generate_gradient_descent():
    """Visualize gradient descent optimization."""
    # Create a simple quadratic function
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = X**2 + Y**2

    # Gradient descent path
    learning_rate = 0.1
    x_path = [-4]
    y_path = [3]

    for _ in range(20):
        grad_x = 2 * x_path[-1]
        grad_y = 2 * y_path[-1]
        x_path.append(x_path[-1] - learning_rate * grad_x)
        y_path.append(y_path[-1] - learning_rate * grad_y)

    fig = plt.figure(figsize=(12, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)
    z_path = [x**2 + y**2 for x, y in zip(x_path, y_path)]
    ax1.plot(x_path, y_path, z_path, 'r-o', linewidth=2, markersize=4, label='GD Path')
    ax1.set_xlabel('Parameter 1', fontsize=10)
    ax1.set_ylabel('Parameter 2', fontsize=10)
    ax1.set_zlabel('Loss', fontsize=10)
    ax1.set_title('Gradient Descent on 3D Surface', fontsize=12, fontweight='bold')
    ax1.legend()

    # 2D contour plot
    ax2 = fig.add_subplot(122)
    contour = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
    ax2.clabel(contour, inline=True, fontsize=8)
    ax2.plot(x_path, y_path, 'r-o', linewidth=2, markersize=6, label='GD Path')
    ax2.plot(x_path[0], y_path[0], 'go', markersize=10, label='Start')
    ax2.plot(x_path[-1], y_path[-1], 'rs', markersize=10, label='End')
    ax2.set_xlabel('Parameter 1', fontsize=12)
    ax2.set_ylabel('Parameter 2', fontsize=12)
    ax2.set_title('Gradient Descent Convergence', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "05_gradient_descent.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Generated: 05_gradient_descent.png")

def main():
    """Generate all core method figures."""
    print("Generating core method figures...")
    print("-" * 40)

    generate_activation_functions()
    generate_perceptron_diagram()
    generate_neural_network_diagram()
    generate_loss_functions()
    generate_gradient_descent()

    print("-" * 40)
    print("Core methods figures complete!")

if __name__ == "__main__":
    main()
