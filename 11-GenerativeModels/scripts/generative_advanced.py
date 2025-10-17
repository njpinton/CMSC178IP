#!/usr/bin/env python3
"""
Advanced Generative Models Visualizations
Covers: GAN architecture, training dynamics, mode collapse
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

# Deep learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import visualkeras

# Set professional style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

# Professional color palette
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']


def create_gan_architecture():
    """
    Create colorful and fun GAN architecture diagram for beginners - it's a battle!
    """
    fig, ax = plt.subplots(figsize=(20, 11))

    from matplotlib.patches import FancyBboxPatch, Circle

    # Fun vibrant colors
    noise_color = '#FFD93D'      # Yellow
    gen_color = '#6BCF7F'        # Green
    fake_color = '#FF8FA3'       # Pink
    real_color = '#74B9FF'       # Blue
    disc_color = '#FD79A8'       # Rose
    judge_color = '#A29BFE'      # Purple

    # GENERATOR SIDE (Left/Top)
    # Starting noise
    noise_box = FancyBboxPatch(
        (0.05, 0.68), 0.08, 0.18,
        boxstyle="round,pad=0.01",
        facecolor=noise_color, edgecolor='white', linewidth=4, zorder=2
    )
    ax.add_patch(noise_box)
    ax.text(0.09, 0.81, '🎲', ha='center', fontsize=30, zorder=3)
    ax.text(0.09, 0.73, 'Random\nNoise', ha='center', fontsize=11,
           weight='bold', color='#2c3e50', zorder=3)

    # Generator layers
    gen_layers_data = [
        {'x': 0.18, 'y': 0.70, 'w': 0.07, 'h': 0.14, 'label': 'Gen\n256', 'icon': '⚡'},
        {'x': 0.30, 'y': 0.695, 'w': 0.075, 'h': 0.15, 'label': 'Gen\n512', 'icon': '⚡'},
        {'x': 0.43, 'y': 0.69, 'w': 0.08, 'h': 0.16, 'label': 'Gen\n1024', 'icon': '⚡'},
    ]

    for layer in gen_layers_data:
        # Shadow
        shadow = plt.Rectangle(
            (layer['x'] + 0.006, layer['y'] - 0.006),
            layer['w'], layer['h'],
            facecolor='gray', alpha=0.3, zorder=1
        )
        ax.add_patch(shadow)

        # Main box
        box = FancyBboxPatch(
            (layer['x'], layer['y']), layer['w'], layer['h'],
            boxstyle="round,pad=0.01",
            facecolor=gen_color, edgecolor='white', linewidth=4, zorder=2
        )
        ax.add_patch(box)

        # Icon and text
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h'] * 0.65,
               layer['icon'], ha='center', fontsize=26, zorder=3)
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h'] * 0.32,
               layer['label'], ha='center', fontsize=10, weight='bold',
               color='#2c3e50', zorder=3)

    # Fake image output
    fake_box = FancyBboxPatch(
        (0.57, 0.68), 0.10, 0.18,
        boxstyle="round,pad=0.01",
        facecolor=fake_color, edgecolor='white', linewidth=4, zorder=2
    )
    ax.add_patch(fake_box)
    ax.text(0.62, 0.81, '🎨', ha='center', fontsize=30, zorder=3)
    ax.text(0.62, 0.73, 'Fake\nImage', ha='center', fontsize=11,
           weight='bold', color='#2c3e50', zorder=3)

    # DISCRIMINATOR SIDE (Right/Bottom)
    # Real image input
    real_box = FancyBboxPatch(
        (0.57, 0.16), 0.10, 0.18,
        boxstyle="round,pad=0.01",
        facecolor=real_color, edgecolor='white', linewidth=4, zorder=2
    )
    ax.add_patch(real_box)
    ax.text(0.62, 0.29, '📷', ha='center', fontsize=30, zorder=3)
    ax.text(0.62, 0.21, 'Real\nImage', ha='center', fontsize=11,
           weight='bold', color='#2c3e50', zorder=3)

    # Discriminator layers
    disc_layers_data = [
        {'x': 0.73, 'y': 0.36, 'w': 0.08, 'h': 0.28, 'label': 'Disc\n512', 'icon': '🔍'},
        {'x': 0.85, 'y': 0.40, 'w': 0.075, 'h': 0.20, 'label': 'Disc\n256', 'icon': '🔍'},
    ]

    for layer in disc_layers_data:
        # Shadow
        shadow = plt.Rectangle(
            (layer['x'] + 0.006, layer['y'] - 0.006),
            layer['w'], layer['h'],
            facecolor='gray', alpha=0.3, zorder=1
        )
        ax.add_patch(shadow)

        # Main box
        box = FancyBboxPatch(
            (layer['x'], layer['y']), layer['w'], layer['h'],
            boxstyle="round,pad=0.01",
            facecolor=disc_color, edgecolor='white', linewidth=4, zorder=2
        )
        ax.add_patch(box)

        # Icon and text
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h'] * 0.62,
               layer['icon'], ha='center', fontsize=26, zorder=3)
        ax.text(layer['x'] + layer['w']/2, layer['y'] + layer['h'] * 0.35,
               layer['label'], ha='center', fontsize=10, weight='bold',
               color='#2c3e50', zorder=3)

    # Final judgment
    judge_circle = Circle((0.95, 0.50), 0.045, facecolor=judge_color,
                          edgecolor='white', linewidth=4, zorder=2)
    ax.add_patch(judge_circle)
    ax.text(0.95, 0.53, '⚖️', ha='center', fontsize=26, zorder=3)
    ax.text(0.95, 0.47, 'Real?\nFake?', ha='center', fontsize=8,
           weight='bold', color='white', zorder=3)

    # ARROWS showing the flow
    arrow_style = dict(arrowstyle='->', lw=3.5, color='#34495e', alpha=0.8)

    # Generator flow
    ax.annotate('', xy=(0.18, 0.77), xytext=(0.135, 0.77), arrowprops=arrow_style)
    ax.annotate('', xy=(0.30, 0.77), xytext=(0.255, 0.77), arrowprops=arrow_style)
    ax.annotate('', xy=(0.43, 0.77), xytext=(0.380, 0.77), arrowprops=arrow_style)
    ax.annotate('', xy=(0.57, 0.77), xytext=(0.515, 0.77), arrowprops=arrow_style)

    # Fake to discriminator
    ax.annotate('', xy=(0.73, 0.62), xytext=(0.62, 0.68),
               arrowprops=dict(arrowstyle='->', lw=4, color=fake_color, alpha=0.8))

    # Real to discriminator
    ax.annotate('', xy=(0.73, 0.38), xytext=(0.62, 0.34),
               arrowprops=dict(arrowstyle='->', lw=4, color=real_color, alpha=0.8))

    # Discriminator flow
    ax.annotate('', xy=(0.85, 0.50), xytext=(0.815, 0.50), arrowprops=arrow_style)
    ax.annotate('', xy=(0.905, 0.50), xytext=(0.930, 0.50), arrowprops=arrow_style)

    # Feedback loop (dashed)
    ax.annotate('', xy=(0.25, 0.65), xytext=(0.88, 0.55),
               arrowprops=dict(arrowstyle='<-', lw=3, color='#e74c3c',
                             linestyle='dashed', alpha=0.7,
                             connectionstyle="arc3,rad=0.3"))
    ax.text(0.55, 0.58, '💥 Training Signal!', ha='center', fontsize=12,
           weight='bold', color='#e74c3c', style='italic',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe6e6',
                    edgecolor='#e74c3c', linewidth=2))

    # TITLE BADGES
    def add_super_badge(x, y, text, bg_color, emoji, subtitle):
        # Main badge
        badge = FancyBboxPatch(
            (x - 0.12, y - 0.05), 0.24, 0.10,
            boxstyle="round,pad=0.015",
            facecolor=bg_color, edgecolor='white', linewidth=4, zorder=5
        )
        ax.add_patch(badge)

        ax.text(x - 0.08, y, emoji, fontsize=28, ha='center', va='center', zorder=6)
        ax.text(x + 0.04, y + 0.01, text, fontsize=16, weight='bold',
               color='white', ha='center', va='center', zorder=6)

        # Subtitle
        ax.text(x, y - 0.08, subtitle, fontsize=10, style='italic',
               ha='center', color='#7f8c8d')

    add_super_badge(0.31, 0.93, 'GENERATOR', '#27ae60', '🎨',
                   '"The Artist" - Creates fakes')
    add_super_badge(0.84, 0.93, 'DISCRIMINATOR', '#c0392b', '🕵️',
                   '"The Detective" - Catches fakes')

    # BATTLE INFO BOX
    battle_box = FancyBboxPatch(
        (0.30, 0.02), 0.40, 0.10,
        boxstyle="round,pad=0.015",
        facecolor='#FFF9C4', edgecolor='#F39C12', linewidth=3, zorder=2
    )
    ax.add_patch(battle_box)
    ax.text(0.50, 0.09, '⚔️ THE BATTLE ⚔️', ha='center', fontsize=15,
           weight='bold', color='#E67E22', zorder=3)
    ax.text(0.50, 0.05, 'Generator tries to FOOL | Discriminator tries to CATCH',
           ha='center', fontsize=11, style='italic', color='#7F8C8D', zorder=3)
    ax.text(0.50, 0.022, 'They train together and both get better!',
           ha='center', fontsize=10, weight='bold', color='#2ECC71', zorder=3)

    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.0)
    ax.axis('off')
    ax.set_title('🎮 GAN: Two Networks Battle Each Other! 🎮',
                fontsize=22, weight='bold', color='#2c3e50', pad=20)

    plt.tight_layout()
    plt.savefig('../figures/gan_architecture.png', dpi=200,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated gan_architecture.png")


def create_gan_training_dynamics():
    """
    Visualize GAN training dynamics over iterations
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    np.random.seed(42)
    iterations = np.arange(0, 1000, 10)

    # Generator loss (decreasing with noise)
    g_loss_base = 2.5 * np.exp(-iterations/400) + 0.5
    g_loss = g_loss_base + 0.1 * np.random.randn(len(iterations))

    # Discriminator loss (oscillating, converging)
    d_loss_real_base = 0.7 - 0.3 * np.exp(-iterations/300)
    d_loss_real = d_loss_real_base + 0.05 * np.random.randn(len(iterations))

    d_loss_fake_base = 2.0 * np.exp(-iterations/350) + 0.3
    d_loss_fake = d_loss_fake_base + 0.08 * np.random.randn(len(iterations))

    d_loss_total = (d_loss_real + d_loss_fake) / 2

    # Plot 1: Loss curves
    ax = axes[0, 0]
    ax.plot(iterations, g_loss, linewidth=2.5, color=colors[2], label='Generator Loss', alpha=0.8)
    ax.plot(iterations, d_loss_total, linewidth=2.5, color=colors[1], label='Discriminator Loss', alpha=0.8)

    ax.set_xlabel('Training Iteration', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('GAN Training: Loss Curves', fontweight='bold', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 2: Discriminator accuracy
    ax = axes[0, 1]
    d_acc_real = 0.5 + 0.45 * (1 - np.exp(-iterations/300)) + 0.02 * np.random.randn(len(iterations))
    d_acc_fake = 0.5 + 0.45 * (1 - np.exp(-iterations/300)) + 0.02 * np.random.randn(len(iterations))

    ax.plot(iterations, d_acc_real, linewidth=2.5, color=colors[0], label='D Accuracy (Real)', alpha=0.8)
    ax.plot(iterations, d_acc_fake, linewidth=2.5, color=colors[2], label='D Accuracy (Fake)', alpha=0.8)
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Random Guess')

    ax.set_xlabel('Training Iteration', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Discriminator Accuracy Over Time', fontweight='bold', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.05)

    # Plot 3: Generated sample quality (simulated)
    ax = axes[1, 0]
    quality_metric = 1 - np.exp(-iterations/400) + 0.03 * np.random.randn(len(iterations))
    quality_metric = np.clip(quality_metric, 0, 1)

    ax.plot(iterations, quality_metric, linewidth=3, color=colors[3], alpha=0.8)
    ax.fill_between(iterations, quality_metric, alpha=0.3, color=colors[3])

    ax.set_xlabel('Training Iteration', fontweight='bold')
    ax.set_ylabel('Sample Quality Score', fontweight='bold')
    ax.set_title('Generated Sample Quality (Simulated)', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    # Add quality milestones
    ax.axvline(200, color='red', linestyle=':', alpha=0.7)
    ax.text(200, 0.95, 'Early\nStage', ha='center', fontsize=9)
    ax.axvline(500, color='orange', linestyle=':', alpha=0.7)
    ax.text(500, 0.95, 'Mid\nStage', ha='center', fontsize=9)
    ax.axvline(800, color='green', linestyle=':', alpha=0.7)
    ax.text(800, 0.95, 'Converged', ha='center', fontsize=9)

    # Plot 4: Nash equilibrium convergence
    ax = axes[1, 1]

    # Distance from Nash equilibrium (decreasing)
    nash_distance = 1.5 * np.exp(-iterations/350) + 0.05 * np.random.randn(len(iterations))
    nash_distance = np.abs(nash_distance)

    ax.plot(iterations, nash_distance, linewidth=3, color=colors[5], alpha=0.8)
    ax.fill_between(iterations, nash_distance, alpha=0.3, color=colors[5])
    ax.axhline(0, color='green', linestyle='--', linewidth=2, label='Nash Equilibrium')

    ax.set_xlabel('Training Iteration', fontweight='bold')
    ax.set_ylabel('Distance from Equilibrium', fontweight='bold')
    ax.set_title('Convergence to Nash Equilibrium', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('GAN Training Dynamics', fontsize=18, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('../figures/gan_training_dynamics.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated gan_training_dynamics.png")


def create_mode_collapse_visualization():
    """
    Illustrate mode collapse problem in GANs
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    np.random.seed(42)

    # True data distribution (multi-modal)
    ax = axes[0]

    # Create multiple modes
    modes = [
        {'center': (-2, 2), 'samples': 50, 'color': colors[0]},
        {'center': (2, 2), 'samples': 50, 'color': colors[1]},
        {'center': (-2, -2), 'samples': 50, 'color': colors[2]},
        {'center': (2, -2), 'samples': 50, 'color': colors[3]},
        {'center': (0, 0), 'samples': 50, 'color': colors[4]},
    ]

    for mode in modes:
        x = np.random.randn(mode['samples']) * 0.4 + mode['center'][0]
        y = np.random.randn(mode['samples']) * 0.4 + mode['center'][1]
        ax.scatter(x, y, c=[mode['color']], s=80, alpha=0.6, edgecolors='black', linewidth=1)

    ax.set_xlabel('Feature 1', fontweight='bold')
    ax.set_ylabel('Feature 2', fontweight='bold')
    ax.set_title('True Data Distribution\n(5 Modes)', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')

    # Good generator (captures all modes)
    ax = axes[1]

    for mode in modes:
        x = np.random.randn(mode['samples']) * 0.5 + mode['center'][0]
        y = np.random.randn(mode['samples']) * 0.5 + mode['center'][1]
        ax.scatter(x, y, c=[mode['color']], s=80, alpha=0.6, edgecolors='black', linewidth=1, marker='s')

    ax.set_xlabel('Feature 1', fontweight='bold')
    ax.set_ylabel('Feature 2', fontweight='bold')
    ax.set_title('✓ Good Generator\n(All Modes Captured)', fontweight='bold', fontsize=14, color='green')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')

    # Mode collapse (only one or two modes)
    ax = axes[2]

    # Only generate from one dominant mode
    collapsed_mode = modes[1]  # Only the (2, 2) mode
    x = np.random.randn(250) * 0.6 + collapsed_mode['center'][0]
    y = np.random.randn(250) * 0.6 + collapsed_mode['center'][1]
    ax.scatter(x, y, c=[collapsed_mode['color']], s=80, alpha=0.6, edgecolors='black', linewidth=1, marker='^')

    # Show missing modes as ghosts
    for i, mode in enumerate(modes):
        if i != 1:  # Not the collapsed mode
            ax.scatter(mode['center'][0], mode['center'][1], c=[mode['color']],
                      s=300, alpha=0.2, marker='x', linewidths=3)

    ax.set_xlabel('Feature 1', fontweight='bold')
    ax.set_ylabel('Feature 2', fontweight='bold')
    ax.set_title('✗ Mode Collapse\n(Missing 4 Modes)', fontweight='bold', fontsize=14, color='red')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')

    plt.suptitle('Mode Collapse Problem in GANs', fontsize=18, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('../figures/mode_collapse.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated mode_collapse.png")


def create_gan_game_theory():
    """
    Visualize GAN as a two-player game
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Payoff matrix visualization
    ax = axes[0]

    # Create game theory payoff visualization
    strategies_g = ['Low Quality', 'Medium Quality', 'High Quality']
    strategies_d = ['Weak Disc.', 'Medium Disc.', 'Strong Disc.']

    payoff_g = np.array([
        [3, 2, 1],
        [4, 3, 2],
        [5, 4, 3]
    ])

    payoff_d = np.array([
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5]
    ])

    # Show heatmap
    im = ax.imshow(payoff_g - payoff_d, cmap='RdYlGn', aspect='auto', vmin=-3, vmax=3)

    # Labels
    ax.set_xticks(np.arange(len(strategies_d)))
    ax.set_yticks(np.arange(len(strategies_g)))
    ax.set_xticklabels(strategies_d)
    ax.set_yticklabels(strategies_g)

    ax.set_xlabel('Discriminator Strategy', fontweight='bold')
    ax.set_ylabel('Generator Strategy', fontweight='bold')
    ax.set_title('GAN as Two-Player Game\n(Green = G Advantage, Red = D Advantage)',
                fontweight='bold', fontsize=12)

    # Add text annotations
    for i in range(len(strategies_g)):
        for j in range(len(strategies_d)):
            text = ax.text(j, i, f'G:{payoff_g[i, j]}\nD:{payoff_d[i, j]}',
                          ha="center", va="center", color="black", fontsize=9, weight='bold')

    # Nash equilibrium
    ax.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, fill=False, edgecolor='blue', linewidth=4))
    ax.text(2, 2.7, '★ Nash\nEquilibrium', ha='center', fontsize=9, weight='bold',
           color='blue', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    plt.colorbar(im, ax=ax, label='Advantage')

    # Convergence landscape
    ax = axes[1]

    # Create contour plot showing loss landscape
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)

    # Simulated loss landscape (saddle point)
    Z = X**2 - Y**2

    contour = ax.contour(X, Y, Z, levels=15, cmap='coolwarm', alpha=0.6)
    ax.contourf(X, Y, Z, levels=15, cmap='coolwarm', alpha=0.3)
    ax.clabel(contour, inline=True, fontsize=8)

    # Show Nash equilibrium point
    ax.plot(0, 0, 'g*', markersize=30, label='Nash Equilibrium', markeredgecolor='black', markeredgewidth=2)

    # Show training trajectory (spiral towards equilibrium)
    theta = np.linspace(0, 4*np.pi, 100)
    r = 2 * np.exp(-theta/(2*np.pi))
    traj_x = r * np.cos(theta)
    traj_y = r * np.sin(theta)
    ax.plot(traj_x, traj_y, 'b-', linewidth=2, alpha=0.7, label='Training Trajectory')
    ax.plot(traj_x[0], traj_y[0], 'ro', markersize=10, label='Start')

    # Arrows showing direction
    for i in range(0, len(traj_x)-1, 15):
        ax.annotate('', xy=(traj_x[i+1], traj_y[i+1]), xytext=(traj_x[i], traj_y[i]),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='blue'))

    ax.set_xlabel('Generator Parameter Space', fontweight='bold')
    ax.set_ylabel('Discriminator Parameter Space', fontweight='bold')
    ax.set_title('Convergence to Nash Equilibrium\n(Saddle Point)', fontweight='bold', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.suptitle('GAN Training as Game Theory', fontsize=18, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('../figures/gan_game_theory.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated gan_game_theory.png")


def create_vae_vs_gan_comparison():
    """
    Compare VAE and GAN approaches side by side with enhanced design
    """
    fig = plt.figure(figsize=(18, 10))

    # Define color scheme
    vae_color = '#3498db'
    gan_color = '#e74c3c'
    highlight_color = '#f39c12'

    # Create grid
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3,
                         left=0.08, right=0.92, top=0.92, bottom=0.08)

    # ============ VAE COLUMN ============

    # VAE Title
    ax_vae_title = fig.add_subplot(gs[0, 0])
    ax_vae_title.text(0.5, 0.5, 'Variational\nAutoencoder (VAE)',
                     ha='center', va='center',
                     fontsize=26, weight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=1.2', facecolor=vae_color,
                              edgecolor='black', linewidth=3))
    ax_vae_title.set_xlim(0, 1)
    ax_vae_title.set_ylim(0, 1)
    ax_vae_title.axis('off')

    # VAE Training Objective
    ax_vae_obj = fig.add_subplot(gs[1, 0])
    ax_vae_obj.text(0.5, 0.85, 'TRAINING OBJECTIVE', ha='center',
                   fontsize=16, weight='bold', color=vae_color)

    # Objective equation in box
    ax_vae_obj.add_patch(plt.Rectangle((0.1, 0.55), 0.8, 0.22,
                                       facecolor='#ecf0f1', edgecolor=vae_color,
                                       linewidth=2.5, zorder=1))
    ax_vae_obj.text(0.5, 0.66, 'Maximize ELBO', ha='center',
                   fontsize=14, weight='bold', color='black')
    ax_vae_obj.text(0.5, 0.58, r'$\mathcal{L} = \mathbb{E}[\log p(x|z)] - D_{KL}[q(z|x) \| p(z)]$',
                   ha='center', fontsize=13, color='black', style='italic')

    # Key characteristics
    ax_vae_obj.text(0.15, 0.40, '✓', fontsize=20, color='green', weight='bold')
    ax_vae_obj.text(0.25, 0.40, 'Explicit likelihood', fontsize=13, va='center')

    ax_vae_obj.text(0.15, 0.28, '✓', fontsize=20, color='green', weight='bold')
    ax_vae_obj.text(0.25, 0.28, 'Stable training', fontsize=13, va='center')

    ax_vae_obj.text(0.15, 0.16, '✓', fontsize=20, color='green', weight='bold')
    ax_vae_obj.text(0.25, 0.16, 'Structured latent space', fontsize=13, va='center')

    ax_vae_obj.text(0.15, 0.04, '✗', fontsize=20, color='red', weight='bold')
    ax_vae_obj.text(0.25, 0.04, 'Blurry/smooth outputs', fontsize=13, va='center')

    ax_vae_obj.set_xlim(0, 1)
    ax_vae_obj.set_ylim(0, 1)
    ax_vae_obj.axis('off')

    # VAE Characteristics Table
    ax_vae_char = fig.add_subplot(gs[2, 0])

    characteristics = [
        ('Training Stability', '★★★★★', 'green'),
        ('Output Quality', '★★★☆☆', 'orange'),
        ('Mode Coverage', '★★★★☆', 'green'),
        ('Latent Space', '★★★★★', 'green'),
        ('Speed', '★★★★☆', 'green'),
    ]

    y_pos = 0.88
    ax_vae_char.text(0.5, 0.95, 'CHARACTERISTICS', ha='center',
                    fontsize=16, weight='bold', color=vae_color)

    for char, rating, color_rate in characteristics:
        ax_vae_char.text(0.05, y_pos, char, fontsize=13, weight='bold', va='center')
        ax_vae_char.text(0.95, y_pos, rating, fontsize=13, ha='right',
                        color=color_rate, weight='bold', va='center')
        y_pos -= 0.18

    ax_vae_char.set_xlim(0, 1)
    ax_vae_char.set_ylim(0, 1)
    ax_vae_char.axis('off')

    # Add background
    ax_vae_char.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.91,
                                        facecolor='#e8f4f8', edgecolor=vae_color,
                                        linewidth=2, alpha=0.3, zorder=-1))

    # ============ GAN COLUMN ============

    # GAN Title
    ax_gan_title = fig.add_subplot(gs[0, 1])
    ax_gan_title.text(0.5, 0.5, 'Generative\nAdversarial Network (GAN)',
                     ha='center', va='center',
                     fontsize=26, weight='bold', color='white',
                     bbox=dict(boxstyle='round,pad=1.2', facecolor=gan_color,
                              edgecolor='black', linewidth=3))
    ax_gan_title.set_xlim(0, 1)
    ax_gan_title.set_ylim(0, 1)
    ax_gan_title.axis('off')

    # GAN Training Objective
    ax_gan_obj = fig.add_subplot(gs[1, 1])
    ax_gan_obj.text(0.5, 0.85, 'TRAINING OBJECTIVE', ha='center',
                   fontsize=16, weight='bold', color=gan_color)

    # Objective equation in box
    ax_gan_obj.add_patch(plt.Rectangle((0.1, 0.55), 0.8, 0.22,
                                       facecolor='#ecf0f1', edgecolor=gan_color,
                                       linewidth=2.5, zorder=1))
    ax_gan_obj.text(0.5, 0.66, 'Minimax Game', ha='center',
                   fontsize=14, weight='bold', color='black')
    ax_gan_obj.text(0.5, 0.58, r'$\min_G \max_D V(D,G) = \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$',
                   ha='center', fontsize=11, color='black', style='italic')

    # Key characteristics
    ax_gan_obj.text(0.15, 0.40, '✓', fontsize=20, color='green', weight='bold')
    ax_gan_obj.text(0.25, 0.40, 'Sharp, realistic outputs', fontsize=13, va='center')

    ax_gan_obj.text(0.15, 0.28, '✓', fontsize=20, color='green', weight='bold')
    ax_gan_obj.text(0.25, 0.28, 'High sample quality', fontsize=13, va='center')

    ax_gan_obj.text(0.15, 0.16, '✗', fontsize=20, color='red', weight='bold')
    ax_gan_obj.text(0.25, 0.16, 'Training instability', fontsize=13, va='center')

    ax_gan_obj.text(0.15, 0.04, '✗', fontsize=20, color='red', weight='bold')
    ax_gan_obj.text(0.25, 0.04, 'Mode collapse risk', fontsize=13, va='center')

    ax_gan_obj.set_xlim(0, 1)
    ax_gan_obj.set_ylim(0, 1)
    ax_gan_obj.axis('off')

    # GAN Characteristics Table
    ax_gan_char = fig.add_subplot(gs[2, 1])

    characteristics_gan = [
        ('Training Stability', '★★☆☆☆', 'red'),
        ('Output Quality', '★★★★★', 'green'),
        ('Mode Coverage', '★★☆☆☆', 'orange'),
        ('Latent Space', '★★☆☆☆', 'orange'),
        ('Speed', '★★★☆☆', 'orange'),
    ]

    y_pos = 0.88
    ax_gan_char.text(0.5, 0.95, 'CHARACTERISTICS', ha='center',
                    fontsize=16, weight='bold', color=gan_color)

    for char, rating, color_rate in characteristics_gan:
        ax_gan_char.text(0.05, y_pos, char, fontsize=13, weight='bold', va='center')
        ax_gan_char.text(0.95, y_pos, rating, fontsize=13, ha='right',
                        color=color_rate, weight='bold', va='center')
        y_pos -= 0.18

    ax_gan_char.set_xlim(0, 1)
    ax_gan_char.set_ylim(0, 1)
    ax_gan_char.axis('off')

    # Add background
    ax_gan_char.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.91,
                                        facecolor='#fce8e6', edgecolor=gan_color,
                                        linewidth=2, alpha=0.3, zorder=-1))

    # Main title
    fig.suptitle('VAE vs GAN: Comprehensive Comparison',
                fontsize=24, weight='bold', y=0.98)

    plt.savefig('../figures/vae_vs_gan.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  ✓ Generated vae_vs_gan.png")


def main():
    """Generate all advanced figures"""
    print("\n" + "="*70)
    print("Generating Advanced Generative Models Figures")
    print("="*70)

    create_gan_architecture()
    create_gan_training_dynamics()
    create_mode_collapse_visualization()
    create_gan_game_theory()
    create_vae_vs_gan_comparison()

    print("\n" + "="*70)
    print("✓ Advanced figures generated successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
