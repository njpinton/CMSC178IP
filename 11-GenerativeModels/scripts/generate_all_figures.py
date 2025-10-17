#!/usr/bin/env python3
"""
Master script to generate all figures for Generative Models module.
Run this script BEFORE compiling LaTeX presentation.

Usage:
    python generate_all_figures.py
"""

import os
import sys

# Create figures directory if needed
os.makedirs('../figures', exist_ok=True)

# Import all generation modules
from generative_core import *
from generative_advanced import *
from generative_applications import *


def main():
    print("=" * 70)
    print("Generating All Figures for Generative Models Module")
    print("=" * 70)
    print()

    # Core concept figures (5 figures)
    print("[1/3] Core Concept Figures")
    print("-" * 70)
    create_generative_vs_discriminative()
    create_autoencoder_architecture()
    create_latent_space_visualization()
    create_vae_architecture()
    create_vae_sampling_process()

    # Advanced technique figures (5 figures)
    print("\n[2/3] Advanced Technique Figures")
    print("-" * 70)
    create_gan_architecture()
    create_gan_training_dynamics()
    create_mode_collapse_visualization()
    create_gan_game_theory()
    create_vae_vs_gan_comparison()

    # Application figures (6 figures)
    print("\n[3/3] Application Figures")
    print("-" * 70)
    create_generation_examples()
    create_latent_space_interpolation()
    create_conditional_generation()
    create_style_transfer_concept()
    create_applications_overview()
    create_training_tips()

    print()
    print("=" * 70)
    print("✓ All figures generated successfully!")
    print("=" * 70)
    print()
    print("📊 Summary:")
    print("   • Total figures: 16 high-quality PNG visualizations")
    print("   • Resolution: 200 DPI (print quality)")
    print("   • Location: ../figures/")
    print()
    print("🎨 Figure Categories:")
    print("   • Core Concepts (5): Fundamentals, autoencoders, VAE")
    print("   • Advanced Topics (5): GANs, training, game theory")
    print("   • Applications (6): Real examples, use cases, tips")
    print()
    print("Next step: Compile LaTeX presentation")
    print("   cd ../slides/")
    print("   pdflatex generative_models_presentation.tex")
    print("=" * 70)


if __name__ == '__main__':
    main()
