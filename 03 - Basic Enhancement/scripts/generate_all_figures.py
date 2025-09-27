"""
Master Script for Generating All Basic Enhancement Figures

This script runs all figure generation modules to create a complete
set of visualizations for the Basic Enhancement topic.

CMSC 178IP Digital Image Processing
"""

import os
import sys
import time
from pathlib import Path

def ensure_output_directory():
    """Ensure the figures output directory exists."""
    figures_dir = Path('../figures')
    figures_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory ensured: {figures_dir.absolute()}")

def run_module(module_name, description):
    """Run a specific module and handle errors."""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        if module_name == "core_methods":
            from core_methods import generate_core_figures
            generate_core_figures()
        elif module_name == "advanced_techniques":
            from advanced_techniques import generate_advanced_figures
            generate_advanced_figures()
        elif module_name == "real_world_examples":
            from real_world_examples import generate_real_world_figures
            generate_real_world_figures()

        elapsed_time = time.time() - start_time
        print(f"✅ {description} completed successfully in {elapsed_time:.2f} seconds")
        return True

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ {description} failed after {elapsed_time:.2f} seconds")
        print(f"Error: {str(e)}")
        return False

def create_summary_figure():
    """Create a summary figure showing all enhancement categories."""
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(12, 8))

    categories = ['Histogram\nOperations', 'Point\nOperations', 'Spatial\nFiltering',
                 'Edge\nDetection', 'Adaptive\nFiltering', 'Morphological\nOps',
                 'Medical\nImaging', 'Photography', 'Surveillance']

    techniques_count = [6, 8, 5, 6, 4, 6, 4, 6, 4]

    bars = ax.bar(categories, techniques_count,
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22'])

    ax.set_title('Basic Image Enhancement: Techniques Coverage', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Techniques Demonstrated', fontsize=12)
    ax.set_xlabel('Enhancement Categories', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, techniques_count):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('../figures/enhancement_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Summary figure created")

def generate_figure_index():
    """Generate an index of all created figures."""
    figures_dir = Path('../figures')

    if not figures_dir.exists():
        print("❌ Figures directory not found")
        return

    png_files = list(figures_dir.glob('*.png'))

    if not png_files:
        print("❌ No PNG files found in figures directory")
        return

    print(f"\n📊 Generated Figures Summary")
    print(f"{'='*60}")
    print(f"Total figures created: {len(png_files)}")
    print(f"Output directory: {figures_dir.absolute()}")
    print(f"\nFigures list:")

    for i, fig_path in enumerate(sorted(png_files), 1):
        size_kb = fig_path.stat().st_size / 1024
        print(f"{i:2d}. {fig_path.name} ({size_kb:.1f} KB)")

    total_size_mb = sum(f.stat().st_size for f in png_files) / (1024 * 1024)
    print(f"\nTotal size: {total_size_mb:.2f} MB")

def main():
    """Main execution function."""
    print("🎯 Basic Image Enhancement - Figure Generation")
    print("CMSC 178IP Digital Image Processing")
    print("=" * 60)

    # Ensure output directory exists
    ensure_output_directory()

    # Define modules to run
    modules = [
        ("core_methods", "Generating Core Methods Figures"),
        ("advanced_techniques", "Generating Advanced Techniques Figures"),
        ("real_world_examples", "Generating Real-World Examples Figures")
    ]

    # Track results
    successful_modules = 0
    total_modules = len(modules)

    # Run each module
    for module_name, description in modules:
        if run_module(module_name, description):
            successful_modules += 1

    # Create summary figure
    print(f"\n{'='*60}")
    print("🔄 Creating Summary Figure")
    print(f"{'='*60}")

    try:
        create_summary_figure()
        print("✅ Summary figure created successfully")
    except Exception as e:
        print(f"❌ Summary figure creation failed: {str(e)}")

    # Generate figure index
    generate_figure_index()

    # Final summary
    print(f"\n🎉 FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Modules completed: {successful_modules}/{total_modules}")

    if successful_modules == total_modules:
        print("✅ All figure generation completed successfully!")
        print("📁 Check the '../figures/' directory for all generated images.")
        print("🔗 These figures are ready for use in presentations and notebooks.")
    else:
        print("⚠️  Some modules failed. Check error messages above.")
        print("🔧 Fix errors and re-run this script.")

    print(f"\n📚 Usage Instructions:")
    print(f"- Include figures in LaTeX presentations using \\includegraphics")
    print(f"- Load figures in Jupyter notebooks using plt.imread() or Image.open()")
    print(f"- All figures are saved as high-resolution PNG files")

if __name__ == "__main__":
    main()