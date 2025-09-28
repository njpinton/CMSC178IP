#!/usr/bin/env python3
"""
Master Script for Generating All Image Restoration and Geometric Processing Figures
CMSC 178IP - Digital Image Processing

This script executes all figure generation scripts in the correct order to create
all visualizations for the Image Restoration and Geometric Processing topic.

Author: Noel Jeffrey Pinton
Course: CMSC 178IP - Digital Image Processing
"""

import os
import sys
import subprocess
import time

def ensure_output_dir():
    """Ensure the figures directory exists."""
    os.makedirs('../figures', exist_ok=True)

def run_script(script_name):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Import and run the script
        if script_name == "core_methods.py":
            import core_methods
            core_methods.main()
        elif script_name == "advanced_techniques.py":
            import advanced_techniques
            advanced_techniques.main()
        elif script_name == "real_world_examples.py":
            import real_world_examples
            real_world_examples.main()

        end_time = time.time()
        print(f"\n✅ {script_name} completed successfully in {end_time - start_time:.2f} seconds")
        return True

    except Exception as e:
        end_time = time.time()
        print(f"\n❌ Error in {script_name} after {end_time - start_time:.2f} seconds:")
        print(f"   {type(e).__name__}: {str(e)}")
        return False

def check_dependencies():
    """Check if required packages are available."""
    required_packages = [
        'numpy', 'matplotlib', 'scipy', 'scikit-image', 'opencv-python'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-image':
                import skimage
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install missing packages using:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False

    print("✅ All required packages are available")
    return True

def list_generated_figures():
    """List all generated figures."""
    figures_dir = '../figures'
    if not os.path.exists(figures_dir):
        print("❌ No figures directory found")
        return

    figures = [f for f in os.listdir(figures_dir) if f.endswith('.png')]

    if not figures:
        print("❌ No figures were generated")
        return

    print(f"\n📊 Generated {len(figures)} figures:")
    for i, figure in enumerate(sorted(figures), 1):
        file_path = os.path.join(figures_dir, figure)
        file_size = os.path.getsize(file_path) / 1024  # KB
        print(f"   {i:2d}. {figure:<35} ({file_size:6.1f} KB)")

def main():
    """Execute all figure generation scripts."""
    print("🎯 Image Restoration and Geometric Processing - Figure Generation")
    print("CMSC 178IP - Digital Image Processing")
    print("Author: Noel Jeffrey Pinton")

    # Check dependencies
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)

    # Ensure output directory exists
    ensure_output_dir()
    print(f"📁 Output directory: {os.path.abspath('../figures')}")

    # Scripts to run in order
    scripts = [
        "core_methods.py",
        "advanced_techniques.py",
        "real_world_examples.py"
    ]

    # Track results
    results = {}
    total_start_time = time.time()

    # Run each script
    for script in scripts:
        results[script] = run_script(script)

    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Summary
    print(f"\n{'='*60}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'='*60}")

    successful = sum(results.values())
    total = len(results)

    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Scripts run: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")

    print(f"\n📋 Script Results:")
    for script, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {script:<30} {status}")

    # List generated figures
    list_generated_figures()

    if successful == total:
        print(f"\n🎉 All scripts completed successfully!")
        print(f"   Generated figures are ready for LaTeX presentation")
        print(f"   Figures location: ../figures/")
    else:
        print(f"\n⚠️  Some scripts failed. Please check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()