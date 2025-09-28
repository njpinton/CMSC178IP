#!/usr/bin/env python3
"""
Master Script for Generating All Noise Reduction Figures

This script executes all figure generation scripts for the noise reduction
techniques educational package.

Author: CMSC 178IP - Digital Image Processing
Course: Noise Reduction Techniques
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name):
    """Run a Python script and capture output"""
    print(f"\n{'='*60}")
    print(f"Executing: {script_name}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name],
                              capture_output=True,
                              text=True,
                              check=True)

        # Print output
        if result.stdout:
            print("Output:")
            print(result.stdout)

        end_time = time.time()
        execution_time = end_time - start_time

        print(f"✅ {script_name} completed successfully in {execution_time:.2f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Error executing {script_name}:")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

    except Exception as e:
        print(f"❌ Unexpected error executing {script_name}: {str(e)}")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'numpy',
        'matplotlib',
        'opencv-python',
        'scipy',
        'scikit-image'
    ]

    print("Checking dependencies...")
    missing_packages = []

    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'scikit-image':
                import skimage
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False

    print("✅ All dependencies satisfied")
    return True

def ensure_output_directory():
    """Ensure the figures output directory exists"""
    figures_dir = '../figures'
    os.makedirs(figures_dir, exist_ok=True)
    print(f"✅ Output directory ensured: {os.path.abspath(figures_dir)}")

def main():
    """Main execution function"""
    print("🎯 Noise Reduction Techniques - Figure Generation")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Ensure output directory exists
    ensure_output_directory()

    # List of scripts to execute in order
    scripts = [
        'core_methods.py',
        'advanced_techniques.py',
        'real_world_examples.py'
    ]

    # Track execution results
    results = {}
    total_start_time = time.time()

    # Execute each script
    for script in scripts:
        if os.path.exists(script):
            success = run_script(script)
            results[script] = success
        else:
            print(f"❌ Script not found: {script}")
            results[script] = False

    # Summary
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")

    successful = sum(results.values())
    total = len(results)

    for script, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{script:<30} {status}")

    print(f"\nOverall: {successful}/{total} scripts executed successfully")
    print(f"Total execution time: {total_execution_time:.2f} seconds")

    # Check output files
    figures_dir = '../figures'
    if os.path.exists(figures_dir):
        png_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        print(f"\n📊 Generated {len(png_files)} figure files:")
        for png_file in sorted(png_files):
            print(f"  • {png_file}")

    if successful == total:
        print(f"\n🎉 All figure generation completed successfully!")
        print("The figures are ready for use in the LaTeX presentation.")
        return 0
    else:
        print(f"\n⚠️  {total - successful} script(s) failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)