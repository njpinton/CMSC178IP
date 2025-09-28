#!/usr/bin/env python3
"""
Master script to generate all figures for Frequency Domain Image Enhancement
Executes all individual scripts to create comprehensive figure set
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_script(script_name):
    """Run a Python script and handle errors."""
    print(f"\n🔧 Running {script_name}...")
    try:
        # Change to script directory
        script_dir = Path(__file__).parent
        os.chdir(script_dir)

        # Run the script
        result = subprocess.run([sys.executable, script_name],
                              capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            if result.stdout:
                print(result.stdout)
        else:
            print(f"❌ {script_name} failed with error:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"💥 Error running {script_name}: {e}")
        return False

    return True

def ensure_figures_directory():
    """Ensure the figures directory exists."""
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)
    print(f"📁 Figures directory: {figures_dir}")
    return figures_dir

def main():
    """Main execution function."""
    print("🚀 Starting Frequency Domain Image Enhancement Figure Generation")
    print("=" * 70)

    start_time = time.time()

    # Ensure figures directory exists
    figures_dir = ensure_figures_directory()

    # List of scripts to run in order
    scripts = [
        "core_methods.py",
        "advanced_techniques.py",
        "real_world_examples.py"
    ]

    successful_scripts = []
    failed_scripts = []

    # Run each script
    for script in scripts:
        if run_script(script):
            successful_scripts.append(script)
        else:
            failed_scripts.append(script)

    # Summary
    print("\n" + "=" * 70)
    print("📊 GENERATION SUMMARY")
    print("=" * 70)

    print(f"✅ Successfully completed: {len(successful_scripts)}/{len(scripts)} scripts")
    for script in successful_scripts:
        print(f"   • {script}")

    if failed_scripts:
        print(f"\n❌ Failed scripts: {len(failed_scripts)}")
        for script in failed_scripts:
            print(f"   • {script}")

    # Check generated figures
    if figures_dir.exists():
        figure_files = list(figures_dir.glob("*.png"))
        print(f"\n📊 Generated figures: {len(figure_files)}")
        for fig_file in sorted(figure_files):
            print(f"   • {fig_file.name}")

    # Execution time
    elapsed_time = time.time() - start_time
    print(f"\n⏱️  Total execution time: {elapsed_time:.1f} seconds")

    if failed_scripts:
        print("\n⚠️  Some scripts failed. Check error messages above.")
        return 1
    else:
        print("\n🎉 All figures generated successfully!")
        print("📁 Check the ../figures/ directory for all generated visualizations")
        return 0

if __name__ == "__main__":
    sys.exit(main())