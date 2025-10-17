"""
Master script to generate all figures for Segmentation and Morphology topic
Executes all visualization scripts in the correct order
"""

import sys
import subprocess
import os

def run_script(script_name):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print('='*60)

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings/Info:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR running {script_name}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Execute all figure generation scripts"""
    # Change to scripts directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("="*60)
    print("SEGMENTATION AND MORPHOLOGY - Figure Generation")
    print("="*60)

    # List of scripts to run in order
    scripts = [
        'segmentation_methods.py',
        'morphological_operations.py',
        'advanced_applications.py'
    ]

    # Track results
    results = {}

    # Run each script
    for script in scripts:
        success = run_script(script)
        results[script] = success

    # Print summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)

    all_success = True
    for script, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status}: {script}")
        if not success:
            all_success = False

    print("="*60)

    if all_success:
        print("\n🎉 All figures generated successfully!")
        print(f"📁 Figures saved to: {os.path.abspath('../figures/')}")
        return 0
    else:
        print("\n⚠️  Some figures failed to generate. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
