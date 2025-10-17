"""
Master script to generate all figures for Computer Vision and Deep Learning I module.
Executes all figure generation scripts in the correct order.
"""

import sys
import os
from pathlib import Path

# Add the scripts directory to the Python path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

def main():
    """Generate all figures for the Computer Vision and Deep Learning I module."""
    print("="*60)
    print("GENERATING ALL CV & DEEP LEARNING I FIGURES")
    print("="*60)

    # Ensure figures directory exists
    figures_dir = script_dir.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    print(f"Figures will be saved to: {figures_dir}")
    print()

    # Import and run each script
    scripts = [
        ("Core Methods", "core_methods"),
        ("Advanced Techniques", "advanced_techniques"),
        ("Real World Examples", "real_world_examples")
    ]

    for script_name, module_name in scripts:
        print(f"Executing {script_name}...")
        print("-" * 40)

        try:
            # Import the module
            module = __import__(module_name)

            # Execute the main function if it exists
            if hasattr(module, 'main'):
                module.main()
            else:
                print(f"Module {module_name} executed successfully")

        except Exception as e:
            print(f"Error executing {script_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"✓ {script_name} completed")
        print()

    print("="*60)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*60)

    # List generated figures
    if figures_dir.exists():
        figures = list(figures_dir.glob("*.png"))
        print(f"Generated {len(figures)} figures:")
        for fig in sorted(figures):
            print(f"  - {fig.name}")

    print()
    print("Next steps:")
    print("1. Review the generated figures in the '../figures/' directory")
    print("2. Compile the LaTeX presentation using the generated figures")
    print("3. Test the Jupyter notebook with the available figures")

if __name__ == "__main__":
    main()
