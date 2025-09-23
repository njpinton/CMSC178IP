#!/usr/bin/env python3
"""
Setup script for the Storage and Compression Interactive Companion
This script helps ensure all dependencies are properly installed and configured.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a system command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def check_python_version():
    """Check if Python version is sufficient"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 7:
        print("✅ Python version is sufficient")
        return True
    else:
        print("❌ Python 3.7 or higher is required")
        return False

def install_requirements():
    """Install required packages"""
    requirements_file = "requirements.txt"

    if not os.path.exists(requirements_file):
        print(f"❌ {requirements_file} not found")
        return False

    print("📦 Installing required packages...")
    command = f"{sys.executable} -m pip install -r {requirements_file}"

    return run_command(command, "Package installation")

def setup_jupyter_widgets():
    """Setup Jupyter widgets for interactive components"""
    print("🔧 Setting up Jupyter widgets...")

    # For Jupyter Notebook
    cmd1 = f"{sys.executable} -m jupyter nbextension enable --py widgetsnbextension --sys-prefix"
    success1 = run_command(cmd1, "Jupyter Notebook widgets setup")

    # For JupyterLab (if available)
    cmd2 = "jupyter labextension install @jupyter-widgets/jupyterlab-manager"
    success2 = run_command(cmd2, "JupyterLab widgets setup (optional)")

    return success1  # JupyterLab extension is optional

def test_imports():
    """Test critical imports"""
    print("🧪 Testing critical imports...")

    critical_packages = [
        "numpy",
        "matplotlib",
        "cv2",
        "PIL",
        "scipy",
        "skimage",
        "ipywidgets"
    ]

    failed_imports = []

    for package in critical_packages:
        try:
            if package == "cv2":
                import cv2
            elif package == "PIL":
                from PIL import Image
            elif package == "skimage":
                from skimage import data
            else:
                __import__(package)
            print(f"✅ {package} imported successfully")
        except ImportError:
            print(f"❌ {package} import failed")
            failed_imports.append(package)

    if failed_imports:
        print(f"\n⚠️  Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All critical packages imported successfully")
        return True

def create_test_notebook():
    """Create a simple test notebook to verify setup"""
    test_content = '''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Setup Test Notebook\\n",
    "This notebook tests that all dependencies are working correctly."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test imports\\n",
    "import numpy as np\\n",
    "import matplotlib.pyplot as plt\\n",
    "import cv2\\n",
    "from PIL import Image\\n",
    "import ipywidgets as widgets\\n",
    "\\n",
    "print('🎉 All imports successful!')\\n",
    "print('✅ Setup is complete and working!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test basic functionality\\n",
    "test_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)\\n",
    "plt.figure(figsize=(6, 4))\\n",
    "plt.imshow(test_array)\\n",
    "plt.title('Test Image - Setup Working!')\\n",
    "plt.axis('off')\\n",
    "plt.show()\\n",
    "\\n",
    "print(f'Image shape: {test_array.shape}')\\n",
    "print('🎯 Basic functionality test passed!')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test interactive widgets\\n",
    "def test_widget(value=5):\\n",
    "    print(f'Widget value: {value}')\\n",
    "    print('🎮 Interactive widgets are working!')\\n",
    "\\n",
    "widgets.interact(test_widget, value=widgets.IntSlider(min=1, max=10, value=5))"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}'''

    try:
        with open("setup_test.ipynb", "w") as f:
            f.write(test_content)
        print("✅ Test notebook created: setup_test.ipynb")
        return True
    except Exception as e:
        print(f"❌ Failed to create test notebook: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Storage and Compression Interactive Companion Setup")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        print("\\n❌ Setup failed: Python version too old")
        return False

    # Install requirements
    if not install_requirements():
        print("\\n❌ Setup failed: Could not install requirements")
        return False

    # Setup Jupyter widgets
    setup_jupyter_widgets()  # This can partially fail but still be okay

    # Test imports
    if not test_imports():
        print("\\n⚠️  Some packages failed to import. Try running:")
        print("   pip install --upgrade -r requirements.txt")
        return False

    # Create test notebook
    create_test_notebook()

    print("\\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\\n📚 Next steps:")
    print("   1. Open Jupyter: jupyter notebook (or jupyter lab)")
    print("   2. Test setup: Open and run 'setup_test.ipynb'")
    print("   3. Start learning: Open '02_Storage_and_Compression_Interactive_Companion.ipynb'")
    print("\\n🎯 Happy learning!")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)