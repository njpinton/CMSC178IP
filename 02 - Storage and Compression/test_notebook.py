#!/usr/bin/env python3
"""
Test script to verify that the companion notebook dependencies work correctly
"""

def test_imports():
    """Test all required imports"""
    try:
        import numpy as np
        print("✅ NumPy imported successfully")

        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")

        import cv2
        print("✅ OpenCV imported successfully")

        from PIL import Image, ImageDraw, ImageFont
        print("✅ PIL imported successfully")

        import os
        print("✅ OS imported successfully")

        import io
        print("✅ IO imported successfully")

        from collections import Counter
        print("✅ Collections imported successfully")

        import heapq
        print("✅ Heapq imported successfully")

        from scipy import ndimage
        print("✅ SciPy imported successfully")

        from skimage import data, filters, color
        print("✅ Scikit-image imported successfully")

        try:
            import ipywidgets as widgets
            print("✅ IPywidgets imported successfully")
        except ImportError:
            print("⚠️  IPywidgets not available (install with: pip install ipywidgets)")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        import numpy as np
        import cv2
        from PIL import Image

        # Create test image
        test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        print("✅ Test image created successfully")

        # Test image conversion
        gray = cv2.cvtColor(test_img, cv2.COLOR_RGB2GRAY)
        print("✅ Image conversion works")

        # Test PIL
        pil_img = Image.fromarray(test_img)
        print("✅ PIL conversion works")

        # Test basic compression simulation
        import tempfile
        import os

        temp_dir = tempfile.mkdtemp()
        try:
            # Test PNG save
            png_path = os.path.join(temp_dir, 'test.png')
            cv2.imwrite(png_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
            png_size = os.path.getsize(png_path)
            print(f"✅ PNG compression test: {png_size} bytes")

            # Test JPEG save
            jpg_path = os.path.join(temp_dir, 'test.jpg')
            cv2.imwrite(jpg_path, cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_JPEG_QUALITY, 90])
            jpg_size = os.path.getsize(jpg_path)
            print(f"✅ JPEG compression test: {jpg_size} bytes")

        finally:
            import shutil
            shutil.rmtree(temp_dir)

        return True

    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def test_algorithms():
    """Test compression algorithms"""
    try:
        from collections import Counter
        import heapq

        # Test Huffman encoding components
        text = "ABRACADABRA"
        freq_counter = Counter(text)
        print(f"✅ Frequency counting works: {dict(freq_counter)}")

        # Test RLE
        def simple_rle(data):
            if not data:
                return []
            encoded = []
            current_char = data[0]
            count = 1
            for char in data[1:]:
                if char == current_char:
                    count += 1
                else:
                    encoded.append((count, current_char))
                    current_char = char
                    count = 1
            encoded.append((count, current_char))
            return encoded

        rle_result = simple_rle(list("AAABBBBCC"))
        print(f"✅ RLE works: {rle_result}")

        return True

    except Exception as e:
        print(f"❌ Algorithm test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Jupyter Notebook Dependencies and Functionality\n")

    print("📦 Testing Imports...")
    imports_ok = test_imports()

    print("\n🔧 Testing Basic Functionality...")
    functionality_ok = test_basic_functionality()

    print("\n🤖 Testing Algorithms...")
    algorithms_ok = test_algorithms()

    print(f"\n📊 Test Results Summary:")
    print(f"   Imports: {'✅' if imports_ok else '❌'}")
    print(f"   Functionality: {'✅' if functionality_ok else '❌'}")
    print(f"   Algorithms: {'✅' if algorithms_ok else '❌'}")

    if imports_ok and functionality_ok and algorithms_ok:
        print("\n🎉 All tests passed! The notebook should run smoothly.")
        print("\n📚 To install missing dependencies:")
        print("   pip install numpy matplotlib opencv-python pillow scipy scikit-image ipywidgets")
        print("   # For Jupyter Lab: jupyter labextension install @jupyter-widgets/jupyterlab-manager")
    else:
        print("\n⚠️  Some tests failed. Please install missing dependencies.")

    return imports_ok and functionality_ok and algorithms_ok

if __name__ == "__main__":
    main()