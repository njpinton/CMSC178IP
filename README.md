# CMSC 178IP - Digital Image Processing

This repository contains the course materials, including lecture notes, Jupyter notebooks, and Python scripts for the Digital Image Processing course (CMSC 178IP) at the University of the Philippines Cebu.

The course covers the fundamentals of digital image processing, from basic image representation and enhancement to advanced topics in computer vision and deep learning.

---

## 📚 Course Modules

The course is divided into the following modules. Each folder contains a `README.md` with more details about the specific topic.

*   [**01 - Introduction and Image Representation**](./01%20-%20Introduction%20and%20Image%20Representation/)
*   [**02 - Storage and Compression**](./02%20-%20Storage%20and%20Compression/)
*   [**03 - Basic Enhancement**](./03%20-%20Basic%20Enhancement/)
*   [**04 - Image Enhancement - Frequency Domain**](./04%20-%20Image%20Enhancement%20-%20Frequency%20Domain/)
*   [**05 - Noise Reduction Techniques**](./05%20-%20Noise%20Reduction%20Techniques/)
*   [**06 - Image Restoration and Geometric Processing**](./06%20-%20Image%20Restoration%20and%20Geometric%20Processing/)
*   [**07 - Feature Extraction**](./07%20-%20Feature%20Extraction/)
*   [**08 - Segmentation and Morphology**](./08-SegmentationMorphology/)
*   [**09 - Computer Vision and Deep Learning I**](./09-ComputerVisionDeepLearningI/)
*   [**10 - Computer Vision and Deep Learning II**](./10-ComputerVisionDeepLearningII/)
*   [**11 - Generative Models**](./11-GenerativeModels/)

---

## ⚙️ Setup and Installation

To run the code in this repository, it is recommended to use a Python virtual environment.

### 1. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

All required Python libraries are listed in the `requirements.txt` file. Install them using pip:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Jupyter Notebooks

Each module contains a `notebooks` directory with interactive Jupyter notebooks (`.ipynb` files). To run them, start the Jupyter server from the root directory:

```bash
jupyter notebook
```

Then, navigate to the desired module and open the notebook file.

### Python Scripts

The `scripts` directory in each module contains Python scripts used to generate figures and demonstrate concepts. You can run them directly from the command line. For example, to generate all figures for Chapter 7:

```bash
python "07 - Feature Extraction/scripts/generate_all_figures.py"
```
