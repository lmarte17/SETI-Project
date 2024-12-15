# SETI Signal Classification Using Unsupervised Learning

## Project Overview
This project develops an unsupervised learning approach to SETI (Search for Extraterrestrial Intelligence) signal classification. It addresses a key challenge in SETI research: efficiently categorizing the vast amounts of signal data collected from radio telescopes, which traditionally requires manual review by researchers.

The project analyzes a dataset of 7,000 spectrograms containing seven distinct types of signals commonly encountered in SETI research, ranging from narrowband transmissions to squiggle patterns.

## Technical Approach
The project combines advanced image processing, feature engineering, and multiple unsupervised learning techniques:

### 1. Image Processing Pipeline
- Signal enhancement and noise reduction
- Preprocessing and normalization
- Edge detection and pattern enhancement
- Signal profile analysis

### 2. Feature Engineering (72 distinct features)
- Statistical measures of signal intensity
- Frequency components through Fourier transforms
- Gradient information for pattern detection
- Peak characteristics for signal strength analysis

### 3. Unsupervised Learning Techniques
- **Dimensionality Reduction**: t-SNE and UMAP for visualization
- **Clustering Algorithms**: K-means, Agglomerative Clustering, and Gaussian Mixture Models
- **Non-negative Matrix Factorization**: Feature extraction and pattern discovery

## Project Structure
```
.
├── data/               # Signal data directory
├── notebooks/         
│   └── seti_analysis.ipynb  # Main analysis notebook
├── src/
│   ├── image_processor.py   # Image processing utilities
│   ├── preprocessor.py      # Data preprocessing pipeline
│   ├── data_utils.py        # Data handling utilities
│   ├── clustering_utils.py  # Clustering algorithms
│   ├── feature_analysis.py  # Feature extraction tools
│   └── evaluator.py         # Model evaluation tools
├── results/           # Output directory for analysis results
└── requirements.txt   # Project dependencies
```

## Signal Types Analyzed
The project focuses on seven distinct signal types commonly found in SETI data:
- Narrowband signals
- Brightpixel signals
- Squiggle patterns
- Square-pulsed narrowband signals
- Noise patterns
- And other signal variations

## Installation
1. Clone the repository:
```bash
git clone https://github.com/lmarte17/SETI-Project.git
cd SETI-Project
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. Download the SETI signal dataset from [Kaggle](https://www.kaggle.com/datasets/tentotheminus9/seti-data) and place it in the `data/` directory

2. Run the preprocessing pipeline:
```python
from src.preprocessor import SETIPreprocessor
preprocessor = SETIPreprocessor()
preprocessor.load_and_preprocess_images('path/to/data')
```

3. Analyze signals using the image processor:
```python
from src.image_processor import ImageProcessor
processor = ImageProcessor()
# Process and analyze signals
```

4. For detailed analysis and examples, refer to `notebooks/seti_analysis.ipynb`

## Future Development
- Incorporating deep learning techniques:
  - Autoencoders for unsupervised feature learning
  - Convolutional Neural Networks (CNNs) for pattern recognition
  - Variational Autoencoders (VAEs) for signal generation
  - Self-supervised learning approaches

