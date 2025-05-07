## Project Overview

The project aims to:
- Analyze the PLOD-CW-25 dataset for abbreviation and long-form detection
- Implement and compare different token classification approaches
- Evaluate model performance using standard metrics
- Provide insights into biomedical abbreviation patterns

## Dataset

The PLOD-CW-25 dataset is a specialized biomedical corpus containing:
- 2,400 examples (2,000 train, 150 validation, 250 test)
- Approximately 89,250 tokens
- 3,821 unique abbreviations
- BIO-format annotations for abbreviations (B-AC) and long forms (B-LF, I-LF)

Download the dataset using the script 'scripts/download_dataset.py'

## Setup and Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd NLP-COMM061
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

### Data Analysis
The `data_analysis.ipynb` notebook contains comprehensive analysis of the dataset, including:
- Basic statistics and distributions
- Abbreviation frequency analysis
- Domain-specific patterns
- Contextual analysis
- Visualization of key patterns

### Running the Analysis
1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `notebooks/data_analysis.ipynb`

3. Run the cells to perform the analysis

## Requirements

- Python 3.8+
- Jupyter Notebook
- Required packages (see requirements.txt):
  - datasets
  - numpy
  - pandas
  - matplotlib
  - scikit-learn