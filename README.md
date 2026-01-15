# BioMOSurv: Biologically informed Multi-Omics SURvival prediction

The project is organised to support end-to-end workflows, including data preprocessing, unimodal modelling, and multimodal fusion strategies, with an accompanying tutorial notebook.


               
Directory Overview

<li>  preprocess.py: Scripts and utilities for preparing raw biological and clinical data for survival analysis. This may include handling missing values, scaling, feature selection, and dataset alignment across modalities. </li>
<li>  unimodal: Implementations of survival models trained on a single modality (e.g., genomics, imaging-derived features, or clinical variables). These models serve as baselines and building blocks for fusion approaches. </li>
<li> fusion: Multimodal survival models that integrate multiple data sources. This directory focuses on fusion strategies (e.g., early, late, or intermediate fusion) and comparative experiments against unimodal baselines. </li>
<li> 1.Tutorial_bulk.ipynb: A practical tutorial demonstrating how to preprocess data, train unimodal models, and apply fusion methods. This notebook is the recommended starting point for new users. </li>

-Repository Structure
```text
BioMOSurv/
├── preprocess/        # Data cleaning, normalisation, and feature preparation
├── unimodal/          # Unimodal survival models (single data modality)
├── fusion/            # Multimodal fusion models and strategies
└── 1.Tutorial_bulk.ipynb  # Step-by-step tutorial and example workflow
```text

## How to run
The following steps are required to run the code:
<li> Python 3.9.x and R version 4.2.x are required, a check on the specification for the used packages (requirement.txt) is required before running the code. </li>
<li> Jupyter notebook server is required. </li>
<li> Ensure all pip dependencies are installed as listed in requirements.txt. </li>
<li> Run through the steps laid out in the notebook in the order of folders (starting with the preprocessing folder, metabolic modelling, to ML models). </li>

## License

This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Public License for more details.

Le Minh Thao Doan - May 2024
