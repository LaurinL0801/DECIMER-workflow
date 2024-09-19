# DECIMER Workflow
### Chemical Structure Segmentation and Prediction

This script performs the segmentation of chemical structures from PDFs, predicts SMILES and confidence values, and generates a comprehensive analysis with visualizations.

## Overview

- **Segmentation**: The PDFs are converted to individual page images, and chemical structures are segmented from each page.
- **Prediction**: SMILES and confidence values are predicted for each segmented structure using the DECIMER library.
- **Visualization**: Images are generated from the predicted SMILES, and a PDF report is created for each publication, showing segmented and predicted images or error messages.
- **CSV Output**: CSV files are generated, categorizing segmented structures based on confidence values.




## Installation
I recommend using a conda environment for installation.
- Conda (https://docs.anaconda.com/free/miniconda/)

   ```
   git clone https://github.com/LaurinL0801/DECIMER-workflow
   cd DECIMER-workflow
   conda create -n decimer_workflow python=3.10
   conda activate decimer_workflow
   conda install -c conda-forge poppler
   conda install pip
   conda install --upgrade pip
   pip install -r requirements.txt
   ```

## Usage

  ```
  python3 decimer_workflow.py -folder path_to_directory_with_publications -c desired_confidence_value_to_split &> path_to_directory_with_publications/out.txt 2>&1
  ```
## Output 

The segmented images, predicted images, and CSV files are stored in individual directories for each publication.

## Acknowledgements 

This script uses the DECIMER library for image segmentation (https://github.com/Kohulan/DECIMER-Image-Segmentation) and SMILES prediction (https://github.com/Kohulan/DECIMER-Image_Transformer).

## License

This project is licensed under the MIT LICENSE - See the  file for details
