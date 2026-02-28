# Project Overview: Advanced Clustering Analysis

This project explores the implementation and performance of K-Means, Fuzzy C-Means (FCM), and their respective Mini-Batch variants on large and high-dimensional datasets.

## File Descriptions

### Core Scripts
- **Algorithms.py**: Contains the manual implementations of all clustering algorithms using NumPy. This includes the logic for both full-batch and mini-batch updates.
- **run_experiments.py**: The main benchmark script. It handles data loading (Covertype and Fashion-MNIST), applies preprocessing (Scaling and PCA), and executes the statistical comparison across 10 trials.
- **run_segmentation.py**: Applies the clustering algorithms to a real-world image to perform color-based and spatial-based image segmentation.

### Research & Reporting
- **report.typ**: The source file for the final report. It uses the Typst markup language and the 'flow-way' package to generate the technical documentation.
- **Anis_Medjdoul_report.pdf**: The compiled version of the technical report, containing all findings, methodology, and visualizations.
- **README.md**: This document, providing a high-level summary of the project structure.


## How to execute

The project is structured to run within the 'ML' or 'torch' Conda environments.
1. Run `python run_experiments.py` to generate the benchmarking results.
2. Run `python run_segmentation.py` to generate the image segmentation results.
3. Use `typst compile report.typ` to update the report PDF if changes are made to the source.
