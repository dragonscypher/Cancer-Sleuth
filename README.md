# Cancer Sleuth

A coursework project where I compared a few classifiers on colon cancer gene-expression data. The goal was one script that tries several models on a dataset and shows which one performs best, so the same code can be reused on other datasets.

## Models

- Decision tree in PySpark, tuned with cross-validation over `maxDepth` and impurity (gini and entropy).
- K-nearest neighbors in scikit-learn, trying different distance metrics.
- SVM and Naive Bayes in scikit-learn.

All of it lives in `pyspark_colon_cancer_detection.py`. It uses a 70/30 train-test split with a fixed seed and prints the test accuracy for each model.

## Data

The datasets came from my university course page and are not redistributed here. They are in LIBSVM format. Set `data_path` at the top of the script to your own file.

## Running it

```bash
pip install pyspark scikit-learn matplotlib numpy
python pyspark_colon_cancer_detection.py
```
