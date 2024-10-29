from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
import sklearn.datasets as sds
import sklearn.neighbors as skn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Initialize Spark Session
spark = SparkSession.builder.appName("Colon Cancer Detection").getOrCreate()

try:
    # Load and preprocess dataset for Spark ML
    data_path = "<your_dataset_path_here>"  # Specify the path to your dataset
    try:
        data = spark.read.format("libsvm").load(data_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise SystemExit("Exiting due to dataset loading error.")

    # Prepare data
    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
    featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
    (trainingData, testData) = data.randomSplit([0.7, 0.3], seed=42)

    # Train a DecisionTree model with a Pipeline
    dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

    # CrossValidator for model tuning
    paramGrid = ParamGridBuilder()\
        .addGrid(dt.maxDepth, [5, 10, 20])\
        .addGrid(dt.impurity, ["gini", "entropy"])\
        .build()

    crossval = CrossValidator(estimator=pipeline,
                              estimatorParamMaps=paramGrid,
                              evaluator=MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy"),
                              numFolds=3)

    # Train model
    cvModel = crossval.fit(trainingData)

    # Make predictions and evaluate
    predictions = cvModel.transform(testData)
    evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
    spark_accuracy = evaluator.evaluate(predictions)
    print(f"Spark Decision Tree Test Accuracy = {spark_accuracy}")

    # Load dataset for scikit-learn
    X, Y = sds.load_svmlight_file(data_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    # Train KNN model with different parameters
    n_neighbors = [3, 5, 7, 9, 11]
    metrics = sorted(skn.VALID_METRICS['brute'])

    best_accuracy = 0
    best_params = {}
    sklearn_accuracies = []

    best_knn_model = None

    for n in n_neighbors:
        for metric in metrics:
            try:
                knn = skn.KNeighborsClassifier(n_neighbors=n, metric=metric)
                knn.fit(X_train, Y_train)
                Y_pred = knn.predict(X_test)
                accuracy = accuracy_score(Y_test, Y_pred)
                sklearn_accuracies.append((n, metric, accuracy))
                print(f"{n} Neighbors & Metric {metric} => Accuracy: {accuracy}")

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_params = {'n_neighbors': n, 'metric': metric}
                    best_knn_model = knn
            except ValueError:
                # Skip invalid metric combinations
                continue

    print(f"Best KNN Accuracy: {best_accuracy} with Parameters: {best_params}")

    # Determine the best model
    if spark_accuracy > best_accuracy:
        best_model = "Spark Decision Tree"
        print("The best model is the Spark Decision Tree.")
    else:
        best_model = "KNN"
        print("The best model is KNN with parameters:", best_params)

    # Ask user if they have a new dataset for prediction
    new_data_path = input("Do you have a new dataset for prediction? Please provide the path or type 'no': ")
    if new_data_path.lower() != 'no':
        if os.path.exists(new_data_path):
            if best_model == "Spark Decision Tree":
                # Load and preprocess new dataset for Spark
                try:
                    new_data = spark.read.format("libsvm").load(new_data_path)
                    new_predictions = cvModel.transform(new_data)
                    new_predictions.select("prediction").show()
                except Exception as e:
                    print(f"Error loading new dataset: {e}")
            else:
                # Load and preprocess new dataset for sklearn
                try:
                    X_new, _ = sds.load_svmlight_file(new_data_path)
                    Y_new_pred = best_knn_model.predict(X_new)
                    print("Predictions for new data:", Y_new_pred)
                except Exception as e:
                    print(f"Error loading new dataset for sklearn: {e}")
        else:
            print("The provided path does not exist. Please check and try again.")

finally:
    # Stop Spark session
    spark.stop()
