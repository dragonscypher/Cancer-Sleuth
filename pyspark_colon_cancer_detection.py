
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt

# Initialize Spark Session
spark = SparkSession.builder.appName("Colon Cancer Detection").getOrCreate()

# Load and preprocess dataset
data_path = "<your_dataset_path_here>"  # Specify the path to your dataset
data = spark.read.format("libsvm").load(data_path)

# Prepare data
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)
featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)
(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a DecisionTree model with a Pipeline
dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures")
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# CrossValidator for model tuning
paramGrid = ParamGridBuilder()     .addGrid(dt.maxDepth, [5, 10, 20])     .addGrid(dt.impurity, ["gini", "entropy"])     .build()

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy"),
                          numFolds=3)

# Train model
cvModel = crossval.fit(trainingData)

# Make predictions and evaluate
predictions = cvModel.transform(testData)
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Test Accuracy = {accuracy}")

# Visualization of model accuracy for comparison
# Assuming you have a list of model names and their accuracies
model_names = ['Decision Tree']
accuracies = [accuracy]

plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracies, color='blue')
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Comparison')
plt.show()

# Stop Spark session
spark.stop()
