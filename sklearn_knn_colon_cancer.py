
import sklearn.datasets as sds
import sklearn.neighbors as skn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data_path = "<your_dataset_path_here>"  # Specify the path to your dataset
X, Y = sds.load_svmlight_file(data_path)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Train KNN model with different parameters
n_neighbors = [3, 5, 7, 9, 11]
metrics = sorted(skn.VALID_METRICS['brute'])

best_accuracy = 0
best_params = {}
accuracies = []

for n in n_neighbors:
    for metric in metrics:
        knn = skn.KNeighborsClassifier(n_neighbors=n, metric=metric)
        knn.fit(X_train, Y_train)
        Y_pred = knn.predict(X_test)
        accuracy = accuracy_score(Y_test, Y_pred)
        accuracies.append(accuracy)
        print(f"{n} Neighbors & Metric {metric} => Accuracy: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {'n_neighbors': n, 'metric': metric}

# Visualization of model accuracies
plt.figure(figsize=(10, 6))
plt.plot(n_neighbors, accuracies, marker='o', linestyle='-', color='r')
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Model Accuracy by Number of Neighbors')
plt.show()

# Concluding results
print(f"Best Accuracy: {best_accuracy} with Parameters: {best_params}")
