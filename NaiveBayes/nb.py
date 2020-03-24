# Make Predictions with Naive Bayes On The Iris Dataset
from csv import reader
from math import sqrt, exp, pi


class Dataset:
    def __init__(self, dataset_path="iris.csv"):
        self.path = dataset_path

    # Load a CSV file
    def load_csv(self):
        dataset = list()
        with open(self.path, "r") as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset

    def str_column_to_float(self, dataset, column):
        """
        In this particular file we have strings instead of numbers, so we just convert them
        """
        for row in dataset:
            row[column] = float(row[column].strip())

    # Convert string column to integer
    def str_column_to_int(self, dataset, column):
        """
        Iris-setosa => 0
        Iris-virginica => 1
        Iris-versicolor => 2
        """
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup

    # Split the dataset by class values, returns a dictionary
    def separate_by_class(self, dataset):
        separated = dict()
        for i in range(len(dataset)):
            row = dataset[i]
            class_value = row[-1]
            if class_value not in separated:
                separated[class_value] = list()
            separated[class_value].append(row)
        return separated

    def get_dataset(self):
        dataset = self.load_csv()
        for i in range(
            len(dataset[0]) - 1
        ):  # dataset[0] because we want to know the len of a row
            self.str_column_to_float(dataset, i)

        # convert class column to integers
        self.str_column_to_int(dataset, len(dataset[0]) - 1)
        return self.separate_by_class(dataset)


class NaiveBayes:
    def __init__(self):
        data = Dataset()
        self.dataset = data.get_dataset()

    # Calculate the mean of a list of numbers
    def mean(self, k):
        return sum(k) / float(len(k))

    # Calculate the standard deviation of a list of numbers
    def standard_deviation(self, k):
        mean_value = self.mean(k)
        variance = sum([(x - mean_value) ** 2 for x in k]) / float(len(k) - 1)
        return sqrt(variance)

    # Calculate the mean, standard_deviation and count for each column in a dataset
    def summarize_dataset(self, dataset):
        summaries = [
            (self.mean(column), self.standard_deviation(column), len(column))
            for column in zip(*dataset)
        ]
        del summaries[-1]
        return summaries

    # Split dataset by class then calculate statistics for each row
    def summarize_by_class(self):
        summaries = dict()
        for class_value, rows in self.dataset.items():
            summaries[class_value] = self.summarize_dataset(rows)
        return summaries

    # Calculate the Gaussian probability distribution function for x
    def calculate_probability(self, x, mean, standard_deviation):
        exponent = exp(-((x - mean) ** 2 / (2 * standard_deviation ** 2)))
        return (1 / (sqrt(2 * pi) * standard_deviation)) * exponent

    # Calculate the probabilities of predicting each class for a given row
    def calculate_class_probabilities(self, statistical_ds, row):
        """
        Input:
        statistical_ds -> dict like: {class_number:
            [(mean, standard deviation, len for the first column),
            ((mean, standard deviation, len for the second column)),
            ((mean, standard deviation, len for the third column))]}
            for class_number in given classes
        row -> new row
        total_rows take row count from len for the x column part of statistical dataset
        This is used in the calculation of the P(class) as the ratio of rows with a given
        class of all rows in the training data (total_rows).

        Next, probabilities are calculated for each input value in the row using
        the Gaussian probability density function and the statistics for that column
        and of that class. Probabilities are multiplied together as they are accumulated.
        """
        total_rows = sum([statistical_ds[label][0][2] for label in statistical_ds])
        probabilities = dict()
        for class_value, class_summaries in statistical_ds.items():
            probabilities[class_value] = statistical_ds[class_value][0][2] / float(
                total_rows
            )
            for i in range(len(class_summaries)):
                mean, standard_deviation, _ = class_summaries[i]
                probabilities[class_value] *= self.calculate_probability(
                    row[i], mean, standard_deviation
                )
        return probabilities

    # Predict the class for a given row
    def predict(self, row):
        summaries = self.summarize_by_class()
        probabilities = self.calculate_class_probabilities(summaries, row)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label


model = NaiveBayes()
# print(model.summarize_by_class())
row = [5.0, 3.4, 1.5, 0.2]
label = model.predict(row)
print("Data=%s, Predicted: %s" % (row, label))
