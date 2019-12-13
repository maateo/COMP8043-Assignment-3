############################
#     Mateusz Oskroba      #
#         R00152957        #
############################

import pandas as pd
import numpy as np
from sklearn import model_selection
import matplotlib.pyplot as plt


##################
#     Task 1     #
##################
def pre_processing():
    data = pd.read_csv("diamonds.csv")  # Load the diamonds file
    relevant_data = data.drop(columns=['Unnamed: 0', 'x', 'y', 'z'])  # Remove unused columns

    # extracts what types of cut qualities [1 point], colour grades [1 point], and clarity grades [1 point] are represented
    category_grade_combinations = relevant_data[['cut', 'color', 'clarity']]

    # Get frequency of each combination
    category_grade_combinations_frequency_count = category_grade_combinations.groupby(category_grade_combinations.columns.tolist()) \
        .size() \
        .reset_index() \
        .rename(columns={0: 'count'})
    print("Frequencies of each category:\n", category_grade_combinations_frequency_count)

    category_grade_combinations_over_800 = category_grade_combinations_frequency_count[category_grade_combinations_frequency_count["count"] > 800]
    print("category_grade_combinations_over_800:\n", category_grade_combinations_over_800)

    datasets_over_800 = []
    for index, row in category_grade_combinations_over_800.iterrows():
        datasets_over_800.append(relevant_data.loc[(data['cut'] == row['cut']) & (relevant_data['color'] == row['color']) & (relevant_data['clarity'] == row['clarity'])])

    print("datasets_over_800:\n", datasets_over_800)

    return datasets_over_800


##################
#     Task 2     #
##################
def num_coefficients_3(degree):
    t = 0
    for n in range(degree + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if i + j + k == n:
                        t = t + 1
    return t


def calculate_model_function(degree_of_polynomial, list_of_feature_vectors, parameter_vector_of_coefficients):
    result = np.zeros(list_of_feature_vectors.shape[0])
    t = 0
    for n in range(degree_of_polynomial + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if i + j + k == n:
                        result += parameter_vector_of_coefficients[k] * (list_of_feature_vectors[:, 0] ** i) * (list_of_feature_vectors[:, 1] ** (n - i))
                        t = t + 1
    return result


##################
#     Task 3     #
##################
def linearize(degree_of_polynomial, feature_vectors, coefficients_of_linearization_point):
    estimated_target_vector = calculate_model_function(degree_of_polynomial, feature_vectors, coefficients_of_linearization_point)
    Jacobian = np.zeros((len(estimated_target_vector), len(coefficients_of_linearization_point)))
    epsilon = 1e-6
    for i in range(len(coefficients_of_linearization_point)):
        coefficients_of_linearization_point[i] += epsilon
        fi = calculate_model_function(degree_of_polynomial, feature_vectors, coefficients_of_linearization_point)
        coefficients_of_linearization_point[i] -= epsilon
        di = (fi - estimated_target_vector) / epsilon
        Jacobian[:, i] = di
    return estimated_target_vector, Jacobian


##################
#     Task 4     #
##################
def calculate_parameter_update(y, f0, Jacobian):
    l = 1e-2
    normal_equation_matrix = np.matmul(Jacobian.T, Jacobian) + l * np.eye(Jacobian.shape[1])
    residual = y - f0
    normal_equation_system = np.matmul(Jacobian.T, residual)
    optimal_parameter_update = np.linalg.solve(normal_equation_matrix, normal_equation_system)
    return optimal_parameter_update


##################
#     Task 5     #
##################
def regression(degree_of_polynomial, training_data_features, training_data_targets):
    max_iterations = 10
    p0 = np.zeros(num_coefficients_3(degree_of_polynomial))
    for i in range(max_iterations):
        f0, Jacobian = linearize(degree_of_polynomial, training_data_features, p0)
        dp = calculate_parameter_update(training_data_targets, f0, Jacobian)
        p0 += dp

    return p0


def main():
    datasets_over_800 = pre_processing()

    ##################
    #     Task 6     #
    ##################
    number_of_kfolds = 5

    kf = model_selection.KFold(n_splits=number_of_kfolds, shuffle=True)

    current_dataset = 0
    for dataset in datasets_over_800:
        current_dataset += 1

        absolute_difference_per_degree = []

        for degree in [0, 1, 2, 3]:
            current_fold = 0

            absolute_differences_for_all_folds = []
            for train_index, test_index in kf.split(dataset):
                current_fold += 1

                print("Current dataset: %d Current degree: %d Current fold: %d" % (current_dataset, degree, current_fold))

                training_sub_set = dataset.iloc[train_index]
                training_sub_set_features = np.array(training_sub_set[['carat', 'depth', 'table']])
                training_sub_set_targets = np.array(training_sub_set['price'])

                p0 = regression(degree, training_sub_set_features, training_sub_set_targets)

                testing_sub_set = dataset.iloc[test_index]
                testing_sub_set_features = np.array(testing_sub_set[['carat', 'depth', 'table']])
                testing_sub_set_targets = np.array(testing_sub_set['price'])

                predicted_testing_price = calculate_model_function(degree, testing_sub_set_features, p0)
                actual_testing_price = testing_sub_set_targets

                absolute_difference = np.absolute(predicted_testing_price.astype('float64') - actual_testing_price.astype('float64'))

                absolute_differences_for_all_folds = np.concatenate((absolute_differences_for_all_folds, absolute_difference), axis=0)

            absolute_difference_per_degree.append(np.mean(absolute_differences_for_all_folds))

        best_degree = absolute_difference_per_degree.index(min(absolute_difference_per_degree))
        print(absolute_difference_per_degree, "BEST DEGREE:", best_degree)

        ##################
        #     Task 7     #
        ##################
        dataset_features = np.array(dataset[['carat', 'depth', 'table']])
        dataset_targets = np.array(np.array(dataset['price']))

        p0 = regression(best_degree, dataset_features, dataset_targets)

        best_degree_predicted_price = calculate_model_function(best_degree, dataset_features, p0)
        actual_price = dataset_targets

        plt.scatter(best_degree_predicted_price, actual_price)
        plt.xlabel("Predicted Price")
        plt.ylabel("Actual Price")

        plt.plot([min(actual_price), max(actual_price)], [min(actual_price), max(actual_price)], color='red')

        plt.show()


main()
