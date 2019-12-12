############################
#     Mateusz Oskroba      #
#         R00152957        #
############################

import pandas as pd
import numpy as np
from sklearn import model_selection


##################
#     Task 1     #
##################
def task1():
    data = pd.read_csv("diamonds.csv")  # Load the diamonds file
    relevant_data = data.drop(columns=['Unnamed: 0', 'x', 'y', 'z'])
    # cut_quality = data['cut']
    # color_grades = data['color']
    # clarity_grades = data['clarity']
    #
    # # Features:
    # carat = data['carat']
    # depth = data['depth']
    # table_value = data['table']
    #
    # # Target:
    # selling_price = data['price']
    #
    # # A data point is a discrete unit of information. In a general sense, any single fact is a data point.
    # # In a statistical or analytical context, a data point is usually derived from a measurement or research
    # # and can be represented numerically and/or graphically
    # #
    # # So... "For each combination of these cut, colour and clarity grades extract the corresponding data-points"
    # # now, ('ideal', 'e', 'vs2') : [(carat, depth, table_value), (carat, depth, table_value), (carat, depth, table_value)]
    #
    # # Going grade-by-grade split the data-points into features [1 point] and targets [1 point]
    # # What is grade by grade? Is grade a combination of quality, color and clarity?
    # #      * Yes, "... to the various different grades (e.g. ('Ideal', 'E', 'VS2'))"...
    # # So...
    #
    # # quality_combination = map(lambda : cut, color, clarity : [cut, color, clarity], cut_quality, color_grades, clarity_grades))

    # extracts what types of cut qualities [1 point], colour grades [1 point], and clarity grades [1 point] are represented
    category_grade_combinations = relevant_data[['cut', 'color', 'clarity']]

    category_grade_combinations_frequency_count = (category_grade_combinations.groupby(category_grade_combinations.columns.tolist()).size().reset_index().rename(columns={0: 'count'}))

    category_grade_combinations_over_800 = category_grade_combinations_frequency_count[category_grade_combinations_frequency_count["count"] > 800]
    print("category_grade_combinations_over_800", category_grade_combinations_over_800)

    datasets_over_800 = []
    for index, row in category_grade_combinations_over_800.iterrows():
        datasets_over_800.append(relevant_data.loc[(data['cut'] == row['cut']) & (relevant_data['color'] == row['color']) & (relevant_data['clarity'] == row['clarity'])])

    print("datasets_over_800", datasets_over_800)

    return datasets_over_800
    # 280 unqie combinations/ categories
    # (cut, color, clarity) : [ (carat, depth, table) : price, (carat, depth, table) : price)

    # determine value from its shape (depth and table), weight (carat)


##################
#     Task 2     #
##################
def num_coefficients_3(d):
    t = 0
    for n in range(d + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if i + j + k == n:
                        t = t + 1
    return t


def calculate_model_function(degree_of_polynomial, list_of_feature_vectors, parameter_vector_of_coefficients):
    r = np.zeros(list_of_feature_vectors.shape[0])
    t = 0
    for n in range(degree_of_polynomial + 1):
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    if i + j + k == n:
                        r += parameter_vector_of_coefficients[k] * (list_of_feature_vectors[:, 0] ** i) * (list_of_feature_vectors[:, 1] ** (n - i))
                        t = t + 1
    return r


##################
#     Task 3     #
##################
def calculate_modal_function_and_jacobian(degree_of_polynomial, feature_vectors, coefficient_of_linerization_point):
    # TODO: calculate and return:
    #  the estimated target vector
    #  Jacobian at the linerisation point

    return 0


def linearize(deg, data, p0):
    f0 = calculate_model_function(deg, data, p0)
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg, data, p0)
        p0[i] -= epsilon
        di = (fi - f0) / epsilon
        J[:, i] = di
    return f0, J


##################
#     Task 4     #
##################
def calculate_update(y, f0, J):
    l = 1e-2
    N = np.matmul(J.T, J) + l * np.eye(J.shape[1])
    r = y - f0
    n = np.matmul(J.T, r)
    dp = np.linalg.solve(N, n)
    return dp


##################
#     Task 5     #
##################
def regression(degree_of_polynomial, training_data_features, training_data_targets):
    max_iter = 10
    p0 = np.zeros(num_coefficients_3(degree_of_polynomial))
    for i in range(max_iter):
        f0, J = linearize(degree_of_polynomial, training_data_features, p0)
        dp = calculate_update(training_data_targets, f0, J)
        p0 += dp

    return p0


# degree of the polynomial, the training data features and the training data targets
# def task5(degree_of_polynomial, training_data_features, training_data_targets):
#     parameter_vector_of_coefficients = np.zeros(dataset.shape[0])


def main():
    datasets_over_800 = task1()

    number_of_kfolds = 5

    kf = model_selection.KFold(n_splits=number_of_kfolds, shuffle=True)

    current_dataset = 0
    for dataset in datasets_over_800:
        current_dataset += 1
        for degree in [0, 1, 2, 3]:
            current_fold = 0
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
                actual_testing_price = testing_sub_set_targets  # I know it's literally line 165 above, but it'll be easier to follow it for now

                absolute_difference = np.absolute(predicted_testing_price.astype('float64'), actual_testing_price.astype('float64'))
                print(absolute_difference)

                #
                # print("Hi5")
                # stuff = regression(degree, training_sub_set_features, training_sub_set_targets)
                # print("Hi6")
                #
                # print(stuff)


main()
