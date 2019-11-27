############################
#     Mateusz Oskroba      #
#         R00152957        #
############################

import pandas as pd


def task1():
    data = pd.read_csv("diamonds.csv")  # Load the diamonds file
    print(data.head())
    relevant_data = data.drop(columns=['Unnamed: 0', 'x', 'y', 'z'])
    print(relevant_data.head())
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
    print(category_grade_combinations_over_800)

    datasets_over_800 = []
    print(type(category_grade_combinations_over_800))
    for index, row in category_grade_combinations_over_800.iterrows():
        datasets_over_800.append(relevant_data.loc[(data['cut'] == row['cut']) & (relevant_data['color'] == row['color']) & (relevant_data['clarity'] == row['clarity'])])

    print(datasets_over_800)

    return datasets_over_800
    # 280 unqie combinations/ categories
    # (cut, color, clarity) : [ (carat, depth, table) : price, (carat, depth, table) : price)

    # determine value from its shape (depth and table), weight (carat)


def main():
    datasets_over_800 = task1()


main()
