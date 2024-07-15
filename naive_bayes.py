import numpy as np
import pandas as pd


# Load the file
def load(filename, include_demographics=True):
    df = pd.read_csv(f"{filename}")
    if not include_demographics:
       df = df.drop(columns=["Demographic"])
    return df


# Computes P(X = 1 | Y = 1) and P(X = 1 | Y = 0), where X is a single feature (column).
# x_column: name of the column containing the feature X.
# y_column: name of the class containing the class label.
# return: [P(X = 1 | Y = 1), P(X = 1 | Y = 0)]
# Applies Laplace Smoothing
def get_p_x_given_y(x_column, y_column, df):
    x_1_given_y_0 = 0
    x_1_given_y_1 = 0
    count_y_0 = 0
    count_y_1 = 0
    for index, row in df.iterrows():
        if row[y_column] == 1: 
            count_y_1 += 1
            if row[x_column] == 1: 
                x_1_given_y_1 += 1
        else: 
            count_y_0 += 1
            if row[x_column] == 1: 
                x_1_given_y_0 += 1
    p_0 = (x_1_given_y_0 + 1)/ (count_y_0 + 2)
    p_1 = (x_1_given_y_1 + 1) / (count_y_1 + 2)
    return [p_0, p_1]


# Store P(X_i=1 | Y=y) in p_x_given_y[i][y]
def get_all_p_x_given_y(y_column, df):
    all_p_x_given_y = np.zeros((df.shape[1]-1, 2))
    for i, col in enumerate(df.columns): 
        if (col != "Label"):  # Skips the "Label" column
            all_p_x_given_y[i][0] = (get_p_x_given_y(col, y_column, df))[0]
            all_p_x_given_y[i][1] = (get_p_x_given_y(col, y_column, df))[1]
    return all_p_x_given_y


# Compute P(Y = 1)
def get_p_y(y_column, df):
    # number of times y = 1 divided by all rows
    p_y = (len(df[df[y_column] == 1])) / (len(df))
    return p_y


# Computes the joint probability of a single row and y 
# P(X, Y) = P(Y) * P(X_1 | Y) * P(X_2 | Y) * ... * P(X_n | Y)
def joint_prob(xs, y, all_p_x_given_y, p_y):
    prob = 1 
    if y == 0:
        prob = 1 - p_y
    else: 
        prob = p_y

    for i in range(len(xs)):
        if xs[i] == 0: 
            prob *= (1 - all_p_x_given_y[i][y])
        else: 
            prob *= all_p_x_given_y[i][y] 
    return prob


# Computes the probability y given a single row.
def get_prob_y_given_x(y, xs, all_p_x_given_y, p_y):
    n, _ = all_p_x_given_y.shape  # n stores number of features/columns
    joint = joint_prob(xs, y, all_p_x_given_y, p_y)  # Applies joint probability function.
    zero = (joint_prob(xs, 0, all_p_x_given_y, p_y))
    one = (joint_prob(xs, 1, all_p_x_given_y, p_y))
    prob_y_given_x = joint/(zero + one)
    return prob_y_given_x


# Computes accuracy by comparing prediction to true values
def compute_accuracy(all_p_x_given_y, p_y, df):
    # split the test set into X and y. The predictions should not refer to test y's.
    X_test = df.drop(columns="Label")  
    y_test = df["Label"]
    num_correct = 0
    total = len(y_test)

    # Predict 1 if P(Y=1|X) >= 0.5
    counter = 0
    for i, xs in X_test.iterrows():
        p_1 = get_prob_y_given_x(1, xs.values, all_p_x_given_y, p_y)
        if (p_1 >= 0.5): 
            counter = 1
        else: 
            counter = 0
        if counter == y_test.iloc[i]: # If prediction was correct
            num_correct += 1
    accuracy = num_correct / total
    return accuracy


def main():
    # load the training set
    df_train = load("netflix-train.csv", False)

    # compute model parameters (i.e. P(Y), P(X_i|Y))
    all_p_x_given_y = get_all_p_x_given_y("Label", df_train)
    p_y = get_p_y("Label", df_train)

    # load the test set
    df_test = load("netflix-test.csv", False)
    print(all_p_x_given_y)

    print(f"Training accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_train)}")
    print(f"Test accuracy: {compute_accuracy(all_p_x_given_y, p_y, df_test)}")

    for i in range(len(all_p_x_given_y)):
        num = all_p_x_given_y[i][1]
        denom = (1 - all_p_x_given_y[i][1])
        p_x_1 = df_train.iloc[:, i].sum() / len(df_train)
        num /= p_x_1
        denom /= (1 - p_x_1)
        print(str(i) + " : " + str(num / denom))


if __name__ == "__main__":
    main()
