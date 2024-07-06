# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression, and use cross entropy as loss function.
    def fit(self, X, y):
        num_samples, num_features = X.shape
        self.weights = np.zeros(num_features)
        self.intercept = 0
        # gradient descent
        for _ in range(self.iteration):
            linear_model = np.dot(X, self.weights) + self.intercept
            y_predicted = self.sigmoid(linear_model)

            # compute gradients
            dw = (1 / num_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / num_samples) * np.sum(y_predicted - y)

            # update parameters
            self.weights -= self.learning_rate * dw
            self.intercept -= self.learning_rate * db
                        
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.intercept
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None
        self.c0 = None
        self.c1 = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        N1, N2, sum0, sum1 = 0, 0, 0, 0
        self.c0 = X[y == 0]
        self.c1 = X[y == 1]
        for i in range(len(y_train)):
            if y_train[i] == 0: 
                # compute the amount of points in class 1
                N1 += 1
                # compute the sum of each data point in class 1
                sum0 += X[i]
            else: 
                # compute the amount of points in class 2
                N2 += 1
                # compute the sum of each data point in class 2
                sum1 += X[i]

        # compute m1 & m2
        self.m0 = sum0 / N1
        self.m1 = sum1 / N2
        ## Your code HERE
        # initialize sw
        self.sw = np.array([[0.0, 0.0], [0.0, 0.0]])
        # compute sw
        for i in range(len(y_train)):
            if y_train[i] == 0:
                tmp = X[i] - self.m0
                self.sw += np.outer(tmp, tmp)
            else:
                tmp = X[i] - self.m1
                self.sw += np.outer(tmp, tmp)
        ## Your code HERE
        tmp = self.m1 - self.m0
        # compute sb = (m2 - m1)(m2 - m1)^T
        self.sb = np.outer(tmp, tmp)
        ## Your code HERE
        # by differentiating the Fisher Criterion with respect to w
        # we can derive that w is in the same direction as sw^-1(m2 - m1)
        sw_inv = np.linalg.inv(self.sw)
        tmp = np.matmul(sw_inv, self.m1 - self.m0)
        # normalize w to unit length
        self.w = tmp / np.linalg.norm(tmp)

    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        # Project the input data onto w
        projected_X = np.dot(X, self.w)

        # Project the means of the two classes onto w
        projected_m0 = np.dot(self.m0, self.w)
        projected_m1 = np.dot(self.m1, self.w)

        # Compute the absolute distances from the projected X to the projected means
        dist_to_m0 = np.abs(projected_X - projected_m0)
        dist_to_m1 = np.abs(projected_X - projected_m1)

        # Predict the class label by comparing the distances
        y_pred = np.where(dist_to_m0 < dist_to_m1, 0, 1)

        return y_pred

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        import matplotlib.pyplot as plt
        m = self.w[1] / self.w[0]
        b = 30
        X = X = np.linspace(-50, 0, 10)
        Y = m * X + b
        c1_x = self.c0.T[0]
        c1_y = self.c0.T[1]
        c2_x = self.c1.T[0]
        c2_y = self.c1.T[1]
        p1_x = (m * c1_y + c1_x - m * b) / (m**2 + 1)
        p1_y = (m**2 * c1_y + m * c1_x + b) / (m**2 + 1)
        p2_x = (m * c2_y + c2_x - m * b) / (m**2 + 1)
        p2_y = (m**2 * c2_y + m * c2_x + b) / (m**2 + 1)
        plt.title(f'Projection Line: w={m}, b={b}')
        plt.plot(X, Y, c='blue', linewidth=0.8)
        plt.plot([c2_x, p2_x], [c2_y, p2_y], c='red', linewidth=0.02)
        plt.plot([c1_x, p1_x], [c1_y, p1_y], c='green', linewidth=0.02)
        plt.plot(c2_x, c2_y, '.', c='red', markersize=1)
        plt.plot(c1_x, c1_y, '.', c='green', markersize=1)
        plt.plot(p2_x, p2_y, '.', c='red', markersize=1)
        plt.plot(p1_x, p1_y, '.', c='green', markersize=1)
        plt.show()
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.0001, iteration=500000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"
    FLD.plot_projection(X_test)