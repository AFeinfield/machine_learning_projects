import os
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive/')

class Data:
    def __init__(self, X=None, y=None):
        """
        Data class.
        
        Attributes
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        """

        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y

    def load(self, filename):
        """
        Load csv file into X array of features and y array of labels.
        
        Parameters
        --------------------
            filename -- string, filename
        """


        # load data
        with open(filename, "r") as fid:
            data = np.loadtxt(fid, delimiter=",")

        # separate features and labels
        self.X = data[:, :-1]
        self.y = data[:, -1]

    def plot(self, **kwargs):
        """Plot data."""

        if "color" not in kwargs:
            kwargs["color"] = "b"

        plt.scatter(self.X, self.y, **kwargs)
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)
        plt.show()

# wrapper functions around Data class
def load_data(filename):
    data = Data()
    data.load(filename)
    return data


def plot_data(X, y, **kwargs):
    data = Data(X, y)
    data.plot(**kwargs)

class PolynomialRegression:
    def __init__(self, m=1, reg_param=0):
        """
        Ordinary least squares regression.
        
        Attributes
        --------------------
            coef_   -- numpy array of shape (d,)
                       estimated coefficients for the linear regression problem
            m_      -- integer
                       order for polynomial regression
            lambda_ -- float
                       regularization parameter
        """

        # self.coef_ represents the weights of the regression model
        self.coef_ = None
        self.m_ = m
        self.lambda_ = reg_param

    def generate_polynomial_features(self, X):
        """
        Maps X to an mth degree feature vector e.g. [1, X, X^2, ..., X^m].
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,1), features
        
        Returns
        --------------------
            Phi     -- numpy array of shape (n,(m+1)), mapped features
        """

        n, d = X.shape


        add_feature = np.repeat(1, n)
        Phi = np.column_stack((add_feature, X))
        m = self.m_
        for i in range(2, m+1):
          column_to_add = np.power(X, i)
          Phi = np.column_stack((Phi, column_to_add))

        return Phi

    def fit_GD(self, X, y, eta=None, eps=0, tmax=10000, verbose=False):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using least squares batch gradient descent.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            eta     -- float, step size
            eps     -- float, convergence criterion
            tmax    -- integer, maximum number of iterations
            verbose -- boolean, for debugging purposes
        
        Returns
        --------------------
            self    -- an instance of self
        """
        if self.lambda_ != 0:
            raise Exception("GD with regularization not implemented")

        X = self.generate_polynomial_features(X)  # map features
        n, d = X.shape
        eta_input = eta
        self.coef_ = np.zeros(d)  # coefficients
        err_list = np.zeros((tmax, 1))  # errors per iteration

        # GD loop
        for t in range(tmax):
            if eta_input is None:
                eta = 1 / (1 + t)
            else:
                eta = eta_input

            prior_pred = np.dot(X, self.coef_)
            for j in range(d):
              sum = 0
              for i in range(n):
                sum += (prior_pred[i] - y[i]) * X[i][j]
              self.coef_[j] = self.coef_[j] - 2 * eta * sum

            y_pred = np.dot(X, self.coef_)
            err_list[t] = np.sum(np.power(y - y_pred, 2)) / float(n)

            if t > 0 and abs(err_list[t] - err_list[t - 1]) <= eps:
                break

            if verbose :
                x = np.reshape(X[:,1], (n,1))
                cost = self.cost(x,y)
                print ("iteration: %d, cost: %f" % (t+1, cost))

        print("number of iterations: %d" % (t + 1))

        return self

    def fit(self, X, y, l2regularize=None):
        """
        Finds the coefficients of a {d-1}^th degree polynomial
        that fits the data using the closed form solution.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
            l2regularize    -- set to None for no regularization. set to positive double for L2 regularization
                
        Returns
        --------------------        
            self    -- an instance of self
        """

        X = self.generate_polynomial_features(X)  # map features

        inv = np.linalg.pinv(np.dot(X.transpose(), X))
        temp = np.dot(inv, X.transpose())
        self.coef_ = np.dot(temp, y)

    def predict(self, X):
        """
        Predict output for X.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
        
        Returns
        --------------------
            y       -- numpy array of shape (n,), predictions
        """
        if self.coef_ is None:
            raise Exception("Model not initialized. Perform a fit first.")

        X = self.generate_polynomial_features(X)  # map features

        y = np.dot(X, self.coef_)

        return y

    def cost(self, X, y):
        """
        Calculates the objective function.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            cost    -- float, objective J(w)
        """

        prediction = self.predict(X)
        n, d = X.shape
        cost = 0
        for i in range(n):
          cost += (prediction[i] - y[i])**2
        return cost

    def rms_error(self, X, y):
        """
        Calculates the root mean square error.
        
        Parameters
        --------------------
            X       -- numpy array of shape (n,d), features
            y       -- numpy array of shape (n,), targets
        
        Returns
        --------------------
            error   -- float, RMSE
        """
        n, d = X.shape
        error = np.sqrt(self.cost(X, y) / n)
        return error

    def plot_regression(self, xmin=0, xmax=1, n=50, **kwargs):
        """Plot regression line."""
        if "color" not in kwargs:
            kwargs["color"] = "r"
        if "linestyle" not in kwargs:
            kwargs["linestyle"] = "-"

        X = np.reshape(np.linspace(0, 1, n), (n, 1))
        y = self.predict(X)
        plot_data(X, y, **kwargs)
        plt.show()


def main():
    # load data with correct file path
    data_directory_path =  "/content/drive/My Drive/ComSciM146/data"

    train_data = load_data(os.path.join(data_directory_path, "train.csv"))
    test_data = load_data(os.path.join(data_directory_path,"test.csv"))


    print("Visualizing data...")
    train_data.plot()
    test_data.plot()

    print("Investigating linear regression...")
    '''model = PolynomialRegression(1)
    model.coef_ = np.zeros(2)
    c = model.cost (train_data.X, train_data.y)
    print(f'model_cost:{c}')
    '''
    model = PolynomialRegression(1)
    model.fit_GD(train_data.X, train_data.y, eps=0, tmax=10000, verbose=True)
    print(model.coef_)

    model.fit(train_data.X, train_data.y)
    print(model.coef_)
    print(model.cost(train_data.X, train_data.y))

    print("Investigating polynomial regression...")
    degrees = []
    train_errs = []
    test_errs = []
    for m in range(11):
      model = PolynomialRegression(m)
      model.fit(train_data.X, train_data.y)
      degrees.append(m)
      train_errs.append(model.rms_error(train_data.X, train_data.y))
      test_errs.append(model.rms_error(test_data.X, test_data.y))
      print(m, ": ", model.rms_error(test_data.X, test_data.y))

    plt.plot(degrees, train_errs, 'r-', label='RMS Train Errors')
    plt.plot(degrees, test_errs, 'b-', label='RMS Test Errors')
    plt.legend()
    plt.show()
    print("Done!")

if __name__ == "__main__":
    main()