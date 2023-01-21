from padasip.filters.base_filter import AdaptiveFilter
# Modified code from https://matousc89.github.io/padasip/_modules/padasip/filters/rls.html
# FilterRLS.adapt
import numpy as np


class FilterRLS(AdaptiveFilter):
    """
    Adaptive RLS filter.

    **Args:**

    * `n` : length of filter (integer) - how many input is input array
      (row of input matrix)

    **Kwargs:**

    * `mu` : forgetting factor (float). It is introduced to give exponentially
      less weight to older error samples. It is usually chosen
      between 0.98 and 1.

    * `eps` : initialisation value (float). It is usually chosen
      between 0.1 and 1.

    * `w` : initial weights of filter. Possible values are:

        * array with initial weights (1 dimensional array) of filter size

        * "random" : create random weights

        * "zeros" : create zero value weights
    """

    def __init__(self, n, mu=0.98, eps=0.35, w="zeros"):
        self.kind = "RLS filter"
        if type(n) == int:
            self.n = n
        else:
            raise ValueError('The size of filter must be an integer')
        self.mu = mu
        self.eps = eps
        self.w = self.init_weights(w, self.n)
        self.R = 1 / self.eps * np.identity(n)
        self.w_history = False

    def adapt(self, d, x):
        """
        Adapt weights according one desired value and its input.

        **Args:**

        * `d` : desired value (float)

        * `x` : input array (1-dimensional array)
        """
        y = np.dot(self.w, x)
        e = d - y
        R1 = np.dot(np.dot(np.dot(self.R, x), x.T), self.R)
        R2 = self.mu + np.dot(np.dot(x, self.R), x.T)
        self.R = 1 / self.mu * (self.R - R1 / R2)
        dw = np.dot(self.R, x.T) * e
        self.w += dw

    def run(self, d, x):
        """
        This function filters multiple samples in a row.

        **Args:**

        * `d` : desired value (1 dimensional array)

        * `x` : input matrix (2-dimensional array). Rows are samples,
            columns are input arrays.

        **Returns:**

        * `y` : output value (1 dimensional array).
          The size corresponds with the desired value.

        * `e` : filter error for every sample (1 dimensional array).
          The size corresponds with the desired value.

        * `w` : history of all weights (2 dimensional array).
          Every row is set of the weights for given sample.
        """
        # measure the data and check if the dimension agree
        N = len(x)
        if not len(d) == N:
            raise ValueError('The length of vector d and matrix x must agree.')
        self.n = len(x[0])
        # prepare data
        try:
            x = np.array(x)
            d = np.array(d)
        except (Exception,):
            raise ValueError('Impossible to convert x or d to a numpy array')
        # create empty arrays
        y = np.zeros(N)
        e = np.zeros(N)
        self.w_history = np.zeros((N, self.n))
        # adaptation loop
        for k in range(N):
            self.w_history[k, :] = self.w
            y[k] = np.dot(self.w, x[k])
            e[k] = d[k] - y[k]
            R1 = np.dot(np.dot(np.dot(self.R, x[k]), x[k].T), self.R)
            R2 = self.mu + np.dot(np.dot(x[k], self.R), x[k].T)
            self.R = 1 / self.mu * (self.R - R1 / R2)
            dw = np.dot(self.R, x[k].T) * e[k]
            self.w += dw
        return y, e, self.w_history

    def predict(self, x):
        """
        This function predicts multiple samples in a row.
        **Args:**
        * `x` : input matrix (2-dimensional array). Rows are samples,
            columns are input arrays.
        **Returns:**
        * `y` : output value (1 dimensional array).
          The size corresponds with the desired value.
        """
        N = len(x)
        try:
            x = np.array(x)
        except (Exception,):
            raise ValueError("Can't to convert x or d to a numpy array")
        # create empty array
        y = np.zeros(N)
        # prediction loop
        for k in range(N):
            y[k] = np.dot(self.w, x[k])
        return y
