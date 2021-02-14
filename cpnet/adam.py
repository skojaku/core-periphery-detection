import numpy as np
from scipy import sparse


class ADAM:
    def __init__(self):
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eta = 0.01

        self.t = 0
        self.mt = None
        self.vt = None
        self.eps = 1e-8

    def update(self, theta, grad, lasso_penalty, positiveConstraint=False):
        """Ascending."""
        if self.mt is None:
            self.mt = np.zeros(grad.shape)
            self.vt = np.zeros(grad.shape)

        self.t = self.t + 1

        self.mt = self.beta1 * self.mt + (1 - self.beta1) * grad
        self.vt = self.beta2 * self.vt + (1 - self.beta2) * np.multiply(grad, grad)

        mthat = self.mt / (1 - np.power(self.beta1, self.t))
        vthat = self.vt / (1 - np.power(self.beta2, self.t))

        new_grad = mthat / (np.sqrt(vthat) + self.eps)

        return self._prox(
            theta + self.eta * new_grad, lasso_penalty * self.eta, positiveConstraint
        )

    def _prox(self, x, lam, positiveConstraint):
        """Soft thresholding operator.

        Parameters
        ----------
        x : float
            Variable.
        lam : float
            Lasso penalty.

        Returns
        -------
        y : float
            Thresholded value of x.
        """
        if positiveConstraint:
            b = ((lam) > 0).astype(int)
            return np.multiply(b, np.maximum(x - lam, np.zeros(x.shape))) + np.multiply(
                1 - b,
                np.multiply(np.sign(x), np.maximum(np.abs(x) - lam, np.zeros(x.shape))),
            )
        else:
            return np.multiply(
                np.sign(x), np.maximum(np.abs(x) - lam, np.zeros(x.shape))
            )
