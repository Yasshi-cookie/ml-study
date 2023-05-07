from abc import abstractmethod

class Model(object):
    def __init__(self, eta: float, n_iter: int) -> None:
        self.eta = eta
        self.n_iter = n_iter

    @abstractmethod
    def fit(self, X, y) -> object:
        pass

    @abstractmethod
    def predict(self, X):
        pass
