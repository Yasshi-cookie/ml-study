import matplotlib.pyplot as plt
import numpy as np

class LinearRegressionService:
    def __init__(self, X: np.ndarray, y: np.ndarray, model) -> None:
        self.X = X
        self.y = y
        self.model = model

    def lin_regplot(self) -> None:
        plt.scatter(self.X, self.y, c='steelblue', edgecolor='white', s=70)
        plt.plot(self.X, self.model.predict(self.X), color='black', lw=2)
