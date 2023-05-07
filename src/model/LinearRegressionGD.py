import numpy as np
from model.Model import Model

class LinearRegressionGD(Model):
    """線形回帰モデル

    パラメータ
    ------------
    eta: float
        学習率（0.0 〜 1.0の値を取る）
    n_iter: int
        訓練データの訓練回数
    random_state: int
        重みを初期化するための乱数シード

    属性
    ------------
    w_: 1次元配列
        適合後の重み
    errors: リスト
        各エポックでの誤分類（更新） の数
    """

    def __init__(self, eta=0.001, n_iter=20) -> None:
        super().__init__(eta, n_iter)

    def fit(self, X: np.ndarray, y) -> object:
        """訓練を実行

        Args:
            X (object): 説明変数
            y (object): 目的変数
        """

        self.w_ = np.zeros(1 + X.shape[1]) # 重みを初期化
        self.cost_ = [] # コスト関数の値を初期化

        for _ in range(self.n_iter): # 訓練回数分まで訓練データ往復
            output = self.net_input(X) # 活性化関数の出力を計算
            # 誤差を計算
            ## erros = y - Xw
            errors = (y - output)
            # 重みw_{1}以降を更新
            ## 誤差関数の勾配ベクトル = (Xの転置行列) (Xw - y)
            ## X.T.dot(errors)は誤差関数の勾配ベクトルの-1倍に等しい
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum() # 重みw_{0}を更新
            cost = (errors**2).sum() / 2.0 # コスト関数を計算
            self.cost_.append(cost) # コスト関数の値を格納

        return self

    def net_input(self, X):
        """総入力を計算

        Args:
            X (object):
        """

        # np.dot(x, y)の返り値について
        # A_1, A_2を行列、b_1, b_2をベクトルとする時、
        # np.dot(A_1, A_2) = A_1 * A_2（行列同士の積）
        # np.dot(A_1, b_1) = A_1 * b（行列とベクトルとの積）
        # np.dot(b_1, A_1) = b_1 * A_1（ベクトルと行列との積）
        # np.dot(b_1, b_2) = b_1 * b_2（ベクトル同士の積）
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """予測値を計算する

        Args:
            X (object): _description_
        """

        return self.net_input(X)
