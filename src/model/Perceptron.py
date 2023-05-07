import numpy as np
from model.Model import Model

class Perceptron(Model):
    """パーセプトロンの分類器

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

    def __init__(self, eta=0.01, n_iter=50, random_state=1) -> None:
        super().__init__(eta, n_iter)
        self.random_state = random_state

    def fit(self, X, y) -> Model:
        """訓練を実行

        Args:
            X (np.ndarray): 説明変数
            y (np.ndarray): 目的変数
        """

        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])


        self.errors_ = []

        for _ in range(self.n_iter): # 訓練回数分まで訓練データ往復
            errors = 0
            for xi, target in zip(X, y):
                # 重み w1, ... wmの更新
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi

                # 重み w0の更新
                self.w_[0] += update

                # 重みの更新が0でない場合はご分類としてカウント
                errors += int(update != 0.0)

            # 反復ごとの誤差を格納
            self.errors_.append(errors)

        return self

    def net_input(self, X):
        """総入力を計算

        Args:
            X (np.ndarray):
        """

        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """ステップ後のクラスラベルを返す

        Args:
            X (np.ndarray): _description_
        """

        return np.where(self.net_input(X) >= 0.0, 1, -1)
