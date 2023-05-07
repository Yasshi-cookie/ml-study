import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix, heatmap
from model.LinearRegressionGD import LinearRegressionGD
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from Services.LinearRegressionService import LinearRegressionService

class Section10UseCase:
    def __init__(self) -> None:
        self.df = pd.read_csv('/Users/teruyayasushi/environment/ml-study/src/dataset/HousingDatasets/boston_house_prices.csv')
        self.cols = [
            'LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV',
        ]

    def show_r2_score(self) -> None:
        # 決定係数R^2を出力
        X = self.df.iloc[:, :-1].values
        # y = self.df['MEDV'].values
        y = 1000 * self.df['MEDV'].values

        # 学習用データ： X_train, y_train
        # 効果検証用データ： X_test, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        slr = LinearRegression()
        slr.fit(X_train, y_train)
        y_train_pred = slr.predict(X_train)
        y_test_pred = slr.predict(X_test)

        print('MSE train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))


    def show_mse(self) -> None:
        # MSE(平均２乗誤差)を出力
        X = self.df.iloc[:, :-1].values
        y = 1000 * self.df['MEDV'].values
        # y = self.df['MEDV'].values

        # 学習用データ： X_train, y_train
        # 効果検証用データ： X_test, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        slr = LinearRegression()
        slr.fit(X_train, y_train)
        y_train_pred = slr.predict(X_train)
        y_test_pred = slr.predict(X_test)

        print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_train_pred), mean_squared_error(y_test, y_test_pred)))

    def show_residual_plot(self) -> None:
        X = self.df.iloc[:, :-1].values
        y = self.df['MEDV'].values
        # 学習用データ： X_train, y_train
        # 効果検証用データ： X_test, y_test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        slr = LinearRegression()
        slr.fit(X_train, y_train)
        y_train_pred = slr.predict(X_train)
        y_test_pred = slr.predict(X_test)

        # 残差プロットを生成
        plt.scatter(y_train_pred, y_train_pred - y_train, c='steelblue', marker='o', edgecolor='white', label='Training data')
        plt.scatter(y_test_pred, y_test_pred - y_test, c='limegreen', marker='s', edgecolor='white', label='Test data')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.legend(loc='upper left')
        plt.hlines(y=0, xmin=-10, xmax=50, color='black', lw=2)
        plt.xlim([-10, 50])
        plt.tight_layout()
        plt.show()

    def show_predicted_result_by_ransac(self) -> None:
        X = self.df[['RM']].values
        y = self.df['MEDV'].values

        # RANSACRegressorのコンストラクタのパラメータの説明
        ## max_trials : イテレーションの最大数
        ## min_samples : ランダムに選択される訓練データの最小数
        ## loss : 損失関数のタイプ
        ## redisual_threshold : 正常値に含まれる範囲。学習曲線に対する縦の距離で指定
        ## random_status : 中心値を初期化するジェネレータ
        ## is_data_valid : 正常値か判断するために呼ばれる関数
        ## is_model_valid : 推定モデルと選択されたデータをパラメータにして呼ばれる関数。Falseが返されると選ばれたデータはスキップされる
        ## max_skips : イテレーションのうちスキップできる最大数
        ## stop_n_inliners : 最初に選ばれる正常値がこの数に達成するとデータ選択を終了する
        ## stop_score : スコアがこのしきい値を超えるとイテレーションを終了する
        ## stop_probability : 正常値の確率がこの値を超えるとデータ選択を終了する
        ransac = RANSACRegressor(
            LinearRegression(),
            max_trials=100,
            min_samples=50,
            loss='absolute_error',
            residual_threshold=5.0,
            random_state=0
        )

        ransac.fit(X, y)

        inlier_mask = ransac.inlier_mask_ # 正常値を表す真偽値を取得
        outlier_mask = np.logical_not(inlier_mask) # 外れ値を表す真偽値を取得
        line_X = np.arange(3, 10, 1) # 3から9までの整数値を作成
        line_y_ransac = ransac.predict(line_X[:, np.newaxis]) # 予測値を計算
        # # 正常値をプロット
        # plt.scatter(X[inlier_mask], y[inlier_mask], c='steelblue', edgecolor='white', marker='o', label='Inliers')
        # # 外れ値をプロット
        # plt.scatter(X[outlier_mask], y[outlier_mask], c='limegreen', edgecolor='white', marker='s', label='Outliers')
        # # 予測値をプロット
        # plt.plot(line_X, line_y_ransac, color='black', lw=2)
        # plt.xlabel('Average number of rooms [RM]')
        # plt.ylabel('Price in $1000s [MEDV]')
        # plt.legend(loc='upper left')
        # plt.show()

        # 「:.3f」は少数第３位までと言う意味を表す
        print('傾き: {:.3f}'.format(ransac.estimator_.coef_[0]))
        print('切片: {:.3f}'.format(ransac.estimator_.intercept_))

    def show_predicted_result_by_scikit_learn(self) -> None:
        X = self.df[['RM']].values
        y = self.df['MEDV'].values

        slr = LinearRegression()
        slr.fit(X, y)
        y_pred = slr.predict(X)
        print('傾き: {}'.format(slr.coef_[0]))
        print('切片: {}'.format(slr.intercept_))

        # LinearRegressionService(X, y, slr).lin_regplot()
        # plt.xlabel('Average number of rooms [RM]')
        # plt.ylabel('Price in $1000s [MEDV]')
        # plt.show()

    def show_predicted_result(self) -> None:
        X = self.df[['RM']].values
        y = self.df['MEDV'].values
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        X_std = sc_x.fit_transform(X)
        y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
        lr = LinearRegressionGD()
        lr.fit(X_std, y_std)

        LinearRegressionService(X_std, y_std, lr).lin_regplot()
        plt.xlabel('Average number of rooms [RM] (standardized)')
        plt.ylabel('Price in $1000s [MEDV] (standardized)')
        plt.show()

    def show_epoch_and_sse(self) -> None:
        X = self.df[['RM']].values
        y = self.df['MEDV'].values
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        X_std = sc_x.fit_transform(X)
        y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()
        lr = LinearRegressionGD()
        lr.fit(X_std, y_std)

        plt.plot(range(1, lr.n_iter + 1), lr.cost_)
        plt.ylabel('SSE')
        plt.xlabel('Epoch')
        plt.show()

    def show_correlation_cofficient_heatmap(self) -> None:
        cm = np.corrcoef(self.df[self.cols].values.T)
        hm = heatmap(cm, row_names=self.cols, column_names=self.cols)
        plt.show()

    def show_scatter_plot_matrix(self) -> None:
        scatterplotmatrix(self.df[self.cols].values, figsize=(10, 8), names=self.cols, alpha=0.5)
        plt.tight_layout()
        plt.show()

    def print_df(self) -> None:
        self.df.columns = [
            'CRIM', # 町ごとの人口1人当たりの犯罪発生率
            'ZN', # 25,000平方フィート以上の住宅区画の割合
            'INDUS', # 町ごとの非小売業の土地面積の割合
            'CHAS', # チャールズ川沿いのダミー変数（川沿いなら1、それ以外は0）
            'NOX', # 町ごとの一酸化窒素濃度（単位はパーツ・パー・10,000,000）
            'RM', # 住居の平均部屋数
            'AGE', # 1940年以前に建てられた持ち家の割合
            'DIS', # ボストンの主要な5つの雇用圏までの重み付き距離
            'RAD', # 高速道路へのアクセスしやすさの指数
            'TAX', # 10,000ドルあたりの所得税率
            'PTRATIO', # 町ごとの児童と教師の比率
            'B', # 町ごとの黒人居住者の割合（1000(Bk - 0.63)^2 として計算）
            'LSTAT', # 低所得者の割合
            'MEDV', # 住宅価格の中央値（単位は1,000ドル）
        ]

        print(self.df.head())
