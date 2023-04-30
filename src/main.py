import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from model.Perceptron import Perceptron

def main() -> None:
    df = pd.read_csv('/Users/teruyayasushi/environment/ml-study/src/dataset/iris.csv', header=None)
    # 1-100行目の目的変数の抽出
    y = df.iloc[0:100, 4].values
    # Iris-setosaを-1、Iris-versicolorを1に変換
    y = np.where(y == 'Iris-setosa', -1, 1)
    # 1-100行目の1、3列目の抽出
    X = df.iloc[0:100, [0, 2]].values

    # plot_dataset(X)
    plot_errors_and_epochs(X, y)

def plot_dataset(X) -> None:

    # 品種setosaのプロット（赤の⚪︎）
    plt.scatter(X[:50,0], X[:50,1], color='red', marker='o', label='setosa')
    # 品種versicolorのプロット（青の×）
    plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x', label='versicolor')
    # 軸のラベルの設定
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    # 判例の設定（左上に配置）
    plt.legend(loc='upper left')
    # 図の表示
    plt.show()

def plot_errors_and_epochs(X, y) -> None:
    # パーセプトロンのオブジェ狗tの生成
    ppn = Perceptron(eta=0.1, n_iter=10)
    # 訓練データへのモデルの適合
    ppn.fit(X, y)
    # エポックと誤分類の関係を表す折れ線グラフをプロット
    plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
    # 軸のラベルの設定
    plt.xlabel('Epochs')
    plt.ylabel('Number of update')
    # 図の表示
    plt.show()

main()
