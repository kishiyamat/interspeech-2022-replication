# README

ラベルと特徴量を要因として(3x3)を2パターン作り、
東京方言と近畿方言のモデルを作成する。
それぞれのモデルに音声を与えて予測を立て、
人のラベリングと特徴量の計算を考察する。

ラベル要因(RLEをベース)
- RLE
- RLE+デルタ

特徴量要因

- pitch
- pitch:delta
- delta

言語モデル
- Tokyo
- Kinki

実験

- testの音声(`L_H`のようなトーンを錯覚する音声)を与えてみて推論させる。

## 補足

### モデルにわたすパラメータ

HSMM自体は初期確率、遷移確率、射出確率、持続確率の
分布で形成されるので、それぞれが必要となる。
さらに射出確率はGaussianの場合と
MultivarieteGaussianの場合で必要なパラメータが異なるので、
[emmition.py](https://github.com/kishiyamat/hsmmlearn/blob/master/hsmmlearn/emissions.py)
を参照すると良い。
以下にそれぞれの解説を置く。


Gaussianの場合、
`GaussianEmissions`


MultivarieteGaussianの場合、
`MultivariateGaussianEmissions`
