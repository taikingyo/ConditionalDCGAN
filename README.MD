# ConditionalDCGAN
tf.kerasでganの勉強のため、実装  
画像解像度に合わせて畳込み層の数を自動調整出来るようにしてみました。  
ラベル情報の付与方法って、正解はあるのでしょうか、Enbeddingしてサイズを揃えてAddしてます。
## Requirement
使用ライブラリ
* numpy
* matplotlib
* sklearn
* tensorflow

## Usage
```
import numpy as np
import tensorflow as tf
from gan import ConditionalDCGAN

(X_train, Y_train), (_, _) = tfk.datasets.cifar10.load_data()
x = X_train / 255 * 2 - 1
y = Y_train.flatten()
gan = ConditionalDCGAN(32, 32, 3, 10)
gan.train(x, y, epochs=20)
```
## Author
Fukuzawa Taiki (taikingyo@gmail.com)