# PythonでAutoEncoder

## 環境
Python 3.5.1  
OSX 10.10.5 Yosemite

## ディレクトリ構成
- **data/**  
データセットを入れとくとこ  
  - データセットファイルの内容
    ```
    N D
    data1_1 data1_2 data1_3 ...
    data2_1 data2_2 data2_3 ...
    ...
    ```
    N データ数  
    D データの次元数  
    1データは一行に，要素はスペース区切り
- **results/**  
計算されたパラメータが入るとこ

## 実行
`$ python 'プログラム名'`

## 何があるか
中間層1つのAutoEncoder．活性化関数は中間層，出力層ともに恒等関数
- **auto_encoder_kadai1-4.py**  
numpyなしで作成したAutoEncoder

- **autoencoder.py**  
numpy使って作成したAutoEncoder

- **autoencoder_adagrad.py**  
autoencoder.pyにadagradをプラス

- **autoencoder_sigmoid.py**  
autoencoder.pyの中間層の活性化関数をsigmoidにしたやつ

- **autoencoder_minibatch.py**  
autoencoder.pyの入力をミニバッチ化

- **autoencoder_all.py**  
上3つを全部入れたやつ

- **data.py**
データファイルとのやりとりとか
