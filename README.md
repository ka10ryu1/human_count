# 概要

画像中の「人」だけを数える。

# 学習結果

# 動作環境

- **Ubuntu** 16.04.4 LTS ($ cat /etc/issue)
- **Python** 3.5.2 ($ python3 -V)
- **chainer** 4.0.0 ($ pip3 show chainer | grep Ver)
- **numpy** 1.14.2 ($ pip3 show numpy | grep Ver)
- **cupy** 4.0.0 ($ pip3 show cupy | grep Ver)
- **opencv-python** 3.4.0.12 ($ pip3 show opencv-python | grep Ver)

# ファイル構成

## 生成方法

```console
$ ls `find ./ -maxdepth 3 -type f -print` | xargs grep 'help = ' --include=*.py >& log.txt
$ tree >& log.txt
```

## ファイル

```console
├── Image
│   ├── background > 背景画像フォルダ
│   ├── other      > 物体画像フォルダ
│   └── people     > 人物画像フォルダ
├── LICENSE
├── Lib
│   ├── network.py
│   ├── plot_report_log.py
│   ├── read_dataset_CV2.py
│   └── read_dataset_PIL.py
├── Model
│   ├── ResNet-50-model.caffemodel > RESNET事前学習済みモデル
│   ├── demo.model                 > デモ用モデル
│   └── param.json                 > デモ用パラメータファイル
├── README.md
├── Tools > 説明省略
├── auto_dataset.sh
├── clean_all.sh
├── create_dataset.py
├── predict.py
├── summary.py
├── train.py
└── write_dataset.py
```

# チュートリアル

チュートリアルも学習も、どちらも初めて実行する場合はResnetのモデルを任意の場所にコピーしておく必要がある。

```console
$ cp ./Model/ResNet-50-model.caffemodel /home/[自分のディレクトリ]/.chainer/dataset/pfnet/chainer/models/
```

以下を実行するとデモが動く。


```console
$ ./predict.py Model/demo.model Model/param.json 
```

# 学習させる

## データを生成する

以下を入力し、データを生成する。
データの生成には時間がかかるので注意すること（HPCで50分くらい）。

```console
$ ./auto_dataset.sh
```

生成された画像は以下のように確認できる。

## データをテキストに出力する

生成したデータのパスとラベルをChainerで読み込めるテキスト形式に出力する。
他に分類させたいデータがある場合は、ここから始めればいい。

```console
$ ./write_dataset.py result/
```

trainとtestが生成されているのを以下のコマンドで確認する。

```console
$ wc result/t*
  16000   32000  512000 result/test_005.txt
 144000  288000 4608000 result/train_005.txt
 160000  320000 5120000 合計
```

ファイルの中身を一部覗くと以下のようになっている。
1列目がパス、2列目がラベルである。

```console
$ tail ./result/train_005.txt 
./result/4/f/2ecb2d7c3bf3.jpg 4
./result/4/b/dc90967bed8c.jpg 4
./result/1/f/dd3075e7d3ad.jpg 1
./result/2/e/f109a8c3b4be.jpg 2
./result/1/g/898acd576815.jpg 1
./result/2/b/0a8429ce56ec.jpg 2
./result/3/b/3bbd2c7b847a.jpg 3
./result/4/h/6f2ed82d8947.jpg 4
./result/1/h/3ab717504557.jpg 1
./result/2/f/06e8c53f8cee.jpg 2
```

## 学習実行

以下のコマンドを実行することで学習を実行できる。
以下のコマンドはHPCでGPUを使用する場合を想定している。
HPCで約3時間程度で学習が終了する。

```console
$ ./train.py -g 0 --no
```

## モデルの評価

以下のコマンドを実行することで、モデルを評価できる。
具体的には適合率と再現率とF値を計算する。

```console
$ ./summary.py Model/demo.model Model/param.json -g 0

〜省略〜

exec time: 1.04[s]
t: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4]
y: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3
 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4]
num|precision|recall|F
  0|    1.000| 1.000| 1.000
  1|    1.000| 1.000| 1.000
  2|    1.000| 1.000| 1.000
  3|    1.000| 1.000| 1.000
  4|    1.000| 1.000| 1.000
```

