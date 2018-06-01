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
│   ├── background
│   ├── other
│   └── people
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