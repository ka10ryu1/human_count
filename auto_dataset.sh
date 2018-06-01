#!/bin/bash
# auto_train.sh
#
# データセットを自動でまとめて生成する
# ※一つのフォルダに数万のデータを入れるとブラウザで見るのに不便なので分割している

echo -e "\n<< label 00 >>\n"
./create_dataset.py -hn 0 -in 4000 -o result/0/a
./create_dataset.py -hn 0 -in 4000 -o result/0/b
./create_dataset.py -hn 0 -in 4000 -o result/0/c
./create_dataset.py -hn 0 -in 4000 -o result/0/d
./create_dataset.py -hn 0 -in 4000 -o result/0/e
./create_dataset.py -hn 0 -in 4000 -o result/0/f
./create_dataset.py -hn 0 -in 4000 -o result/0/g
./create_dataset.py -hn 0 -in 4000 -o result/0/h

echo -e "\n<< label 01 >>\n"
./create_dataset.py -hn 1 -in 4000 -o result/1/a
./create_dataset.py -hn 1 -in 4000 -o result/1/b
./create_dataset.py -hn 1 -in 4000 -o result/1/c
./create_dataset.py -hn 1 -in 4000 -o result/1/d
./create_dataset.py -hn 1 -in 4000 -o result/1/e
./create_dataset.py -hn 1 -in 4000 -o result/1/f
./create_dataset.py -hn 1 -in 4000 -o result/1/g
./create_dataset.py -hn 1 -in 4000 -o result/1/h

echo -e "\n<< label 02 >>\n"
./create_dataset.py -hn 2 -in 4000 -o result/2/a
./create_dataset.py -hn 2 -in 4000 -o result/2/b
./create_dataset.py -hn 2 -in 4000 -o result/2/c
./create_dataset.py -hn 2 -in 4000 -o result/2/d
./create_dataset.py -hn 2 -in 4000 -o result/2/e
./create_dataset.py -hn 2 -in 4000 -o result/2/f
./create_dataset.py -hn 2 -in 4000 -o result/2/g
./create_dataset.py -hn 2 -in 4000 -o result/2/h

echo -e "\n<< label 03 >>\n"
./create_dataset.py -hn 3 -in 4000 -o result/3/a
./create_dataset.py -hn 3 -in 4000 -o result/3/b
./create_dataset.py -hn 3 -in 4000 -o result/3/c
./create_dataset.py -hn 3 -in 4000 -o result/3/d
./create_dataset.py -hn 3 -in 4000 -o result/3/e
./create_dataset.py -hn 3 -in 4000 -o result/3/f
./create_dataset.py -hn 3 -in 4000 -o result/3/g
./create_dataset.py -hn 3 -in 4000 -o result/3/h

echo -e "\n<< label 04 >>\n"
./create_dataset.py -hn 4 -in 4000 -o result/4/a
./create_dataset.py -hn 4 -in 4000 -o result/4/b
./create_dataset.py -hn 4 -in 4000 -o result/4/c
./create_dataset.py -hn 4 -in 4000 -o result/4/d
./create_dataset.py -hn 4 -in 4000 -o result/4/e
./create_dataset.py -hn 4 -in 4000 -o result/4/f
./create_dataset.py -hn 4 -in 4000 -o result/4/g
./create_dataset.py -hn 4 -in 4000 -o result/4/h

# echo -e "\n<< label 05 >>\n"
# ./create_dataset.py -hn 5 -in 4000 -o result/5/0
# ./create_dataset.py -hn 5 -in 4000 -o result/5/1
# ./create_dataset.py -hn 5 -in 4000 -o result/5/2
# ./create_dataset.py -hn 5 -in 4000 -o result/5/3
# ./create_dataset.py -hn 5 -in 4000 -o result/5/4
