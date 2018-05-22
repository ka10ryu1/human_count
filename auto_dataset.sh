#!/bin/bash
# auto_train.sh
# train.pyをいろいろな条件で試したい時のスクリプト
# train.pyの引数を手入力するため、ミスが発生しやすい。
# auto_train.shを修正したら、一度-cオプションを実行してミスがないか確認するべき

# オプション引数を判定する部分（変更しない）

usage_exit() {
    echo "Usage: $0 [-c]" 1>&2
    echo " -c: 設定が正常に動作するか確認する"
    exit 1
}

FLAG_CHK=""
while getopts ch OPT
do
    case $OPT in
        c)  FLAG_CHK="--only_check"
            ;;
        h)  usage_exit
            ;;
        \?) usage_exit
            ;;
    esac
done

shift $((OPTIND - 1))

# 以下自由に変更する部分（オプション引数を反映させるなら、$FLG_CHKは必要）

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
