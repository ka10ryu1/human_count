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
./create_dataset.py -hn 0 -in 10000 -o result/00
echo -e "\n<< label 01 >>\n"
./create_dataset.py -hn 1 -in 10000 -o result/01
echo -e "\n<< label 02 >>\n"
./create_dataset.py -hn 2 -in 10000 -o result/02
echo -e "\n<< label 03 >>\n"
./create_dataset.py -hn 3 -in 10000 -o result/03
echo -e "\n<< label 04 >>\n"
./create_dataset.py -hn 4 -in 10000 -o result/04
echo -e "\n<< label 05 >>\n"
./create_dataset.py -hn 5 -in 10000 -o result/05
