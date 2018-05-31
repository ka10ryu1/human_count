#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '画像を読み込んでデータセットを作成する'
#

import logging
# basicConfig()は、 debug()やinfo()を最初に呼び出す"前"に呼び出すこと
logging.basicConfig(format='%(message)s')
level = logging.INFO
logging.getLogger('Tools').setLevel(level=level)

import os
import cv2
import time
import argparse
import numpy as np

import Tools.imgfunc as IMG
import Tools.getfunc as GET
import Tools.func as F


class Timer(object):
    def __init__(self):
        self._start = 0

    def reset(self):
        self._start = time.time()

    def stop(self):
        return time.time() - self._start

    def view(self, add=''):
        print(add+'{0:.3f}[s]'.format(self.stop()))


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('-ot', '--other_path', default='./Image/other/',
                        help='動物、怪獣の画像フォルダ (default: ./Image/other/')
    parser.add_argument('-hu', '--human_path', default='./Image/people/',
                        help='人間の画像フォルダ (default: ./Image/people/')
    parser.add_argument('-bg', '--background_path', default='./Image/background/',
                        help='背景の画像フォルダ (default: ./Image/background/')
    parser.add_argument('-os', '--obj_size', type=int, default=64,
                        help='挿入する画像サイズ [default: 64 pixel]')
    parser.add_argument('-is', '--img_size', type=int, default=256,
                        help='生成される画像サイズ [default: 256 pixel]')
    parser.add_argument('-in', '--img_num', type=int, default=200,
                        help='画像を生成する数 [default: 200]')
    parser.add_argument('-on', '--obj_num', type=int, default=3,
                        help='障害物の最大数 [default: 3]')
    parser.add_argument('-hn', '--human_num', type=int, default=2,
                        help='人間の最大数 [default: 2]')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='・ (default: ./result/)')
    args = parser.parse_args()
    F.argsPrint(args)
    return args


def getSomeImg(imgs, num, size, random=True):
    """
    画像リストから任意の数画像をランダムに取得し、大きさも揃える
    [in]  imgs:ランダムに取得したい画像リスト
    [in]  num: 取得する数（1以下の数値で1枚）
    [in]  size:画像サイズ [pixel]
    [out] 取得した画像リスト
    """

    label = range(len(imgs))
    # 複数枚取得する
    if(num > 1):
        if random:
            choice_num = np.random.randint(1, num)
        else:
            choice_num = num

        pickup = np.random.choice(label, choice_num, replace=False)
        # リサイズする
        if size > 1:
            return [IMG.resizeP(img, size) for img in imgs[pickup]]
        # リサイズしない
        else:
            imgs[pickup]

    # 一枚取得する
    else:
        pickup = np.random.choice(label, 1, replace=False)[0]
        # リサイズする
        if size > 1:
            return [IMG.resizeP(imgs[pickup], size)]
        # リサイズしない
        else:
            return [imgs[pickup]]


def rondom_crop(img, size):
    """
    画像をランダムに切り取る
    ※横長の画像に対して有効
    [in]  img:  切り取りたい画像
    [in]  size: 切り取るサイズ（正方形）
    [out] 切り取った画像
    """

    w, h = img.shape[: 2]
    # 短辺を取得
    short_side = min(img.shape[: 2])
    x = np.random.randint(0, w - short_side + 1)
    y = np.random.randint(0, h - short_side + 1)
    # リサイズする
    if size > 1:
        return IMG.resizeP(img[x: x + short_side, y: y + short_side], size)
    # リサイズしない
    else:
        return img[x: x + short_side, y: y + short_side]


def getImg(imgs, size):
    """
    画像リストから画像を一枚取得する
    [in]  imgs: 取得したい画像リスト
    [in]  size: 画像のサイズ
    [out] 取得したい画像
    """

    return getSomeImg(imgs, 1, size)


def getImgN(path):
    """
    入力されたフォルダにある画像を全て読み込む
    [in] path:
    [out] 読みこんだ画像リスト
    """

    if not os.path.isdir(path):
        print('path not found:', path)
        exit(1)

    from os.path import join as opj
    return np.array([cv2.imread(opj(path, f), IMG.getCh(0))
                     for f in os.listdir(path) if IMG.isImgPath(opj(path, f))])


def create(obj_path, h_path, bg_path,
           obj_size, img_size, obj_num, img_num, create_num):
    """
    前景（障害物、対象物）と背景をいい感じに重ね合わせてデータセットを作成する
    [in]  obj_path:   障害物の画像があるフォルダのパス
    [in]  h_path:     対象物の画像があるフォルダのパス
    [in]  bg_path:    背景の画像があるフォルダのパス
    [in]  obj_size:   障害物のサイズ
    [in]  img_size:   対象物のサイズ
    [in]  obj_num:    障害物の数
    [in]  img_num:    対象物の数
    [in]  create_num: 生成する画像の枚数
    [out] 生成された入力画像
    [out] 生成された正解画像
    """

    obj = getImgN(obj_path)
    hum = getImgN(h_path)
    bg = getImgN(bg_path)
    x = []
    for i in range(create_num):
        background = rondom_crop(getImg(bg, -1)[0], img_size)
        objects = []
        if(obj_num > 0):
            objects.extend(getSomeImg(obj, obj_num, obj_size))
            for j in objects:
                background, _ = IMG.paste(j, background)

        human = []
        if(img_num > 0):
            human.extend(getSomeImg(hum, img_num, obj_size, random=False))
            for k in human:
                background, _ = IMG.paste(k, background)

        x.append(background[:, :, :3])

    return x


def main(args):

    timer = Timer()
    print('create images...')
    timer.reset()
    x = create(args.other_path,
               args.human_path,
               args.background_path,
               args.obj_size, args.img_size,
               args.obj_num, args.human_num, args.img_num)
    timer.view()

    print('save images...')
    timer.reset()
    w_path = [F.getFilePath(args.out_path, GET.datetimeSHA(GET.randomStr(10), str_len=12), '.jpg')
              for i in x]
    [cv2.imwrite(w, i) for i, w in zip(x, w_path)]
    timer.view()

    print('save param...')
    timer.reset()
    F.dict2json(args.out_path, 'dataset', F.args2dict(args))
    timer.view()


if __name__ == '__main__':
    main(command())
