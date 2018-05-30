#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'モデルとモデルパラメータを利用して推論実行する'
#

import cv2
import time
import argparse
import numpy as np

try:
    import cupy
except ImportError:
    logger.warning('not import cupy')

import chainer
import chainer.links as L
# from chainer.cuda import to_cpu
import chainer.functions as F

from chainer.links.model.vision import resnet

# import Tools.imgfunc as IMG
# import Tools.getfunc as GET
import Tools.func as FNC
import Tools.getfunc as GET
from Lib.network import KB
from create_dataset import create


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('model',
                        help='使用する学習済みモデル')
    parser.add_argument('param',
                        help='使用するモデルパラメータ')
    parser.add_argument('-i', '--image', default='',
                        help='別途使用したい画像があれば')
    parser.add_argument('-ot', '--other_path', default='./Image/other/',
                        help='動物、怪獣の画像フォルダ (default: ./Image/other/')
    parser.add_argument('-hu', '--human_path', default='./Image/people/',
                        help='人間の画像フォルダ (default: ./Image/people/')
    parser.add_argument('-bg', '--background_path', default='./Image/background/',
                        help='背景の画像フォルダ (default: ./Image/background/')
    parser.add_argument('-os', '--obj_size', type=int, default=64,
                        help='挿入する画像サイズ [default: 64 pixel]')
    parser.add_argument('-on', '--obj_num', type=int, default=3,
                        help='画像を生成する数 [default: 3]')
    parser.add_argument('-is', '--img_size', type=int, default=256,
                        help='生成される画像サイズ [default: 256 pixel]')
    parser.add_argument('-in', '--img_num', type=int, default=30,
                        help='1種類あたりの画像数 [default: 30]')
    parser.add_argument('--batch', '-b', type=int, default=100,
                        help='ミニバッチサイズ [default: 100]')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID [default -1]')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先[default: ./result/]')
    return parser.parse_args()


def imgs2resnet(imgs, xp=np):
    dst = [resnet.prepare(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
           for img in imgs]
    return xp.array(dst)


def main(args):
    # jsonファイルから学習モデルのパラメータを取得する
    n_out = GET.jsonData(args.param, ['n_out'])
    # 学習モデルを生成する
    model = L.Classifier(KB(n_out=n_out))

    # load_npzのpath情報を取得し、学習済みモデルを読み込む
    load_path = FNC.checkModelType(args.model)
    try:
        chainer.serializers.load_npz(args.model, model, path=load_path)
    except:
        import traceback
        traceback.print_exc()
        print(FNC.fileFuncLine())
        exit()

    # GPUの設定
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
        xp = cupy
    else:
        xp = np
    #     model.to_intel64()

    # 画像の生成
    x = []
    t = []
    for i in range(n_out):
        x.extend(
            create(args.other_path, args.human_path, args.background_path,
                   args.obj_size, args.img_size, args.obj_num, i, args.img_num)
        )
        t.extend([i]*args.img_num)

    x = imgs2resnet(np.array(x), xp)
    t = xp.array(t, dtype=np.int8)
    print(x.shape, t.shape)

    # 学習モデルを実行する
    with chainer.using_config('train', False):
        st = time.time()
        y = model.predictor(x)
        print('exec time: {0:.2f}[s]'.format(time.time() - st))

    p, r, f, _ = F.classification_summary(y, t)
    precision = p.data.tolist()
    recall = r.data.tolist()
    F_score = f.data.tolist()
    print('num|precision|recall|F')
    [print('{0:3}|   {1:6.4f}|{2:6.4f}|{3:6.4f}'.format(i, elem[0], elem[1], elem[2]))
     for i, elem in enumerate(zip(precision, recall, F_score))]


if __name__ == '__main__':
    args = command()
    FNC.argsPrint(args)
    main(args)
