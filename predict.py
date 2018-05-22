#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = 'モデルとモデルパラメータを利用して推論実行する'
#

import cv2
import time
import argparse
import numpy as np

import chainer
import chainer.links as L
#from chainer.cuda import to_cpu

from Lib.network import KB
from create_dataset import create
import Tools.imgfunc as IMG
#import Tools.getfunc as GET
import Tools.func as F

from PIL import Image


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
    parser.add_argument('--batch', '-b', type=int, default=100,
                        help='ミニバッチサイズ [default: 100]')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID [default -1]')
    parser.add_argument('--out_path', '-o', default='./result/',
                        help='生成物の保存先[default: ./result/]')
    return parser.parse_args()


def predict(model, img, batch, gpu):
    """
    推論実行メイン部
    [in]  model:     推論実行に使用するモデル
    [in]  img:       入力画像
    [in]  batch:     バッチサイズ
    [in]  gpu:       GPU ID
    [out] img:       推論実行で得られた生成画像
    """

    # dataには圧縮画像と分割情報が含まれているので、分離する
    st = time.time()
    # バッチサイズごとに学習済みモデルに入力して画像を生成する
    x = chainer.links.model.vision.resnet.prepare(img)
    ch, w, h = x.shape
    x = np.array(x).reshape((-1, ch, w, h))
    print(x.shape)
    y = model.predictor(x)
    print(y)
    print(y.data)
    print(y.data.argmax(axis=1))
    print('exec time: {0:.2f}[s]'.format(time.time() - st))
    return np.argmax(y.data[0])


def main(args):
    # jsonファイルから学習モデルのパラメータを取得する
    # net, unit, ch, size, layer, sr, af1, af2 = GET.modelParam(args.param)
    # 学習モデルを生成する
    model = L.Classifier(KB(n_out=5))

    # load_npzのpath情報を取得し、学習済みモデルを読み込む
    load_path = F.checkModelType(args.model)
    print(args.model)
    print('AAAA', load_path)
    try:
        chainer.serializers.load_npz(args.model, model, path=load_path)
    except:
        import traceback
        traceback.print_exc()
        print(F.fileFuncLine())
        exit()

    # GPUの設定
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    # else:
    #     model.to_intel64()

    # 画像の生成
    if args.image == '':
        x = create(args.other_path,
                   args.human_path,
                   args.background_path,
                   args.obj_size, args.img_size,
                   args.obj_num, 2, 1)[0]
        print(x.shape)
    elif IMG.isImgPath(args.image):

        #x = Image.open(args.image)

        x = cv2.cvtColor(
            cv2.imread(args.image, IMG.getCh(0)), cv2.COLOR_RGB2BGR
        )

        w, h = x.shape[:2]
        #x = np.array(x).reshape(1, w, h, -1)
    else:
        print('input image path is not found:', args.image)
        exit()

    # 学習モデルを実行する
    with chainer.using_config('train', False):
        num = predict(model, x, args.batch, args.gpu)

    # 生成結果を保存する
    name = F.getFilePath(args.out_path, 'predict-' + str(num).zfill(2), '.jpg')
    print('save:', name)
    cv2.imwrite(name, cv2.cvtColor(np.asarray(x), cv2.COLOR_BGR2RGB))
    cv2.imshow(name,  cv2.cvtColor(np.asarray(x), cv2.COLOR_BGR2RGB))
    cv2.waitKey()


if __name__ == '__main__':
    args = command()
    F.argsPrint(args)
    main(args)
