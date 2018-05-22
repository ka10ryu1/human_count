#!/usr/bin/env python3
# -*-coding: utf-8 -*-
#
help = '学習メイン部'
#

import logging
# basicConfig()は、 debug()やinfo()を最初に呼び出す"前"に呼び出すこと
logging.basicConfig(format='%(message)s')
logging.getLogger('Tools').setLevel(level=logging.INFO)

import os
import json
import argparse

import numpy as np

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.iterators import MultiprocessIterator

from Lib.network import KB
from Lib.plot_report_log import PlotReportLog
# import Tools.imgfunc as IMG
import Tools.getfunc as GET
import Tools.func as F
from Lib.read_dataset_CV2 import LabeledImageDataset


class Transform(chainer.dataset.DatasetMixin):
    def __init__(self, dataset, prepare, x_dtype=np.float32, y_dtype=np.int8):
        self._dataset = dataset
        self._prepare = prepare
        self._x_dtype = x_dtype
        self._y_dtype = y_dtype
        self._len = len(self._dataset)

    def __len__(self):
        # データセットの数を返します
        return self._len

    def get_example(self, i):
        # データセットのインデックスを受け取って、データを返します
        inputs = self._dataset[i]
        x, y = inputs
        x = self._prepare(x)
        return x.astype(self._x_dtype), y.astype(self._y_dtype)


def command():
    parser = argparse.ArgumentParser(description=help)
    parser.add_argument('-i', '--in_path', default='./result/',
                        help='入力データセットのフォルダ [default: ./result/]')
    parser.add_argument('-opt', '--optimizer', default='adam',
                        help='オプティマイザ [default: adam, other: ada_d/ada_g/m_sgd/n_ag/rmsp/rmsp_g/sgd/smorms]')
    parser.add_argument('-lf', '--lossfun', default='mse',
                        help='損失関数 [default: mse, other: mae, ber, gauss_kl]')
    parser.add_argument('-b', '--batchsize', type=int, default=20,
                        help='ミニバッチサイズ [default: 20]')
    parser.add_argument('-e', '--epoch', type=int, default=10,
                        help='学習のエポック数 [default 10]')
    parser.add_argument('-f', '--frequency', type=int, default=-1,
                        help='スナップショット周期 [default: -1]')
    parser.add_argument('-g', '--gpu_id', type=int, default=-1,
                        help='使用するGPUのID [default -1]')
    parser.add_argument('-o', '--out_path', default='./result/',
                        help='生成物の保存先[default: ./result/]')
    parser.add_argument('-r', '--resume', default='',
                        help='使用するスナップショットのパス[default: no use]')
    parser.add_argument('--noplot', dest='plot', action='store_false',
                        help='学習過程をPNG形式で出力しない場合に使用する')
    parser.add_argument('--only_check', action='store_true',
                        help='オプション引数が正しく設定されているかチェックする')
    args = parser.parse_args()
    F.argsPrint(args)
    return args


def getDataset(folder):

    # 探索するフォルダがなければ終了
    if not os.path.isdir(folder):
        print('[Error] folder not found:', folder)
        print(F.fileFuncLine())
        exit()

    # 学習用データとテスト用データを発見したらTrueにする
    train_flg = False
    test_flg = False
    out_n = 0

    for l in os.listdir(folder):
        name, ext = os.path.splitext(os.path.basename(l))
        if os.path.isdir(l):
            pass
        elif('train_' in name)and('.txt' in ext)and(train_flg is False):
            train = LabeledImageDataset(os.path.join(folder, l))
            train = Transform(train, chainer.links.model.vision.resnet.prepare)

            train_flg = True
            out_n = int(name.split('_')[1])
        elif('test_' in name)and('.txt' in ext)and(test_flg is False):
            test = LabeledImageDataset(os.path.join(folder, l))
            test = Transform(test, chainer.links.model.vision.resnet.prepare)
            test_flg = True
            out_n = int(name.split('_')[1])

    return train, test, out_n


def main(args):

    # 各種データをユニークな名前で保存するために時刻情報を取得する
    exec_time = GET.datetimeSHA()
    # Load dataset
    train, test, out_n = getDataset(args.in_path)
    # モデルを決定する
    model = L.Classifier(KB(n_out=out_n, view=args.only_check))

    if args.gpu_id >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu_id).use()
        model.to_gpu()  # Copy the model to the GPU
        chainer.global_config.autotune = True
    # else:
    #     model.to_intel64()

    # Setup an optimizer
    optimizer = GET.optimizer(args.optimizer).setup(model)

    for func_name in model.predictor.base._children:
        for param in model.predictor.base[func_name].params():
            param.update_rule.hyperparam.alpha *= 0.25

    # Setup iterator
    train_iter = MultiprocessIterator(train, args.batchsize)
    test_iter = MultiprocessIterator(test, args.batchsize,
                                     repeat=False, shuffle=False)

    # train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    # test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
    #                                              repeat=False, shuffle=False)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, device=args.gpu_id)
    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out_path)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu_id))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(
        extensions.dump_graph('main/loss', out_name=exec_time + '_graph.dot')
    )

    # Take a snapshot for each specified epoch
    frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
    trainer.extend(
        extensions.snapshot(filename=exec_time + '_{.updater.epoch}.snapshot'),
        trigger=(frequency, 'epoch')
    )

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(log_name=exec_time + '.log'))
    # trainer.extend(extensions.observe_lr())

    # Save two plot images to the result dir
    if args.plot and extensions.PlotReport.available():
        trainer.extend(
            PlotReportLog(['main/loss', 'validation/main/loss'],
                          'epoch', file_name='loss.png')
        )

        trainer.extend(
            extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],
                                  'epoch', file_name='acc.png')
        )
        # trainer.extend(
        #     PlotReportLog(['lr'],
        #                   'epoch', file_name='lr.png', val_pos=(-80, -60))
        # )

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport([
        'epoch',
        'main/loss',
        'validation/main/loss',
        'main/accuracy',
        'validation/main/accuracy',
        # 'lr',
        'elapsed_time'
    ]))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # predict.pyでモデルを決定する際に必要なので記憶しておく
    model_param = {i: getattr(args, i) for i in dir(args) if not '_' in i[0]}
    if args.only_check is False:
        # predict.pyでモデルのパラメータを読み込むjson形式で保存する
        with open(F.getFilePath(args.out_path, exec_time, '.json'), 'w') as f:
            json.dump(model_param, f, indent=4, sort_keys=True)

    # Run the training
    trainer.run()

    # 最後にモデルを保存する
    # スナップショットを使ってもいいが、
    # スナップショットはファイルサイズが大きいので
    chainer.serializers.save_npz(
        F.getFilePath(args.out_path, exec_time, '.model'),
        model
    )


if __name__ == '__main__':
    main(command())
