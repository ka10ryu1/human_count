#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'keytouch_camのネットワーク部分'
#

import time

from chainer import Chain
import chainer.initializers as I
import chainer.functions as F
import chainer.links as L


class KB(Chain):
    def __init__(self, n_unit, n_out, base=L.ResNet50Layers(),
                 layer='pool5', actfun=F.relu, dropout=0.0, view=False):
        """
        [in] n_unit:    中間層のユニット数
        [in] n_out:     出力チャンネル
        [in] actfun1: 活性化関数（Layer A用）
        [in] actfun2: 活性化関数（Layer B用）
        """

        super(KB, self).__init__()
        with self.init_scope():
            self.base = base
            self.fc = L.Linear(None, n_out)

        self.layer = layer
        self.view = view
        self.timer = 0

        # print('[Network info]', self.__class__.__name__)
        # print('  Unit:\t{0}\n  Out:\t{1}\n  Drop out:\t{2}\nAct Func:\t{3}'.format(
        #     n_unit, n_out, dropout, actfun.__name__)
        # )

    def __call__(self, x):

        # print(x.shape)
        # print(self.base.available_layers)
        h = self.base(x, layers=[self.layer])
        return self.fc(h[self.layer])
        # if self.view:
        #     self.timer = time.time()

        # if self.view:
        #     print('Output {0:5.3f} s: {1}'.format(time.time()-self.timer, y.shape))
        #     exit()
        # else:
        #     return y
