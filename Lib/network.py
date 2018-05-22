#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'human_countのネットワーク部分'
#

from chainer import Chain
import chainer.links as L
import chainer.functions as F


class KB(Chain):
    def __init__(self, n_out, actfun=F.relu,
                 base=L.ResNet50Layers(),
                 layer='pool5', view=False):
        """
        [in] n_out:     出力チャンネル
        """

        super(KB, self).__init__()
        with self.init_scope():
            self.base = base
            self.l1 = L.Linear(None, 1024)
            self.l2 = L.Linear(None, 512)
            self.l3 = L.Linear(None, 256)
            self.lN = L.Linear(None, n_out)

        self.actfun = actfun
        self.layer = layer
        self.view = view
        self.timer = 0

    def __call__(self, x):
        h = self.base(x, layers=[self.layer])
        h = self.actfun(self.l1(h[self.layer]))
        h = self.actfun(self.l2(h))
        h = self.actfun(self.l3(h))
        return self.lN(h)
