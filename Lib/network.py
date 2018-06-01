#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
help = 'human_countのネットワーク部分'
#

from chainer import Chain
import chainer.links as L
import chainer.functions as F


class CNT(Chain):
    def __init__(self, n_out, n_unit=1024, actfun=F.relu, dropout=0.0,
                 base=L.ResNet50Layers(), layer='pool5'):
        """
        [in] n_out:     出力チャンネル
        """

        super(CNT, self).__init__()
        with self.init_scope():
            self.base = base
            self.l1 = L.Linear(None, n_unit)
            self.l2 = L.Linear(None, n_unit//2)
            self.l3 = L.Linear(None, n_unit//4)
            self.lN = L.Linear(None, n_out)

        self._actfun = actfun
        self._dropout_ratio = dropout
        self._layer = layer

    def _linear(self, l, h):
        h = self._actfun(l(h))
        if self._dropout_ratio > 0:
            h = F.dropout(h, self._dropout_ratio)

        return h

    def __call__(self, x):
        h = self.base(x, layers=[self._layer])
        h = self._linear(self.l1, h[self._layer])
        h = self._linear(self.l2, h)
        h = self._linear(self.l3, h)
        return self.lN(h)
