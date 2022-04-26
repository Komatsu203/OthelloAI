import chainer
import chainer.links as L
import chainer.functions as F

class Net_AI(chainer.Chain):

    def __init__(self, n_in=64, n_hidden=64, n_out=64):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hidden)
            self.l2 = L.Linear(n_hidden, n_hidden)
            self.l3 = L.Linear(n_hidden, n_hidden)
            self.l4 = L.Linear(n_hidden, n_hidden)
            self.l5 = L.Linear(n_hidden, n_out)

    def forward(self, x):
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        h = F.sigmoid(self.l3(h))
        h = F.sigmoid(self.l4(h))
        h = self.l5(h)

        return h

class Net_Eval(chainer.Chain):

    def __init__(self, n_in=64, n_hidden=64, n_out=1):
        super().__init__()
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_hidden)
            self.l2 = L.Linear(n_hidden, n_hidden)
            self.l3 = L.Linear(n_hidden, n_hidden)
            self.l4 = L.Linear(n_hidden, n_hidden)
            self.l5 = L.Linear(n_hidden, n_out)

    def forward(self, x):
        h = F.sigmoid(self.l1(x))
        h = F.sigmoid(self.l2(h))
        h = F.sigmoid(self.l3(h))
        h = F.sigmoid(self.l4(h))
        h = self.l5(h)

        return h