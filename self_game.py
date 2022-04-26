import numpy as np
import random as r
import copy
import chainer
from network import Net_AI, Net_Eval
from game import Game
from alphabeta import *

COM1 = Net_AI()
chainer.serializers.load_npz('./model/ai_ver.2.net', COM1)

N = 100000

state = np.empty((0,64))
value = np.empty((0))

for n in range(N):
    G = Game()
    X = np.empty((0,64))
    while True:
        G.search()
        # G.display()
        idx = G.seek(2)
        num_idx = len(idx)
        if num_idx > 0:
            values = COM1.forward(np.reshape(copy.deepcopy(G.state),[1,64]).astype('float32'))
            if num_idx > 2:
                tmp = argsubmax(values.array[0][:] + np.where(np.reshape(copy.deepcopy(G.state),[1,64]) == 2, 1000, 0)[0][:],r.randint(0,1)+1)
                x = tmp[0] // 8
                y = tmp[0] % 8
            else:
                tmp = argsubmax(values.array[0][:] + np.where(np.reshape(copy.deepcopy(G.state),[1,64]) == 2, 1000, 0)[0][:],r.randint(0,num_idx-1)+1)
                x = tmp[0] // 8
                y = tmp[0] % 8
            X = np.append(X ,np.reshape(copy.deepcopy(G.state),[1,64]),0)
            G.action(x,y)
        else:
            G.PASS()
        if G.judge() != 0:
            if G.judge() == G.BLACK:
                state = np.concatenate([state, np.reshape(X[r.randint(0,X.shape[0]-1)][:],[1,64]), np.reshape(X[r.randint(0,X.shape[0]-1)][:],[1,64]), np.reshape(X[r.randint(0,X.shape[0]-1)][:],[1,64])])
                value = np.concatenate([value, [1], [1], [1]])
            elif G.judge() == G.WHITE:
                state = np.concatenate([state, np.reshape(X[r.randint(0,X.shape[0]-1)][:],[1,64]), np.reshape(X[r.randint(0,X.shape[0]-1)][:],[1,64]), np.reshape(X[r.randint(0,X.shape[0]-1)][:],[1,64])])
                value = np.concatenate([value, [-1], [-1], [-1]])
            else:
                state = np.concatenate([state, np.reshape(X[r.randint(0,X.shape[0]-1)][:],[1,64]), np.reshape(X[r.randint(0,X.shape[0]-1)][:],[1,64]), np.reshape(X[r.randint(0,X.shape[0]-1)][:],[1,64])])
                value = np.concatenate([value, [0], [0], [0]])
            break
    if n % 1000 == 0:
        print('Game:' + str(n))

print(state.shape, value.shape)
np.save("./dataset/state_.npy",state)
np.save("./dataset/value_.npy",value)
