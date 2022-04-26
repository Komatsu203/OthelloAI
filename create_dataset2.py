import numpy as np
import random as r
import copy
import csv
from game import Game

X = np.empty((0,64))
T = np.empty((0))

for frameCnt in range(33):
    print("loading <./kifu/wthor-"+str(frameCnt + 1)+".csv>")
    csv_file = open("./kifu/wthor-"+str(frameCnt + 1)+".csv", "r", encoding="utf_8", errors="", newline="" )
    f = csv.reader(csv_file, delimiter=",", doublequote=True, lineterminator="\r\n", quotechar='"', skipinitialspace=True)
    header = next(f)
    for row in f:
        kifu = row[8].replace("a",'0').replace("b",'1').replace("c",'2').replace("d",'3').replace("e",'4').replace("f",'5').replace("g",'6').replace("h",'7')
        G = Game()
        k = 0

        X1 = np.empty((0,64))

        while k < len(kifu):
            G.search()
            if len(G.seek(2)) > 0:
                x = int(kifu[k + 1])-1
                y = int(kifu[k])
                X1 = np.append(X1, np.reshape(copy.deepcopy(G.state),[1,64]), axis = 0)
                G.action(x,y)
                k += 2
            else:
                G.PASS()
        if len(G.seek(+1)) > len(G.seek(-1)):
            X = np.concatenate([X, np.reshape(X1[r.randint(0,X1.shape[0]-1)][:],[1,64])])
            T = np.append(T,1)
        elif len(G.seek(+1)) < len(G.seek(-1)):
            X = np.concatenate([X, np.reshape(X1[r.randint(0,X1.shape[0]-1)][:],[1,64])])
            T = np.append(T,-1)
        else:
            X = np.concatenate([X, np.reshape(X1[r.randint(0,X1.shape[0]-1)][:],[1,64])])
            T = np.append(T,0)

print(X.shape,T.shape)

## 90°回転
rot_X = X
rot_T = T

for n in range(rot_X.shape[0]):
    tmpX = np.reshape(rot_X[n][:],[8,8])
    rot_X[n][:] = np.reshape(np.rot90(tmpX),[1,64])

X = np.concatenate([X, rot_X], 0)
T = np.concatenate([T, rot_T], 0)

## 180°回転
for n in range(rot_X.shape[0]):
    tmpX = np.reshape(rot_X[n][:],[8,8])
    rot_X[n][:] = np.reshape(np.rot90(tmpX),[1,64])

X = np.concatenate([X, rot_X], 0)
T = np.concatenate([T, rot_T], 0)

## 270°回転
for n in range(rot_X.shape[0]):
    tmpX = np.reshape(rot_X[n][:],[8,8])
    rot_X[n][:] = np.reshape(np.rot90(tmpX),[1,64])

X = np.concatenate([X, rot_X], 0)
T = np.concatenate([T, rot_T], 0)


rng = np.random.default_rng(seed=0)
rng.shuffle(X)
rng = np.random.default_rng(seed=0)
rng.shuffle(T)

print(X,T)
print(X.shape,T.shape)
np.save("./dataset/state.npy",X)
np.save("./dataset/value.npy",T)
        
