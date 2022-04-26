from matplotlib.pyplot import axis
import numpy as np
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
        T1 = np.empty((0))
        X2 = np.empty((0,64))
        T2 = np.empty((0))

        while k < len(kifu):
            G.search()
            # G.display()
            if len(G.seek(2)) > 0:
                if G.turn == G.BLACK:
                    x = int(kifu[k + 1])-1
                    y = int(kifu[k])
                    X1 = np.append(X1, np.reshape(copy.deepcopy(G.state),[1,64]), axis = 0)
                    T1 = np.append(T1, 8*x+y)
                else:
                    x = int(kifu[k + 1])-1
                    y = int(kifu[k])
                    X2 = np.append(X2, np.reshape(copy.deepcopy(G.state),[1,64]), axis = 0)
                    T2 = np.append(T2, 8*x+y)
                G.action(x,y)
                k += 2
            else:
                G.PASS()
        if len(G.seek(+1)) > len(G.seek(-1)):
            X = np.append(X, X1, axis = 0)
            T = np.append(T, T1)
        elif len(G.seek(+1)) < len(G.seek(-1)):
            X = np.append(X, X2, axis = 0)
            T = np.append(T, T2)

print(X,T)
print(X.shape,T.shape)
np.save("./dataset/X_.npy",X)
np.save("./dataset/Y_.npy",T)
        
