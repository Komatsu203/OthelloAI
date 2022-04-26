import numpy as np
import copy
import chainer
from network import Net_AI, Net_Eval

COM = Net_AI()
chainer.serializers.load_npz('./model/ai_ver.2.net', COM)
EVAL = Net_Eval()
chainer.serializers.load_npz('./model/eval_ver.2.net', EVAL)

def action(state, turn, x, y):
    if state[x][y] == 2:
        state[x][y] = turn
        for dx in [-1, 0, +1]:
            for dy in [-1, 0, +1]:
                if dx == 0 and dy == 0:
                    pass
                else:
                    tx = dx
                    ty = dy
                    flag = 0
                    while True:
                        if x+tx < 0 or y+ty < 0 or x+tx > 7 or y+ty > 7:
                            break
                        if state[x+tx][y+ty] == -turn:
                            tx += dx
                            ty += dy
                        else:
                            break
                        if x+tx < 0 or y+ty < 0 or x+tx > 7 or y+ty > 7:
                            break
                        if state[x+tx][y+ty] == turn:
                            flag = 1
                            break
                    # print(flag)
                    if flag == 1:
                        while True:
                            if x+tx == x and y+ty == y:
                                break
                            else:
                                state[x+tx][y+ty] = turn
                                tx -= dx
                                ty -= dy
        turn *= -1
    else:
        print(state)
        print(x,y)
        print('Error:action()')

    return state

def expand(state, turn):
    children = np.empty((0,64))
    values = COM.forward(copy.deepcopy(state).astype('float32'))
    n = np.sum(np.where(copy.deepcopy(state) == 2, 1, 0))
    state = np.reshape(state, [8,8])
    if n > 2:
        for number in range(2):
            idx = argsubmax(values.array[0][:] + np.where(np.reshape(copy.deepcopy(state),[1,64]) == 2, 1000, 0)[0][:],number+1)
            x = idx[0] // 8
            y = idx[0] % 8
            children = np.append(children, np.reshape(action(copy.deepcopy(state), turn, x, y),[1,64]), 0)
    elif n > 0:
        for number in range(n):
            idx = argsubmax(values.array[0][:] + np.where(np.reshape(copy.deepcopy(state),[1,64]) == 2, 1000, 0)[0][:],number+1)
            x = idx[0] // 8
            y = idx[0] % 8
            children = np.append(children, np.reshape(action(copy.deepcopy(state), turn, x, y),[1,64]), 0)
    else:
        return children

    return children
    
def argsubmax(arr, number):
    idxs = np.where(arr == np.sort(arr)[-number])
    return idxs[0]

def minmax(state, turn, depth):
    return alphabeta(state, turn, depth, -np.inf, np.inf)

def alphabeta(state, turn, depth, alpha, beta):
    state = search(state,turn)
    children = expand(state, -turn)
    if len(children) == 0 or depth == 0:
        return eval(state)
    if turn == +1:
        for i in range(len(children)):
            alpha = max(alpha , alphabeta(children[i][:], -turn, depth-1, alpha, beta))
            if alpha >= beta:
                break
        return alpha
    else:
        for i in range(len(children)):
            beta = min(beta, alphabeta(children[i][:], -turn, depth-1, alpha, beta))
            if alpha >= beta:
                break
        return beta

def eval(state):
    value = EVAL.forward(state.astype('float32'))
    return value.array[0][0]

def search(state, turn):
    state = np.reshape(state, [8,8])
    for x in range(8):
        for y in range(8):
            cnt = 0
            if state[x][y] == 2:
                state[x][y] = 0
            if state[x][y] == 0:
                for dx in [-1, 0, +1]:
                    for dy in [-1, 0, +1]:
                        if dx == 0 and dy == 0:
                            pass
                        else:
                            tx = dx
                            ty = dy
                            while True:
                                if x+tx < 0 or y+ty < 0 or x+tx > 7 or y+ty > 7:
                                    break
                                if state[x+tx][y+ty] == -turn:
                                    tx += dx
                                    ty += dy
                                else:
                                    break
                                if x+tx < 0 or y+ty < 0 or x+tx > 7 or y+ty > 7:
                                    break
                                if state[x+tx][y+ty] == turn:
                                    cnt += 1
                                    break
            if cnt > 0:
                state[x][y] = 2
    return np.reshape(state, [1,64])

def choise(state, turn, depth):
    values = COM.forward(copy.deepcopy(state).astype('float32'))
    n = np.sum(np.where(copy.deepcopy(state) == 2, 1, 0))
    state = np.reshape(state, [8,8])
    best = -np.inf
    if n > 2:
        for number in range(2):
            idx = argsubmax(values.array[0][:] + np.where(np.reshape(copy.deepcopy(state),[1,64]) == 2, 100, -100)[0][:],number+1)
            x = idx[0] // 8
            y = idx[0] % 8
            value = minmax(np.reshape(action(copy.deepcopy(state), turn, x, y),[1,64]),turn, depth)
            if value > best:
                best = value
                x_ = x
                y_ = y
    else:
        for number in range(n):
            idx = argsubmax(values.array[0][:] + np.where(np.reshape(copy.deepcopy(state),[1,64]) == 2, 100, -100)[0][:],number+1)
            x = idx[0] // 8
            y = idx[0] % 8
            value = minmax(np.reshape(action(copy.deepcopy(state), turn, x, y),[1,64]),turn, depth)
            if value > best:
                best = value
                x_ = x
                y_ = y

    return x_, y_

# state = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
#                   [ 0, 0, 0, 0, 0, 0, 0, 0],
#                   [ 0, 0, 0, 2, 0, 0, 0, 0],
#                   [ 0, 0, 2,-1, 1, 0, 0, 0],
#                   [ 0, 0, 0, 1,-1, 2, 0, 0],
#                   [ 0, 0, 0, 0, 1, 0, 0, 0],
#                   [ 0, 0, 0, 0, 0, 0, 0, 0],
#                   [ 0, 0, 0, 0, 0, 0, 0, 0]])

# turn = 1

# # children = expand(np.reshape(state,[1,64]),turn)

# # for i in range(children.shape[0]):
# #     print(np.reshape(children[i][:],[8,8]))

# state = np.reshape(state,[1,64])

# print(choise(state, turn, depth=5))