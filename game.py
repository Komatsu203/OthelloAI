import numpy as np
import random as r

class Game:
    def __init__(self):
        self.state = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                               [ 0, 0, 0, 0, 0, 0, 0, 0],
                               [ 0, 0, 0, 0, 0, 0, 0, 0],
                               [ 0, 0, 0,-1, 1, 0, 0, 0],
                               [ 0, 0, 0, 1,-1, 0, 0, 0],
                               [ 0, 0, 0, 0, 0, 0, 0, 0],
                               [ 0, 0, 0, 0, 0, 0, 0, 0],
                               [ 0, 0, 0, 0, 0, 0, 0, 0]])

        self.BLACK = +1
        self.WHITE = -1
        self.pass_flag = 0
        self.turn = self.BLACK

    def search(self):
        for x in range(8):
            for y in range(8):
                cnt = 0
                if self.state[x][y] == 2:
                    self.state[x][y] = 0
                if self.state[x][y] == 0:
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
                                    if self.state[x+tx][y+ty] == -self.turn:
                                        tx += dx
                                        ty += dy
                                    else:
                                        break
                                    if x+tx < 0 or y+ty < 0 or x+tx > 7 or y+ty > 7:
                                        break
                                    if self.state[x+tx][y+ty] == self.turn:
                                        cnt += 1
                                        break
                if cnt > 0:
                    self.state[x][y] = 2

    def action(self, x, y):
        if self.state[x][y] == 2:
            self.state[x][y] = self.turn
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
                            if self.state[x+tx][y+ty] == -self.turn:
                                tx += dx
                                ty += dy
                            else:
                                break
                            if x+tx < 0 or y+ty < 0 or x+tx > 7 or y+ty > 7:
                                break
                            if self.state[x+tx][y+ty] == self.turn:
                                flag = 1
                                break
                        # print(flag)
                        if flag == 1:
                            while True:
                                if x+tx == x and y+ty == y:
                                    break
                                else:
                                    self.state[x+tx][y+ty] = self.turn
                                    tx -= dx
                                    ty -= dy
            self.pass_flag = 0
            self.turn *= -1
        else:
            print('Error:action()')
    
    def PASS(self):
        self.pass_flag += 1
        self.turn *= -1
    
    def judge(self):
        if len(self.seek(0)) + len(self.seek(2)) == 0 or self.pass_flag == 2:
            if len(self.seek(self.BLACK)) > len(self.seek(self.WHITE)):
                return self.BLACK
            elif len(self.seek(self.BLACK)) < len(self.seek(self.WHITE)):
                return self.WHITE
            else:
                return 2
        elif len(self.seek(self.BLACK)) == 0:
            return self.WHITE
        elif len(self.seek(self.WHITE)) == 0:
            return self.BLACK
        else:
            return 0
        
    def seek(self,target):
        idx = []
        for x in range(8):
            for y in range(8):
                if self.state[x][y] == target:
                    idx.append([x,y])
        
        return idx

    def display(self):
        print(str(self.state).replace("0","").replace("-1", "x").replace(" 1","o").replace(" 2","."))


# N = 1000
# Cnt_b = 0
# Cnt_w = 0
# for n in range(N):
#     t = Game()
#     while True:
#         t.search()
#         # t.display()
#         idx = t.seek(2)
#         if len(idx) > 0:
#             l = r.randint(0,len(idx)-1)
#             t.action(idx[l][0],idx[l][1])
#         else:
#             t.PASS()
#         # print(t.judge())
#         if t.judge() != 0:
#             break
#     if t.judge() == t.BLACK:
#         Cnt_b += 1
#     elif t.judge() == t.WHITE:
#         Cnt_w += 1

# print(Cnt_b/N)
# print(Cnt_w/N)