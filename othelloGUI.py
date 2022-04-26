import numpy as np
import copy
import tkinter as tk
from tkinter import messagebox
# import chainer
# from network import Net_AI
from game import Game
from alphabeta import *

G = Game()
# COM1 = Net_AI()
# chainer.serializers.load_npz('./model/ai_ver.2.net', COM1)

class GUI:
    def __init__(self, root):
        G.search()
        self.canvas = tk.Canvas(
            root,
            width=500,
            height=600,
            background="white"
        )
        self.Display()
        
        ret = messagebox.askyesno('先後の決定', '先手にしますか？')
        if ret == True:
            self.PLAYER = 1
            self.COM = self.PLAYER * (-1)
        else:
            self.PLAYER = -1
            self.COM = self.PLAYER * (-1)
            self.CPU()
        
        
        self.canvas.pack()
        self.Event()
        self.Display()
    
    def Event(self):
        self.canvas.bind('<Button-1>', self.click)
    
    def click(self, event):
        if event.x > 50 and event.x < 450 and event.y > 50 and event.y < 450:
            y = event.x // 50 - 1
            x = event.y // 50 - 1
        
        if G.turn == self.PLAYER:
            if len(G.seek(2)) == 0:
                G.PASS()
                messagebox.showinfo('PLAYER', 'PASS')
                G.search()
                self.Display()
                self.CPU()
            else:
                G.action(x,y)
                G.search()
                self.Display()
                self.CPU()
        
        if G.judge() != 0:
            self.Display()
            messagebox.showinfo('ゲーム終了', 'PLAYER:'+str(len(G.seek(self.PLAYER)))+'COM:'+str(len(G.seek(self.COM))))
            root.destroy()
            

    def Display(self):
        for i in range(8):
            for j in range(8):
                if G.state[i][j] == 0:
                    self.canvas.create_rectangle(50 + 50 * j, 50 + 50 * i, 100 + 50 * j, 100 + 50 * i, width=2, fill = "green")
                elif G.state[i][j] == 1:
                    self.canvas.create_rectangle(50 + 50 * j, 50 + 50 * i, 100 + 50 * j, 100 + 50 * i, width=2, fill = "green")
                    self.canvas.create_oval(50 + 5 + 50 * j, 50 + 5 + 50 * i, 100 - 5 + 50 * j, 100 - 5 + 50 * i,width=3, outline="black", fill = "black")
                elif G.state[i][j] == -1:
                    self.canvas.create_rectangle(50 + 50 * j, 50 + 50 * i, 100 + 50 * j, 100 + 50 * i, width=2, fill = "green")
                    self.canvas.create_oval(50 + 5 + 50 * j, 50 + 5 + 50 * i, 100 - 5 + 50 * j, 100 - 5 + 50 * i,width=3, outline="black", fill = "white")
                elif G.state[i][j] == 2:
                    self.canvas.create_rectangle(50 + 50 * j, 50 + 50 * i, 100 + 50 * j, 100 + 50 * i, width=2, fill = "green")
                    self.canvas.create_rectangle(50 + 20 + 50 * j, 50 + 20 + 50 * i, 100 - 20 + 50 * j, 100 - 20 + 50 * i, fill = "darkgreen",outline="")
    
    def CPU(self):
        if G.turn == self.COM:
            if len(G.seek(2)) == 0:
                G.PASS()
                messagebox.showinfo('COM', 'PASS')
                G.search()
                self.Display()
            else:
                x, y = choise(np.reshape(copy.deepcopy(G.state),[1,64]),copy.deepcopy(G.turn), depth=6)
                # tmp = np.reshape(copy.deepcopy(G.state),[1,64]).astype('float32')
                # tmp = COM1.forward(tmp)
                # tmp = np.argmax(tmp.array+np.reshape(np.where(copy.deepcopy(G.state)==2, 100, -100),[1,64]))
                # x = tmp // 8
                # y = tmp % 8
                G.action(x,y)
                G.search()
                self.Display()
        




root = tk.Tk()
root.title('Othello')
tmp = GUI(root)
root.mainloop()
