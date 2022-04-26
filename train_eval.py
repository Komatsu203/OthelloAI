import numpy as np
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import chainer.functions as F
from network import Net_Eval
from sklearn.model_selection import train_test_split
from function import *

X = np.load("./dataset/state_.npy",allow_pickle=True)
T = np.load("./dataset/value_.npy",allow_pickle=True)
X_ = np.load("./dataset/state.npy",allow_pickle=True)
T_ = np.load("./dataset/value.npy",allow_pickle=True)

rot90X = rotation(X)
rot180X = rotation(rot90X)
rot270X = rotation(rot180X)

X = np.concatenate([X, rot90X, rot180X, rot270X, X_], 0)
T = np.concatenate([T, T, T, T, T_], 0)

rng = np.random.default_rng(seed=0)
rng.shuffle(X)
rng = np.random.default_rng(seed=0)
rng.shuffle(T)

T = np.reshape(T, [len(T),1])

print('X:', X.shape)
print('T:', T.shape)

x = X.astype('float32')
t = T.astype('float32')

x_train_val, x_test, t_train_val, t_test = train_test_split(x, t, test_size=0.2, random_state=0)
x_train, x_val, t_train, t_val = train_test_split(x_train_val, t_train_val, test_size=0.2, random_state=0)

eval = Net_Eval()
optimizer = chainer.optimizers.Adam()
optimizer.setup(eval)

n_epoch = 50
n_batchsize = 100

iteration = 0

# ログの保存用
results_train = {
    'loss': [],
    'accuracy': []
}
results_valid = {
    'loss': [],
    'accuracy': []
}

for epoch in range(n_epoch):

    # データセット並べ替えた順番を取得
    order = np.random.permutation(range(len(x_train)))

    # 各バッチ毎の目的関数の出力と分類精度の保存用
    loss_list = []
    accuracy_list = []

    for i in range(0, len(order), n_batchsize):
        # バッチを準備
        index = order[i:i+n_batchsize]
        x_train_batch = x_train[index,:]
        t_train_batch = t_train[index]

        # 予測値を出力
        y_train_batch = eval.forward(x_train_batch)

        # 目的関数を適用し、分類精度を計算
        loss_train_batch = F.mean_squared_error(y_train_batch, t_train_batch)
        # accuracy_train_batch = F.accuracy(y_train_batch, t_train_batch)

        loss_list.append(loss_train_batch.array)
        # accuracy_list.append(accuracy_train_batch.array)

        # 勾配のリセットと勾配の計算
        eval.cleargrads()
        loss_train_batch.backward()

        # パラメータの更新
        optimizer.update()

        # カウントアップ
        iteration += 1

    # 訓練データに対する目的関数の出力と分類精度を集計
    loss_train = np.mean(loss_list)
    accuracy_train = np.mean(accuracy_list)

    # 1エポック終えたら、検証データで評価
    # 検証データで予測値を出力
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        y_val = eval(x_val)

    # 目的関数を適用し、分類精度を計算
    loss_val = F.mean_squared_error(y_val, t_val)
    # accuracy_val = F.accuracy(y_val, t_val)

    # 結果の表示
    print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'.format(
        epoch, iteration, loss_train, loss_val.array))

    # ログを保存
    results_train['loss'] .append(loss_train)
    results_train['accuracy'] .append(accuracy_train)
    results_valid['loss'].append(loss_val.array)
    # results_valid['accuracy'].append(accuracy_val.array)

# 目的関数の出力 (loss)
plt.plot(results_train['loss'], label='train')  # label で凡例の設定
plt.plot(results_valid['loss'], label='valid')  # label で凡例の設定
plt.legend()  # 凡例の表示
plt.show()

chainer.serializers.save_npz('./model/eval_ver.2.net', eval)