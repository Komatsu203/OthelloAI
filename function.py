import numpy as np

def rotation(input):
    ## input[:][64], output[:][64]
    output = input
    for n in range(input.shape[0]):
        tmp = np.reshape(input[n][:],[8,8])
        output[n][:] = np.reshape(np.rot90(tmp),[1,64])
    
    return output

def rot_T(input):
    ## input[:], output[:]
    output = input
    for n in range(input.shape[0]):
        x = input[n] // 8 - 3.5
        y = input[n] % 8 -3.5
        x_ = x + 3.5
        y_ = -y + 3.5
        output[n] = 8 * x_ + y_
    
    return output