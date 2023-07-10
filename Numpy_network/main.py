import numpy as np

if __name__ == '__main__':
    N,D_in,H,D_out = 64,1000,100,10 #N:批量大小，H：隐藏维度，D_in｜｜D_out：输入维度｜｜输出维度

    #创建随机输入输出数据
    x = np.random.randn(N,D_in)
    y = np.random.randn(N,D_out)

    # random the weight
    w1 = np.random.randn(D_in,H)
    w2 = np.random.randn(H,D_out)

    learning_rate = 0.01
    for t in range(10000):
    # forward ,calculate the y_predict
        h =x.dot(w1)
        h_relu = np.maximum(h,0)
        y_pred = h_relu.dot(w2)
        # calculate the loss
        loss = np.square(y_pred-y).sum()
        print(t,loss)
        #backward,calculate the grad of w1 & w2
        grad_y_pred = 2.0*(y_pred-y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu =grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h<0]=0
        grad_w1 = x.T.dot(grad_h)
        # update the weight
        w1 -= learning_rate *grad_w1
        w2 -= learning_rate *grad_w2

