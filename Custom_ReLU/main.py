import torch
import MyReLU as M
if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N,D_in,H,D_out = 64,1000,100,10
    x= torch.randn(N,D_in,device=device)
    y = torch.randn(N,D_out,device =device)

    w1 =torch.randn(D_in,H,device=device,requires_grad=True)
    w2 =torch.randn(H,D_out,device=device,requires_grad=True)

    learning_rate = 1e-6
    for t in range(500):
        y_pred = M.MyReLU.apply(x.mm(w1).mm(w2))
        loss = (y_pred-y).pow(2).sum()
        print(t,loss.item())
        loss.backward()
        with torch.no_grad():
            w1 -= learning_rate*w1.grad
            w2 -= learning_rate*w2.grad

            w1.grad.zero_()
            w2.grad.zero_()
