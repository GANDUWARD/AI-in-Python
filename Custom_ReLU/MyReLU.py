import  torch

class MyReLU(torch.autograd.Function):
    # We can build the son class of torch.autograd to build our custom functions of autograd
    @staticmethod
    def forward(ctx,x):# accept a context object and a tensor include input then return a tensor include output
        #what's more,we can use the context object to save the object so that we can use it in backward
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx,grad_output):
        #accept a context object and a tensor include the grad of outputs in forward
        # we can search the bufferd data in context object and must calculate and return the loss grad
        # in forward's inputs
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_x[x<0]= 0
        return grad_x
