from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


class Softmax(object):

    def __call__(self, x):
        """
        Softmax operation is a vector function: (N, ) -> (N, )

        Inputs:
        - X: A numpy array of shape (N, ) containing N class scores of 1 datapoint.
        """
        self.x = x
        self.n = x.shape[0]

        # For numerical stability of softmax
        x = x - np.max(x)

        # Use later in local grad calculation
        self.f1 = np.exp(x)                     # f1:  (N, 1) -> (N, 1)
        self.f2 = np.sum(self.f1)               # f2:  (N, 1) -> (1, 1)

        self.f = self.f1 / self.f2              # f3:f1(N, 1) -> (N, 1)
                                                #    f2(1, 1) -> (N, 1)
        return self.f
    

    def backward(self, grad):
        """Analitical gradient of Softmax"""

        # Kronecker delta
        I = np.eye(self.n)

        local = self.f[:, None] * (I - self.f)

        return  np.matmul(local, grad).squeeze()


class VecSoftmax(Softmax):

    def __call__(self, x):
        
        """
        Softmax operation is a vector function: (N, ) -> (N, )

        Inputs:
        - X: A numpy array of shape (N, C) containing N vectors with C scores.

        Returns:
        - f = softmax(X). A numpy array of size (N, C)
        """
        
        if x.ndim == 1:
            x = x[None, :]
        
        self.x_shape = x.shape

        # For numerical stability of softmax
        x -= x.max(axis=1, keepdims=True)

        # Calculate Softmax and store intermediates to use later in backprob
        self.f1 = np.exp(x)                     
        self.f2 = self.f1.sum(axis=1, keepdims=True)       
        self.f = self.f1 / self.f2              

        # fn:  (in_dims) -> (out_dims) 
        # f1:  (N, 1) -> (N, 1)
        # f2:  (N, 1) -> (1, 1)
        # f3:  (N, 1) and (1, 1) -> (N, 1)

        return self.f.squeeze()    
    
    def backward(self, grad_in):

        """
        Analitical gradient of Softmax
        """

        n = self.x_shape[0]
        c = self.x_shape[1]

        # Kronecker delta. 1(i=j)
        I = np.zeros((n, c, c))
        di = (slice(n), range(c), range(c))
        I[di] = 1
        
        grad_local = self.f[:, :, None] * (I - self.f[:, None, :])

        grad_out = np.matmul(grad_local, grad_in[:, :, None])

        return grad_out.squeeze()


class JacobSoftmax(Softmax):

    def backward(self, grad):
        """
        Backpropagation along Softmax computational graph using chain rule.
        Full Jacobian matrixes are composed in this implementation

        N is number of classes. 

        Inputs:
        - grad: A numpy array of shape (N,) containing gradient wrt 
        Softmax output

        Returns:
        - Gradient wrt input X.
        """ 

        # Softmax computational graph:
        # x <- f1 <- f2 <- f3 <-... loss
        #         \________/

        # Jacobians (Local gradients) of shape (out, in)                              
        df3_f1 = np.diag(np.tile(1 / self.f2, self.n)) # (N, N)      
        df3_f2 = - self.f1 / self.f2**2                # (N, 1)
        df2_f1 = np.ones_like(self.f1)                 # (1, N)
        df1_x = np.diag(self.f1)                       # (N, N)
        
        # Reshape local gradients before matmul
        df3 = grad[:, None]
        df2_f1 = df2_f1.reshape(1, self.n)
        df3_f2 = df3_f2.reshape(self.n, 1)
        
        # Chain rule formula:
        # d_in = J.T * d_out 
        df2 = np.matmul(df3_f2.T, df3)     # (1, 1) = (N, 1).T * (N, 1) 
        df1 = np.matmul(df2_f1.T, df2)     # (N, 1) = (1, N).T * (1, 1)
        df1 += np.matmul(df3_f1.T, df3)    # (N, 1) = (N, N).T * (N, 1)
        dX = np.matmul(df1_x.T, df1)       # (N, 1) = (N, N).T * (N, 1)

        return dX.squeeze()


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    for i in range(num_train):
        scores = np.matmul(X[i], W)  

        # softmax
        softmax = JacobSoftmax()
        pred_prob = softmax(scores)

        # cross-entropy loss
        loss -= np.log(pred_prob[y[i]])
        dsoftmax_i = -1/pred_prob[y[i]] 
        
        dsoftmax = np.zeros_like(scores)
        dsoftmax[y[i]] = dsoftmax_i

        dscores = softmax.backward(dsoftmax)

        dW += np.outer(X[i], dscores)

    loss /= num_train
    dW /= num_train
    
    # L2 regularization
    loss += reg * np.sum(W**2)
    dW += reg * 2*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    scores = np.matmul(X, W)        

    # softmax
    softmax = VecSoftmax()
    pred_prob = softmax(scores) + 1e-15    

    one_hot_y = np.eye(num_classes)[y]

    # cross-entropy loss
    cross_entropy_loss = -np.sum(one_hot_y * np.log(pred_prob))
    cross_entropy_loss /= num_train

    # L2 regularizarion
    reg_loss = reg * np.sum(W**2)

    loss = cross_entropy_loss + reg_loss

    dloss = 1/num_train
    dsoftmax = -one_hot_y/pred_prob * dloss
    dscores = softmax.backward(dsoftmax) 

    dW += reg * 2*W
    dW += np.matmul(X.T, dscores)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
