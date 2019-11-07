from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


class Softmax_Op(object):

    def __call__(self, x):
        """
        Softmax operation is a vector function: (N, ) -> (N, )

        Inputs:
        - X: A numpy array of shape (N, ) containing N class scores of 1 datapoint.
        """
        self.n = x.shape[0]
        self.x = x

        # For numerical stability of softmax
        x = x - np.max(x)

        # Use later in local grad calculation
        self.f1 = np.exp(x)                     # f1:  (N, 1) -> (N, 1)
        self.f2 = np.sum(self.f1)               # f2:  (N, 1) -> (1, 1)

        self.f = self.f1 / self.f2              # f3:f1(N, 1) -> (N, 1)
                                                #    f2(1, 1) -> (N, 1)
        return self.f
    

    def backprop(self, grad):
        """Analitical gradient of Softmax"""

        # Kronecker delta
        I = np.eye(self.n)

        local = self.f[:, None] * (I - self.f)

        return  np.matmul(local, grad)


class Softmax_Jacob(Softmax_Op):

    def backprop(self, grad):
        """
        Backpropagation along Softmax computational graph using chain rule.
        Full Jacobian matrixes are composed in this implementation

        N is number of classes. 

        Inputs:
        - grad: A numpy array of shape (N,) containing gradient wrt 
        Softmax output

        Returns:
        - Gradient wrt Softmax input X.
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

        # print('x', self.x)
        # print('f1', self.f1)
        # print('f2', self.f2)
        # print('f', self.f)
        # print('df3_f1', df3_f1)
        # print('df3_f2', df3_f2)
        # print('df2_f1', df2_f1)
        # print('df1_x', df1_x)
        # print('df2 CORRECT', df2)
        # print('df12 CORRECT', np.matmul(df2_f1.T, df2))
        # print('df13 CORRECT', np.matmul(df3_f1.T, df3))
        # print('df1 CORRECT', df1)

        return dX.flatten()


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
        softmax_op = Softmax_Op()
        pred_prob = softmax_op(scores)

        # cross-entropy loss
        loss -= np.log(pred_prob[y[i]])
        d_softmax_i = -1/pred_prob[y[i]] 
        
        d_softmax = np.zeros_like(scores)
        d_softmax[y[i]] = d_softmax_i

        d_scores = softmax_op.backprop(d_softmax)

        dW += np.outer(X[i], d_scores)

    loss /= num_train
    dW /= num_train
    
    # L2 regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

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

    scores = np.matmul(X, W)        #(N, C)

    # softmax
    softmax_op = Softmax_Op()
    pred_prob = softmax_op(scores)

    # cross-entropy loss
    loss -= np.log(pred_prob[y[i]])
    d_softmax_i = -1/pred_prob[y[i]] 
    
    d_softmax = np.zeros_like(scores)
    d_softmax[y[i]] = d_softmax_i

    d_scores = softmax_op.backprop(d_softmax)

    dW += np.outer(X[i], d_scores)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
