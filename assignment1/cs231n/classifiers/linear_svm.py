from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W) # [D, ] [DxC] = [C, ]
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] + 1 - correct_class_score # note delta = 1
            if margin > 0:
                loss += margin
                # There are two dependency chains for dl/dW

                # dl/dW = dl/d_margin * d_margin/d_scores[j] * d_scores[j]/dW 
                #       + dl/d_margin * d_margin/scores[y[i]] * scores[y[i]]/dW 

                # dl/dW = 1 * 1 * d_scores[j]/dW 
                #       + 1 * (-1) * scores[y[i]]/dW 

                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Hinge loss part
    dW = dW / num_train

    # Regularization part
    df = reg * 1        # f = reg * x
    df = 1 * df         # f = np.sum(x)
    dW += 2*W * df      # f =  W * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = np.matmul(X, W)   #(N, C)
    
    num_samples = X.shape[0]
    all_ = np.arange(num_samples)
    correct_scores = scores[all_, y]

    margins = 1 + scores - correct_scores[:, np.newaxis]
    margins[all_, y] = 0

    # Hinge loss
    loss += np.sum(np.maximum(margins, 0)) / num_samples

    # Regularization
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    
    # Add scores with positive margins
    margin_mask = (margins > 0).astype(int)                 #(N, C) 

    # Substruct correct score for each positive margin
    # margin_mask[i] = [0, 1, 1, 1, -3, 0, 0, ...]
    margin_mask[all_, y] = -np.sum(margin_mask, axis=1)
    
    for i in all_:
        dW += np.outer(X[i], margin_mask[i])
    dW /= num_samples

    # Regularization: Grads
    dW += 2 * reg * W

    return loss, dW
