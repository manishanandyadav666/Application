# No additional 3rd party external libraries are allowed
import numpy as np


def SGD(model,  train_X, train_y, lr=0.1, R=100):
    #Updated_Weights = None
    #TODO

    #
    layer= int(model.__class__.__name__[-1])
    W= model.W


    def get_random_Wj(w):
        W_flatten= np.squeeze(w.flatten.reshape(1, -1))
        random_choice_j= np.random.choice(W_flatten, size=1, replace= False)
        random_choice_j_value= random_choice_j[0]
        mask= np.ma.masked_equal(w, random_choice_j_value).mask.astype('float')
        W_j= np.multiply(w, mask)
        return W_j

    if layer==1:
        W_cloned= np.copy(W)
        for _ in range(R):
            W_j= get_random_Wj(W_cloned)
            grad_W_j= model.emp_loss_grad(train_X, train_y, W_j, layer)

            W= W-lr*grad_W_j
            return W

    elif layer==2:
        for _ in range(R):
            w_j1= get_random_Wj(W[0])
            w_j2= get_random_Wj(W[1])
            W_j= [w_j1] +[w_j2]

            grad_W_j= model.emp_loss_grad(train_X, train_y, W_j, layer)
            W[0]= W[0]-lr*grad_W_j[0]
            W[1]= W[1]-lr*grad_W_j[1]
            return W


    #return Updated_Weights
    raise NotImplementedError("SGD not implemented")
      