# No additional 3rd party external libraries are allowed
import numpy as np

def Adam(model,  train_X, train_y):
    #TODO
    layer= int(model.__class__.__name__[-1])
    W= model.W
    delta= 10**-8
    alpha=0.01
    beta1=0.9
    beta2=0.999
    if layer==1:
        for w in W:
            m= np.random.random(1).item()
            s= np.random.random(1).item()

            for r in range(5):
                m= beta1*m+ (1-beta1)*model.emp_loss_grad(train_X, train_y, w+beta1*m, layer)
                s= beta2*s+ (1-beta2)*np.dot(model.emp_loss_grad(train_X, train_y, w, layer).T,
                                             model.emp_loss_grad(train_X, train_y, w, layer))
                w= w-m*alpha/(np.sqrt(s)+delta)
        return W
    elif layer==2:
        for i in range(layer):
            m= np.random.random(1).item()
            s= np.random.random(1).item()

            for r in range(5):
                Wm=[w+beta1*m for w in W]
                m= beta1*m+(1-beta1)*model.emp_loss_grad(train_X, train_y, Wm, layer)[i]
                s= beta2*s+(1-beta2)*np.dot(model.emp_loss_grad(train_X, train_y, W, layer)[i].T,
                                            model.emp_loss_grad(train_X, train_y, W, layer)[i])
                W[i]= W[i]- m*alpha/(np.sqrt(s)+delta)
            return W
    raise NotImplementedError("Adam Not Implemented")