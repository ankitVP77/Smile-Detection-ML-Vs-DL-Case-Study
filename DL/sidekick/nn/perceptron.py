import numpy as np

class Perceptron:
    def __init__(self,N,alpha=0.01):
        self.alpha=alpha
        self.W=np.random.randn(N+1)/np.sqrt(N)

    def step(self,x):
        return 1 if x>0 else 0

    def fit(self,X,y,epochs=20):
        X=np.c_[X,np.ones((X.shape[0]))]
        for e in range(epochs):
            for(x,target) in zip(X,y):
                pred=self.step(np.dot(x,self.W))

                if pred!=target:
                    error=pred-target
                    self.W+=-self.alpha*error*x

    def predict(self,X,addbias=True):
        X=np.atleast_2d(X)

        if addbias:
            X=np.c_[X,np.ones((X.shape[0]))]

        return self.step(np.dot(X,self.W))