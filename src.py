import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
import time
import mlflow

class LogisticRegression:
    
    def __init__(self, k, n, method, alpha = 0.001, max_iter=5000,l=0.1,use_penalty=bool):
        self.k = k
        self.n = n
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.l=l
        self.use_penalty=use_penalty
    
    def fit(self, X, Y):
        self.W = np.random.rand(self.n, self.k)
        self.losses = []
        
        if self.method == "batch":
            start_time = time.time()
            for i in range(self.max_iter):
                loss, grad =  self.gradient(X, Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "minibatch":
            start_time = time.time()
            batch_size = int(0.3 * X.shape[0])
            for i in range(self.max_iter):
                ix = np.random.randint(0, X.shape[0]) #<----with replacement
                batch_X = X[ix:ix+batch_size]
                batch_Y = Y[ix:ix+batch_size]
                loss, grad = self.gradient(batch_X, batch_Y)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        elif self.method == "sto":
            start_time = time.time()
            list_of_used_ix = []
            for i in range(self.max_iter):
                idx = np.random.randint(X.shape[0])
                while i in list_of_used_ix:
                    idx = np.random.randint(X.shape[0])
                X_train = X[idx, :].reshape(1, -1)
                Y_train = Y[idx]
                loss, grad = self.gradient(X_train, Y_train)
                self.losses.append(loss)
                self.W = self.W - self.alpha * grad
                
                list_of_used_ix.append(i)
                if len(list_of_used_ix) == X.shape[0]:
                    list_of_used_ix = []
                if i % 500 == 0:
                    print(f"Loss at iteration {i}", loss)
            print(f"time taken: {time.time() - start_time}")
            
        else:
            raise ValueError('Method must be one of the followings: "batch", "minibatch" or "sto".')
        
        
    def gradient(self, X, Y): #-------------------Add penatlry(Ridge Regression) to gradient value
        m = X.shape[0]
        h = self.h_theta(X, self.W)
        loss = - np.sum(Y*np.log(h)) / m
        error = h - Y
        grad = self.softmax_grad(X, error)
        if self.use_penalty:
            W_reg=self.W.copy()
            loss = loss+ (self.l / (2.0 * m)) * np.sum(W_reg ** 2)  #--------- Penalty  
            grad = grad+(self.l / m) * W_reg                           
        return loss, grad

    def softmax(self, theta_t_x):
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):  
        return  X.T @ error 

    def h_theta(self, X, W):
        '''
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        '''
        return self.softmax(X @ W)
    
    def predict(self, X_test):
        return np.argmax(self.h_theta(X_test, self.W), axis=1)
    
    def plot(self):
        plt.plot(np.arange(len(self.losses)) , self.losses, label = "Train Losses")
        plt.title("Losses")
        plt.xlabel("epoch")
        plt.ylabel("losses")
        plt.legend()
    def accuracy(self,y_pred,y_test):
        self.y_pred=y_pred
        self.y_test=y_test
        correct=0
        F=0
        for i in y_pred==y_test:
            if i==True:
                correct=correct+1
            else:
                F=F+1
        total=correct/(correct+F)
        print(f'Accuracy: {round(total,2)}')
        mlflow.log_metric("Accuracy", total)
        # Initialize weights of each classe 
        wclasse=[]
        self.wclasse=wclasse
        classes,count=np.unique(y_test,return_counts=True)
        for i in range(k):
            i=count[i]/sum(count)
            wclasse.append(i)

    def precision(self):
        Y_test_encoded = np.zeros((m, k))
        for each_class in range(k):
            cond = self.y_test==each_class
            Y_test_encoded[np.where(cond), each_class] = 1
        Y_pred_encoded = np.zeros((m, k))
        for each_class in range(k):
            cond = self.y_pred==each_class
            Y_pred_encoded[np.where(cond), each_class] = 1
        self.Y_pred_encoded=Y_pred_encoded
        self.Y_test_encoded=Y_test_encoded
        TP=[]
        FP=[]
        TN=[]
        FN=[]
        self.TP=TP
        self.FP=FP
        self.TN=TN
        self.FN=FN
        Tprecision=[]
        self.Tprecision=Tprecision
        for i in range(k):
            TP1=(Y_pred_encoded[:,i]==1) & (Y_test_encoded[:,i]==1) 
            TP.append(np.size(TP1[TP1]))
            FP1=(Y_pred_encoded[:,i]==1) & (Y_test_encoded[:,i]==0)
            FP.append(np.size(FP1[FP1]))
            TN1=(Y_pred_encoded[:,i]==0) & (Y_test_encoded[:,i]==0)
            TN.append(np.size(TN1[TN1]))
            FN1=(Y_pred_encoded[:,i]==0) & (Y_test_encoded[:,i]==1)
            FN.append(np.size(FN1[FN1]))
        for i in range(k):
            Tprecision1=TP[i]/(TP[i]+FP[i])
            Tprecision.append(Tprecision1)
            print(f'Precision for {i} class: {Tprecision1:.2f}.')
        return Tprecision
    def recall(self):
        Trecall=[]
        self.Trecall=Trecall
        for i in range(k):
            Trecall1=self.TP[i]/(self.TP[i]+self.FN[i])
            Trecall.append(Trecall1)
            print(f'Recall for {i} class: {Trecall1:.2f}.')
        return Trecall
    def f1score(self):
        Tf1score=[]
        self.Tf1score=Tf1score
        for i in range(k):
            Tf1score1=(2*self.Tprecision[i]*self.Trecall[i])/(self.Trecall[i]+self.Tprecision[i])
            Tf1score.append(Tf1score1)
            print(f'F1 score for {i} class: {Tf1score1:.2f}.')
            
    def macroprecision(self):
        Tmacroprecision=sum(self.Tprecision)/k
        self.Tmacroprecision=Tmacroprecision
        print(f'Macro precision: {Tmacroprecision:.2f}')
        mlflow.log_metric("Macroprecision", Tmacroprecision)
        return Tmacroprecision
    def macrorecall(self):
        Tmacrorecall=sum(self.Trecall)/k
        self.Tmacrorecall=Tmacrorecall
        print(f'Macro precision: {Tmacrorecall:.2f}')
        mlflow.log_metric("Macrorecall", Tmacrorecall)
        return Tmacrorecall
    def macrof1(self):
        Tmacrof1=sum(self.Tf1score)/k
        self.Tmacrof1=Tmacrof1
        print(f'Macro f1 score: {Tmacrof1:.2f}')
        mlflow.log_metric("Macrof1", Tmacrof1)
        return Tmacrof1
    
    def wprecision(self):
        wprecision=np.array(self.Tprecision).T@np.array(self.wclasse)
        print(f'Weighted precision: {wprecision:.2f}')
        mlflow.log_metric("Weighted precision", wprecision)
            
    def wrecall(self):
        wprecall=np.array(self.Trecall).T@np.array(self.wclasse)
        print(f'Weighted recall: {wprecall:.2f}')
        mlflow.log_metric("Weighted recall", wprecall)
    def wf1(self):
        wf1=np.array(self.Tf1score).T@np.array(self.wclasse)
        print(f'Weighted f1: {wf1:.2f}')
        mlflow.log_metric("Weighted f1", wf1)
        