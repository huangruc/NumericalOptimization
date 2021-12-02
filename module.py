## 函数定义
import numpy as np
import matplotlib.pyplot as plt

class Watson(object):
    def __init__(self, m):
        self.m = m
        
    def r(self,x,i):
        n = len(x)
        first = 0
        for j in range(2,n+1):
            first += (j-1) * x[j-1] * (i/29)**(j-2)

        second = 0
        for j in range(1,n+1):
            second += x[j-1] * (i/29)**(j-1)

        return first-second**2-1
        
    def calculate(self,x):        
        result = 0
        for i in range(1,self.m+1):
            result += self.r(x,i)**2
        return result
    
    def grad_r(self,x,i):
        grad_list = []
        n = len(x)
        ti = i/29
        second = 0
        for j in range(1,n+1):
            second += x[j-1] * ti**(j-1)
            
        for j in range(1,n+1):
            if j == 1:
                grad_list.append(-2*second)
            else:
                grad_list.append( (j-1)*ti**(j-2)-2*second*ti**(j-1) )
                
        return grad_list
        
    def gradient(self,x):
        Grad = None
        for i in range(1,self.m+1):
            temp = 2*self.r(x,i)*np.array(self.grad_r(x,i))
            if i == 1:
                Grad = temp
            else:
                for i,v in enumerate(temp):
                    Grad[i] += v
        return np.array(Grad)
    
    def hessian(self,x):
        n = len(x)
        H = np.zeros([n,n])
        for m in range(1,self.m+1):
            ti = m/29
            h = np.zeros([n,n])
            for i in range(n):
                for j in range(i,n):
                    temp1 = self.grad_r(x,m)
                    temp2 = self.r(x,m)
                    if i == j:
                        h[i,j] = 2*temp1[i]*temp1[j]+2*temp2*(-2*ti**(i-1+j-1))
                    else:
                        h[j,i] = 2*temp1[i]*temp1[j]+2*temp2*(-2*ti**(i-1+j-1))
                        h[i,j] = 2*temp1[i]*temp1[j]+2*temp2*(-2*ti**(i-1+j-1))
            H += h
        return H
    
    def plot(self,x=[-5,5],y=[-5,5]):
        X = np.arange(x[0],x[-1],0.1)
        Y = np.arange(y[0],y[-1],0.1)
        Z = np.zeros((len(X),len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                Z[i,j] = self.calculate([X[i],Y[j]])
        fig, ax = plt.subplots(figsize=(5,5),dpi=100)
        plt.rcParams['axes.unicode_minus']=False
        CS = ax.contourf(X, Y, Z)
        ax.set_title('contour of 2-dimensional watson')

class DiscreteBoundaryValue(object):
    def __init__(self, m):
        self.m = m
        
    def init_point(self):
        h = 1 / (self.m+1)
        init = []
        for i in range(1,self.m+1):
            ti = i*h
            init.append(ti*(ti-1))
        return np.array(init)
    
    def r(self,x,i):
        n = len(x)
        h = 1 / (n+1)
        if i == 1:
            return 2*x[i-1] - x[i] + h**2 * (x[i-1]+i*h+1)**3 / 2
        elif i == n:
            return 2*x[i-1] - x[i-2] + h**2 * (x[i-1]+i*h+1)**3 / 2
        else:
            return 2*x[i-1] - x[i-2] -x[i] + h**2 * (x[i-1]+i*h+1)**3 / 2
    
    def calculate(self,x):
        result = 0
        for i in range(1,self.m+1):
            result += self.r(x,i)**2
        return result
    
    def grad_r(self,x,i):
        grad_list = []
        n = len(x)
        h = 1/(n+1)
        ti = i*h
        for j in range(1,n+1):
            if j<i-1 or j>i+1:
                grad_list.append(0.)
            elif j == i-1:
                grad_list.append(-1.)
            elif j == i+1:
                grad_list.append(-1.)
            else:
                grad_list.append( 2.+(x[j-1]+ti+1) ** 2 / 2 * 3 * h **2 )
                
        return grad_list
        
    def gradient(self,x):
        Grad = None
        for i in range(1,self.m+1):
            temp = 2*self.r(x,i)*np.array(self.grad_r(x,i))
            if i == 1:
                Grad = temp
            else:
                for i,v in enumerate(temp):
                    Grad[i] += v
        return np.array(Grad)
    
    def hessian(self,x):
        n = len(x)
        h = 1 / (n+1)
        H = np.zeros([n,n])
        for m in range(1,self.m+1):
            ti = m*h
            for i in range(n):
                H[i,i] += 3*h**2 * (x[i]+ti+1)
        return H

    def plot(self,x=[-5,5],y=[-5,5]):
        X = np.arange(x[0],x[-1],0.1)
        Y = np.arange(y[0],y[-1],0.1)
        Z = np.zeros((len(X),len(Y)))
        for i in range(len(X)):
            for j in range(len(Y)):
                Z[i,j] = self.calculate([X[i],Y[j]])
        fig, ax = plt.subplots(figsize=(5,5),dpi=100)
        plt.rcParams['axes.unicode_minus']=False
        CS = ax.contourf(X, Y, Z)
        ax.set_title('contour of discrete binary value')
        
def BasicNewton(x0,loss):
    maxIter = 1000
    eps = 1.e-4
    x = x0
    for i in range(1,maxIter+1):
        gradient = loss.gradient(x)
        hessian = loss.hessian(x)
        inv_hessian = np.linalg.inv(hessian)
        
        if abs(np.linalg.det(inv_hessian)) < 1.e-6:
            print('Warning: Hessian矩阵不可逆！')
            break  
        elif np.all(np.linalg.eigvals(inv_hessian) > 0): 
            d = np.inner(inv_hessian,-gradient)
            x = x+d 
        else:
            print('Warning: Hessian矩阵非正定！')
            print('\t iter:',i,' x:',x,' loss:',loss.calculate(x))
            d = np.inner(inv_hessian,-gradient)
            x = x+d

        if np.abs(loss.gradient(x)).max() < eps:
            break
    print('迭代结束-iter:',i,' x:',x,' loss:',loss.calculate(x))
    
def ExactLineSearch(x0,loss):
    maxIter = 1000
    eps = 1.e-4
    x = x0
    for i in range(1,maxIter+1):
        gradient = loss.gradient(x)
        hessian = loss.hessian(x)
        inv_hessian = np.linalg.inv(hessian)
        
        if abs(np.linalg.det(inv_hessian)) < 1.e-6:
            print('Warning: Hessian矩阵不可逆！')
            break  
        else:
            d = np.inner(inv_hessian,-gradient)
            step_size = -np.inner(gradient,d) / (np.inner(d,np.inner(hessian,d)))
            x = x + step_size*d 

        if np.abs(loss.gradient(x)).max() < eps:
            break
    print('迭代结束-iter:',i,' x:',x,' loss:',loss.calculate(x))
    
def StepSearch(loss,x,d,start=0.,end=3.,size=0.5,rho=0.2,sigma=0.5):
    '''根据Wolfe准则选择步长'''
    temp = []
    gradient = loss.gradient(x)
    for step in np.arange(start,end,size)[1:]:
        if loss.calculate(x+step*d) <= loss.calculate(x) + rho*np.inner(gradient,step*d) \
            and np.inner(loss.gradient(x+step*d),d) >= sigma*np.inner(gradient,d):
            temp.append(step)
    return temp

class Goldstein(object):
    def __init__(self,start=0.,end=3.,size=0.5,rho=0.25):
        self.start = start
        self.end = end
        self.size = size
        self.rho = rho       
        
    def search(self,loss,x,d):
        temp = []
        gradient = loss.gradient(x)
        for step in np.arange(self.start,self.end,self.size)[1:]:
            if loss.calculate(x+step*d) <= loss.calculate(x) + self.rho*np.inner(gradient,step*d) \
                and loss.calculate(x+step*d) >= loss.calculate(x) + (1-self.rho)*np.inner(gradient,step*d):
                temp.append(step)
        return temp

class Wolfe(object):
    def __init__(self,start=0.,end=3.,size=0.5,rho=0.2,sigma=0.5):
        self.start = start
        self.end = end
        self.size = size
        self.rho = rho
        self.sigma = sigma
        
    def search(self,loss,x,d):
        temp = []
        gradient = loss.gradient(x)
        for step in np.arange(self.start,self.end,self.size)[1:]:
            if loss.calculate(x+step*d) <= loss.calculate(x) + self.rho*np.inner(gradient,step*d) \
                and np.inner(loss.gradient(x+step*d),d) >= self.sigma*np.inner(gradient,d):
                temp.append(step)
        return temp
    
class StrongWolfe(object):
    def __init__(self,start=0.,end=3.,size=0.5,rho=0.2,sigma=0.5):
        self.start = start
        self.end = end
        self.size = size
        self.rho = rho
        self.sigma = sigma
        
    def search(self,loss,x,d):
        temp = []
        gradient = loss.gradient(x)
        for step in np.arange(self.start,self.end,self.size)[1:]:
            if loss.calculate(x+step*d) <= loss.calculate(x) + self.rho*np.inner(gradient,step*d) \
                and np.abs(np.inner(loss.gradient(x+step*d),d)) <= -self.sigma*np.inner(gradient,d):
                temp.append(step)
        return temp

def DampedNewton(x0,loss,method):
    maxIter = 20000
    eps = 1.e-6
    x = x0
    for i in range(1,maxIter+1):
        gradient = loss.gradient(x)
        hessian = loss.hessian(x)
        inv_hessian = np.linalg.inv(hessian)
        d = np.inner(inv_hessian,-gradient)
        
        if abs(np.linalg.det(inv_hessian)) < 1.e-6:
            print('Warning: Hessian矩阵不可逆！')
            break  
        else:
            step = method.search(loss,x,d)
            if len(step) == 0:
                step_size = 0.1
            else:
                step_size = np.random.choice(step)
            x = x + step_size*d 

        if np.abs(loss.gradient(x)).max() < eps:
            break
    print('迭代结束-iter:',i,' x:',x,' loss:',loss.calculate(x))
    
def MixNewton(x0,loss,eps1=0.2,eps2=0.1):
    maxIter = 10000
    eps = 1.e-4
    x = x0
    for i in range(1,maxIter+1):
        gradient = loss.gradient(x)
        hessian = loss.hessian(x)
        inv_hessian = np.linalg.inv(hessian)
        d = np.inner(inv_hessian,-gradient)
        
        if abs(np.linalg.det(inv_hessian)) < 1.e-6:
            print('Warning: Hessian矩阵奇异，采用负梯度方向')
            print('\t iter:',i,' x:',x,' loss:',loss.calculate(x))
            d = -gradient
        elif np.inner(gradient,d) > eps1*np.linalg.norm(gradient)*np.linalg.norm(d):
            print('Warning: Hessian矩阵非正定，方向取反')
            print('\t iter:',i,' x:',x,' loss:',loss.calculate(x))
            d = -d
        elif np.abs(np.inner(gradient,d)) <= eps2*np.linalg.norm(gradient)*np.linalg.norm(d):
            print('Warning: 迭代方向几乎与负梯度方向正交，采用负梯度方向')
            print('\t iter:',i,' x:',x,' loss:',loss.calculate(x))
            d = -gradient
            
        step = StepSearch(loss,x,d)
        if len(step) == 0:
            step_size = 0.01
        else:
            step_size = np.random.choice(step)
        x = x + step_size*d 

        #print('iter:',i,' x:',x,' loss:',loss.calculate(x))
        if np.abs(loss.gradient(x)).max() < eps:
            break
    print('迭代结束-iter:',i,' x:',x,' loss:',loss.calculate(x))
    
def SR1(x0,loss):
    maxIter = 1000
    eps = 1.e-4
    x = x0
    g = loss.gradient(x0)
    G = loss.hessian(x0)
    H = np.linalg.inv(G)
    for i in range(1,maxIter+1):
        if abs(np.linalg.det(H)) < 1.e-6:
            print('Warning:Hessian矩阵奇异！')
            break
        '''if np.all(np.linalg.eigvals(H) > 0) == False:
            print('Warning:矩阵非正定！')'''
        d = - np.inner(H,g)

        step = StepSearch(loss,x,d)
        if len(step) == 0:
            step_size = 0.01
        else:
            step_size = np.random.choice(step)

        s = step_size * d
        y = loss.gradient(x+s) - loss.gradient(x)
        u = s - np.inner(H,y).reshape(len(x),1)
        H += np.matmul(u,u.T) / np.inner(y,u.T)

        x = x + s
        g = loss.gradient(x)
        if loss.calculate(x) > 1.e+10:
            print('Warning:Hessian矩阵非正定导致溢出！')
            break
        if np.abs(g).max() < eps:
            break
        
    print('迭代结束-iter:',i,' x:',x,' loss:',loss.calculate(x))
    
def DFP(x0,loss):
    maxIter = 1000
    eps = 1.e-4
    x = x0
    n = len(x)
    g = loss.gradient(x0)
    G = loss.hessian(x0)
    H = np.linalg.inv(G)
    for i in range(1,maxIter+1):
        if abs(np.linalg.det(H)) < 1.e-6:
            print('Warning:Hessian矩阵奇异！')
            print('\t iter:',i,' x:',x,' loss:',loss.calculate(x))
            break
        else:
            d = - np.inner(H,g)

            step = StepSearch(loss,x,d)
            if len(step) == 0:
                step_size = 0.01
            else:
                step_size = np.random.choice(step)

            s = step_size * d
            y = loss.gradient(x+s) - loss.gradient(x)
            term1 = np.matmul(s.reshape(n,1),s.reshape(1,n)) / np.inner(s,y)
            term2 = - np.matmul(np.inner(H,y).reshape(n,1),np.inner(H,y).reshape(1,n)) / np.inner(y,np.inner(H,y))
            H = H + term1 + term2
            x = x + s
            g = loss.gradient(x)
            
            if np.abs(g).max() < eps:
                break
    print('迭代结束-iter:',i,' x:',x,' loss:',loss.calculate(x))
    
def BFGS(x0,loss):
    maxIter = 10000
    eps = 1.e-4
    x = x0
    n = len(x)
    g = loss.gradient(x0)
    B = loss.hessian(x0)
    for i in range(1,maxIter+1):
        if abs(np.linalg.det(B)) < 1.e-6:
            print('Warning:Hessian矩阵奇异！')
            print('\t iter:',i,' x:',x,' loss:',loss.calculate(x))
            break
        else:
            H = np.linalg.inv(B)
            d = - np.inner(H,g)

            step = StepSearch(loss,x,d)
            if len(step) == 0:
                step_size = 0.01
            else:
                step_size = np.random.choice(step)

            s = step_size * d
            y = loss.gradient(x+s) - loss.gradient(x)            
            term1 = np.matmul(y.reshape(n,1),y.reshape(1,n)) / np.inner(s,y)
            term2 = - np.matmul(np.inner(B,s).reshape(n,1),np.inner(B,s).reshape(1,n)) / np.inner(s,np.inner(B,s))
            B = B + term1 + term2
            x = x + s
            g = loss.gradient(x)
            
            if np.abs(g).max() < eps:
                break
    print('迭代结束-iter:',i,' x:',x,' loss:',loss.calculate(x))
    
def Broyden(x0,loss,phi):
    maxIter = 1000
    eps = 1.e-4
    x = x0
    n = len(x)
    g = loss.gradient(x0)
    B = loss.hessian(x0)
    H = H_DFP = H_BFGS = np.linalg.inv(B)
    for i in range(1,maxIter+1):
        if abs(np.linalg.det(H)) < 1.e-7:
            print('Warning:Hessian矩阵奇异！')
            break
        else:
            
            d = - np.inner(H,g)

            step = StepSearch(loss,x,d)
            if len(step) == 0:
                step_size = 0.01
            else:
                step_size = np.random.choice(step)

            s = step_size * d
            y = loss.gradient(x+s) - loss.gradient(x)
            term1 = np.matmul(s.reshape(n,1),s.reshape(1,n)) / np.inner(s,y)
            term2 = - np.matmul(np.inner(H,y).reshape(n,1),np.inner(H,y).reshape(1,n)) / np.inner(y,np.inner(H,y))
            H_DFP = H_DFP + term1 + term2
            
            term1 = (1+np.inner(y,np.inner(H,y))) / np.inner(y,s) * np.matmul(s.reshape(n,1),s.reshape(1,n)) / np.inner(s,y)
            term2 = - (np.matmul(np.matmul(s.reshape(n,1),y.reshape(1,n)),H) + np.matmul(H,np.matmul(y.reshape(n,1),s.reshape(1,n)))) / np.inner(s,y)
            H_BFGS = H_BFGS + term1 + term2
            
            H = (1-phi)*H_DFP + phi*H_BFGS
            x = x + s
            g = loss.gradient(x)
            
            if np.abs(g).max() < eps:
                break
    print('迭代结束-iter:',i,' x:',x,' loss:',loss.calculate(x))
    
