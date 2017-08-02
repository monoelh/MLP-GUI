#######################################################################################
# Little app for learning python and some neural nets
# Author: Manuel Hass
# 2017
# 
#######################################################################################



################################# IMPORTS ##########################################
try:
    from appJar import gui
except ImportError:
    print ('ERROR -> MODULE MISSING: appJar(Tkinter)  http://appjar.info/ ')
try:    
    import numpy as np
    numpy = np
except ImportError:
    print ('ERROR -> MODULE MISSING: numpy ')
try:
    import time
    import datetime
except ImportError:
    print ('ERROR -> MODULE MISSING: time  AND/OR datetime ')
try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import matplotlib.gridspec as gridspec
except ImportError:
    print ('ERROR -> MODULE MISSING: matplotlib ')
try:
    from PIL import Image
except ImportError:
    print ('ERROR -> MODULE MISSING: PIL ')
import glob, os

################################ creating data ############################################
def create_data(grid=True):
    if grid:
        ############# 2D MESHGRID ############
        R = np.linspace(-2, 2, 128, endpoint=True) # with 128x128
        A,B = np.meshgrid(R,R)
        G = [] 
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                G += [[A[i][i],A[i][j]]]
        G = np.array(G)
        
        # GRID MAX
        R = np.linspace(-2, 2, 1000, endpoint=True) # with 1000x1000
        A,B = np.meshgrid(R,R)
        G_max = [] 
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                G_max += [[A[i][i],A[i][j]]]
        G_max = np.array(G_max)
    ###################### XOR data (with thrid class in the center) ################
    p1 = np.zeros([120,2])
    p2 = np.zeros([120,2])
    p3 = np.zeros([240,2])
    mu1 = np.array([-.75,.75])
    mu2 = np.array([.75,-.75])
    mu3 = np.array([.75,.75])
    mu4 = np.array([-.75,-.75])
    var = np.diag([0.1,0.1])
    choice1 = np.random.rand(120)
    choice2 = np.random.rand(120)
    for i in range(p1.shape[0]):
        if choice1[i]>.5 :
            p1[i] = np.random.multivariate_normal(mu1,var)
        else:
            p1[i] = np.random.multivariate_normal(mu2,var)
        if choice2[i]>.5 :
            p2[i] = np.random.multivariate_normal(mu3,var)
        else:
            p2[i] = np.random.multivariate_normal(mu4,var)
    for i in range(p3.shape[0]):       
        p3[i] = np.random.multivariate_normal([0,0],var)
    P = np.concatenate((p1,p2,p3),0) 
    t = np.ones(480)
    t[120:240] = -1
    t[240:] = 3
    if grid: return P,t,G,G_max
    else: return P,t
def plotting(P,G,t,G_lab,er): 
    # 2D scatter from -2 to 2, with errorbar
    fig1 = plt.figure(figsize=(9,9))
    Grr = gridspec.GridSpec(6, 1)
    fig1.add_subplot(Grr[:5, :])
    plt.scatter(G[:,0],G[:,1],s=25,alpha=.5,c=G_lab,cmap=str(app.getEntry("colormap")),edgecolor='None')
    plt.colorbar()
    plt.scatter(P[:,0],P[:,1],s=25,alpha=1,c=t,cmap='bwr')
    #plt.scatter(P[:,0],P[:,1],s=10,alpha=1,c=pred,cmap='hot',edgecolor='None')
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.grid()
    plt.title('Training data')
    fixi = fig1.add_subplot(Grr[5, :])
    plt.plot(range(er.shape[0]),er,'blue',antialiased=True,linewidth=1)
    #plt.gca().patch.set_facecolor('black')
    plt.xlim(0,er.shape[0])
    fixi.spines['top'].set_visible(False)
    fixi.spines['right'].set_visible(False)
    plt.yticks(np.round([np.min(er),np.max(er)],2))
    fig1.show()
    return None
def plot_imgs(X,y,ind=112):
    # plotting first 112 imagaes from Test data, and error plot
    if (np.sqrt(X.shape[1]) != float(int(np.sqrt(X.shape[1])))): return None
    X = X.reshape(X.shape[0],int(np.sqrt(X.shape[1])),int(np.sqrt(X.shape[1])))
    figx = plt.figure(figsize=(20,10))
    for i in range(ind):
        figx.add_subplot(7,16,i+1)
        plt.imshow(X[i],cmap = 'Blues');
        plt.axis('off')
        plt.title("%i  p:%i"%(i,y[i]))
    plt.suptitle('Testset predictions')    
    figx.show()
    
    return None
def plot_er(er,er2=[],msg=''):
    akkk = plt.figure(figsize=(12,5))
    akk = akkk.add_subplot(121)
    plt.grid()
    plt.xlabel('iterations')
    plt.ylabel('error')
    akk.plot(range(er.shape[0]),er,'blue',label='training error')
    akk.scatter(0,er[0],s=.001,label=msg)
    plt.xlim(0.,er.shape[0])
    plt.ylim(0.,1.1*np.max(er))
    if (er2!=[]):
        akk.plot(range(er.shape[0]),er2,'red',label='test error')
    plt.legend(loc='center left', bbox_to_anchor=(1, .5))
    akkk.show()
    return None

###################### error functions ####################################################
###binary error function (cross entropy)
def bce(ya,yta,dev=False):  ############ work in progress
    if (dev==True):
        return (yta-ya)/((1-yta)*yta)
    return -(np.sum(ya*np.log(yta)+(1.-yta)*np.log(1.-yta))/(yta.shape[0]*2.0))
###Quadratic error function
def qef(ya,yta,dev=False):
    if (dev==True):
        return (yta-ya)
    return np.sum((yta-ya)**2)/(yta.shape[0]*2.0)
###Psudo Huber Loss
def phl(y,yt,dev=False,delta=1.):
    a = (yt-y)
    if (dev==True):
        return  a/( np.sqrt(a**2/delta**2 +1) ) 
    return np.sum((delta**2)*(np.sqrt(1+(a/delta)**2)-1)/(yt.shape[0]*2.0))

###################### regularization ####################################################
### L2 norm
def L2(lam,a):  
    return lam*a
### L1 norm
def L1(lam,a):
    return lam*np.sign(a)
### max-out norm (not working so great, do better)
def MX(lam,a): 
    return lam*np.sign(a[np.argmax(np.linalg.norm(a,axis=1))])
### forget norm for autoencoder (work in progress...)
def FN(lam,a,lam2,W):
    #toDO get weights (1 or 3 hiddenlayer) from before
    # loading weights in here is too slow, find a way to reference it without giving it as an argument
    # W has to be the loaded weight matching a
    return lam*a + lam2*(W-a)
###################### activation functions ####################################################
##### robuts logistic transfer fct ##### WORKING 
# sigmoid -1/1
def f_lgtr(a,dev=False):
    if (dev==True):
        return (1-np.tanh(a/2.)**2)/2.
    return  (np.tanh(a/2.)+1)/2.
'''
#### logistic transfer fct ##### ~working, make robust
# sigoimd 0/1
def f_lgtr(a,dev=False):
    if (dev==True):
        return (1.0/(1+np.exp(-a))*(1-1.0/(1+np.exp(-a))))
    return  1.0/(1+np.exp(-a))
''' and None
#### stochastic transfer fct ##### WORKING
# sigoimd 0/1
def f_stoch(a,dev=False):
    if (dev==True):
        return np.zeros(a.shape)  
    x = f_lgtr(a,dev=False)
    rand = np.random.random(x.shape)
    return  np.where(rand < x,1,0)
#### tanh transfer fct ##### WORKING
# sigmoid -1/1
def f_tanh(a,dev=False):
    if (dev==True):
        return (1-np.tanh(a)**2)
    return  np.tanh(a)
#### atan transfer fct ##### WORKING
# sigmoid -pi/pi
def f_atan(a,dev=False):
    if (dev==True):
        return (1/(a**2+1))
    return  np.arctan(a)
##### softplus transfer fct ##### ~working, make robust
# softplus 0/a
def f_sp(a,dev=False):
    if (dev==True):
        return np.exp(a)/(np.exp(a)+1.)
    return  np.log(np.exp(a)+1.)
##### RELU ##### WORKING     
# max 0/a
def f_relu(a,dev=False):
    if (dev==True):
        return (np.sign(a)+1 / 2.0) ### korrigieren!!!
    return  np.maximum(0.0,a)
##### Bent ident ##### WORKING 
def f_bi(a,dev=False):
    if (dev==True):
         return a / ( 2.0*np.sqrt(a**2+1) ) + 1
    return  (np.sqrt(a**2+1)-1)/2.0 + a
##### ident ##### WORKING 
def f_iden(a,dev=False):
    if (dev==True):
         return np.ones(a.shape)
    return  a
##### binary function ##### WORKING
# 0/1
def f_bin(a,dev=False):
    if (dev==True):
         return np.zeros(a.shape) 
    return  np.sign(f_relu(a))
##### rounded identity ##### ~working
def f_rint(a,dev=False):
    if (dev==True):
         return np.zeros(a.shape)
    return  np.round(a,0)

###################### MLP #################################################################
def MLP1(x,w,v,f=f_tanh,f2=f_iden):
    x1 = np.vstack((x.T,np.ones(x.shape[0]))).T
    H = (np.dot(x1,w.T)).T
    s = f(H)
    s1 = np.vstack((s,np.ones(s.shape[1])))
    y_ = f2(np.dot(s1.T,v.T))
    return y_
def MLP2(x,w,v,u,f=f_tanh,f2=f_iden):
    x1_ = np.vstack((x.T,np.ones(x.shape[0]))).T
    H_ = (np.dot(x1_,w.T)).T
    s_ = f(H_)
    s1_ = np.vstack((s_,np.ones(s_.shape[1])))
    H2_ = (np.dot(s1_.T,v.T))
    s2_ = f(H2_.T)
    s2_ = np.vstack((s2_,np.ones(s2_.shape[1]).T))
    H3_ = np.dot(s2_.T,u.T)
    y__ = f2(H3_)
    return y__
def MLP3(x,w,v,u,z,f=f_tanh,f2=f_iden):
    x1_ = np.vstack((x.T,np.ones(x.shape[0]))).T
    H_ = (np.dot(x1_,w.T)).T
    s_ = f(H_)
    s1_ = np.vstack((s_,np.ones(s_.shape[1])))
    H2_ = (np.dot(s1_.T,v.T))
    s2_ = f(H2_.T)
    s2_ = np.vstack((s2_,np.ones(s2_.shape[1]).T))
    H3_ = np.dot(s2_.T,u.T)
    s3_ = f(H3_)
    s3_ = np.vstack((s3_.T,np.ones(s3_.T.shape[1]).T))
    H4_ = np.dot(s3_.T,z.T)
    y__ = (f2(H4_))
    return y__ 

###################### TRAIN MLP #################################################################
def MMLP1(xt,y,out,node1=3,steps=500,f=f_tanh,f2=f_iden,err=qef,
    method='Adam',beta1=0.9,beta2=0.99,eta=0.05,reg=L2,lam=0.,
    testx=[],testy=[],w=[],v=[],
    rec=False,stoch=1.,drop1=1.,drop2=1.):
    ### training 
    outneurons = out   
    dream = np.array([])
    if rec:
        ############# 2D MESHGRID ############
        R = np.linspace(-2, 2, 128, endpoint=True)
        A,B = np.meshgrid(R,R)
        G = [] 
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                G += [[A[i][i],A[i][j]]]
        G = np.array(G)
    if (out == 1):       
        x1 = np.vstack((xt.T,np.ones(xt.shape[0]))).T  
        X1 = np.copy(x1)
        yn = np.copy(y)  
        if w == [] : w = np.random.uniform(-1.,1,(node1,x1.shape[1]))
        if v == [] : v = np.random.uniform(-1.,1,node1+1)
        mW = np.random.uniform(0.,1,w.shape)
        mV = np.random.uniform(0.,1,v.shape)
        vW = np.random.uniform(0.,1,w.shape)
        vV = np.random.uniform(0.,1,v.shape)      
        step_error = np.array([])
        step_error2 = np.array([])
        stat = time.perf_counter()
        a=0
        for i in range(steps):
            if stoch != 0.:
                j = np.random.choice(range(X1.shape[0]),int(X1.shape[0]*stoch),replace=False)
                x1 = X1[j]
                y = yn[j]
            if (i >= a+np.rint(steps/10.)): print('------------',i,'----of----',steps,'------ iterations in ',time.perf_counter()-stat,'s');a+=np.rint(steps/10.)
            
            ########### dropout now !! go get it!! yeah!! do it!!!
            mask0 = np.random.choice([0, 1], size=(x1.shape[1],), p=[1-drop1, drop1])

            H =  (np.dot(mask0 *x1,w.T)).T
            mask1 = np.random.choice([0, 1], size=(node1,), p=[1-drop2, drop2])

            s = (mask1 * f(H).T).T
            s1 = np.vstack((s,np.ones(s.shape[1])))
            #dout = np.random.choice([0, 1], size=(s1.shape[1],), p=[drop2, 1-drop2])
            #s1_ = dout * s1
            y_ = (f2(np.dot(s1.T,v.T)))
            dV = -1.0/xt.shape[0] * np.dot(s1,err(y,y_,dev=True).T) - reg(lam,v)
            dW = -1.0/xt.shape[0] * np.dot(np.diag(v[1:]),(mask1*f(H,True).T).T).dot(np.diag(err(y,y_,dev=True))).dot((mask0*x1)) - reg(lam,w)
            eps = 0.000001
            b1 = beta1#0.9 # 0.95,  0.99 , 0.999
            b2 = beta2#0.999 # 0.95,  0.99 , 0.999 ### decay rate if RMSprop
            mW = b1*mW + (1-b1)*dW
            mV = b1*mV + (1-b1)*dV
            vW =  b2*vW + (1-b2)* dW**2 
            vV = b2*vV + (1-b2)* dV**2 
            if(method=='RMSprop'):
                w += eta* dW / (np.sqrt(vW) +eps) 
                v += eta* dV / (np.sqrt(vV) +eps)
            if(method=='normal'):
                w += eta* dW 
                v += eta* dV  
            if (method=='Adam'):
                w += eta* mW / (np.sqrt(vW) +eps)
                v += eta* mV / (np.sqrt(vV) +eps)
            yt = y_
            error = err(y,yt,dev=False)
            #messi += '\n'+ (error)
            step_error = np.append(step_error,error)
            if rec :
                dreamstep = MLP1(G,w,v,f,f2)
                dream = np.append(dream,dreamstep)
            if ((testx !=[])and(testy !=[])):
                predtest = MLP1(testx,w,v,f,f2)
                error2 = err(testy,predtest,False)
                step_error2 = np.append(step_error2,error2)
    else:
        x1 = np.vstack((xt.T,np.ones(xt.shape[0]))).T 
        X1 = np.copy(x1)
        yn = np.copy(y)   
        if w == [] : w = np.random.uniform(-1.,1,(node1,x1.shape[1]))
        if v == [] : v = np.random.uniform(-1.,1,(outneurons,node1+1))
        mW = np.random.uniform(0.,1,w.shape)
        mV = np.random.uniform(0.,1,v.shape)
        vW = np.random.uniform(0.,1,w.shape)
        vV = np.random.uniform(0.,1,v.shape)
        step_error = np.array([])
        step_error2 = np.array([])
        stat = time.perf_counter()
        a=0
        for i in range(steps):
            if stoch != 0.:
                j = np.random.choice(range(X1.shape[0]),int(X1.shape[0]*stoch),replace=False)
                x1 = X1[j]
                y = yn[j]
            if (i >= a+np.rint(steps/10.)): print('------------',i,'----of----',steps,'------ iterations in ',time.perf_counter()-stat,'s');a+=np.rint(steps/10.)
            #mask = 
            mask0 = np.random.choice([0, 1], size=(x1.shape[1],), p=[1-drop1, drop1])

            H =  (np.dot(mask0 *x1,w.T)).T
            #H = (np.dot(x1,w.T)).T
            mask1 = np.random.choice([0, 1], size=(node1,), p=[1-drop2, drop2])
            s = (mask1 * f(H).T).T

            s1 = np.vstack((s,np.ones(s.shape[1])))
            H1 = np.dot(s1.T,v.T)
            y_ = (f2(H1))
            d_v = (err(y,y_,dev=True)*f2(H1,True)).T
            dV = -1.0/xt.shape[0] * (np.dot(d_v,s1.T)) - reg(lam,v)
            d_w = np.dot(v.T[1:],d_v)*(mask1*f(H,True).T).T 
            dW = -1.0/xt.shape[0] * (np.dot(d_w,mask0*x1)) - reg(lam,w)
            eps = 0.000001
            b1 = beta1#0.9 # 0.95,  0.99 , 0.999
            b2 = beta2#0.999 # 0.95,  0.99 , 0.999 ### decay rate if RMSprop
            mW = b1*mW + (1-b1)*dW
            mV = b1*mV + (1-b1)*dV
            vW =  b2*vW + (1-b2)* dW**2 
            vV = b2*vV + (1-b2)* dV**2 
            if(method=='RMSprop'):
                w += eta* dW / (np.sqrt(vW) +eps)
                v += eta* dV / (np.sqrt(vV) +eps)
            if(method=='normal'):
                w += eta* dW 
                v += eta* dV 
            if (method=='Adam'):
                w += eta* mW / (np.sqrt(vW) +eps)
                v += eta* mV / (np.sqrt(vV) +eps)
            yt = y_
            error = err(y,yt,dev=False)
            step_error = np.append(step_error,error)
            if ((testx !=[])and(testy !=[])):
                predtest = MLP1(testx,w,v,f,f2)
                error2 = err(testy,predtest,dev=False)
                step_error2 = np.append(step_error2,error2)
    if rec : 
        a = int(steps)
        b = np.sqrt(G.shape[0]).astype(int)
        dream = dream.reshape(a,b,b)
    return drop1*w,drop2*v,step_error,step_error2,dream 
def MMLP2(xt,y,out,node1=4,node2=3,steps=500,f=f_tanh,f2=f_iden,err=qef,
    method='Adam',beta1=0.9,beta2=0.99,eta=0.05,reg=L2,lam=0.,
    testx=[],testy=[],w=[],v=[],u=[],rec=False,stoch=1.,drop1=1.,drop2=1.):
    ### training 
    outneurons = out
    dream = np.array([])
    if rec:
        ############# 2D MESHGRID ############
        R = np.linspace(-2, 2, 128, endpoint=True)
        A,B = np.meshgrid(R,R)
        G = [] 
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                G += [[A[i][i],A[i][j]]]
        G = np.array(G)
    if (out==1):
        step_error = np.array([])
        step_error2 = np.array([])
        x1 = np.vstack((xt.T,np.ones(xt.shape[0]))).T    
        X1 = np.copy(x1)
        yn = np.copy(y)
        if w == [] : w = np.random.uniform(-1.,1,(node1,x1.shape[1]))
        if v == [] : v = np.random.uniform(-1.,1,(node2,node1+1))
        if u == [] : u = np.random.uniform(-1.,1,node2+1) 
        cW = np.ones(w.shape)
        cV = np.ones(v.shape)
        cU = np.ones(u.shape)
        mW = np.random.uniform(0.,1,w.shape)
        mV = np.random.uniform(0.,1,v.shape)
        mU = np.random.uniform(0.,1,u.shape)
        vW = np.random.uniform(0.,1,w.shape)
        vV = np.random.uniform(0.,1,v.shape)
        vU = np.random.uniform(0.,1,u.shape)
        stat = time.perf_counter()
        a=0
        for i in range(steps):
            if stoch != 0.:
                j = np.random.choice(range(X1.shape[0]),int(X1.shape[0]*stoch),replace=False)
                x1 = X1[j]
                y = yn[j]
            if (i >= a+np.rint(steps/10.)): print('------------',i,'----of----',steps,'------ iterations in ',time.perf_counter()-stat,'s');a+=np.rint(steps/10.)
            mask0 = np.random.choice([0, 1], size=(x1.shape[1],), p=[1-drop1, drop1])
            mask1 = np.random.choice([0, 1], size=(node1,), p=[1-drop2, drop2])
            mask2 = np.random.choice([0, 1], size=(node2,), p=[1-drop2, drop2])
            H =  (np.dot(mask0*x1,w.T)).T

            s = ( mask1*f(H).T).T
            s1 = np.vstack((s,np.ones(s.shape[1])))
            H2 = (np.dot(s1.T,v.T))
            s2 = (mask2*f(H2.T).T).T
            s2 = np.vstack((s2,np.ones(s2.shape[1]).T))
            H3 = np.dot(s2.T,u.T)
            y_ = f2(H3)
            dU = -1.0/xt.shape[0] * (np.dot(err(y,y_,dev=True),s2.T)) - reg(lam,u)

            du = np.diag(err(y,y_,dev=True))## np.diag(np.dot(np.diag(z[1:]),f(H3.T,True)).dot(dz).T.sum(axis=1))
            dV = -1.0/xt.shape[0] * np.dot(np.diag(u[1:]),(mask2*f(H2.T,True).T).T).dot(du).dot(s1.T) - reg(lam,v)
            dv = np.diag(np.dot(np.diag(u[1:]),(mask2*f(H2.T,True).T).T).dot(du).T.sum(axis=1))     
            dW = -1.0/xt.shape[0] * np.dot(np.diag(v.T[1:].sum(axis=1)),(mask1*f(H,True).T).T).dot(dv).dot(mask0*x1) - reg(lam,w)

            ############# adaptive 'learning rate' #########
            ### Adam  // good with tanh, relu not so. init mX,vX with 1 is better than 0
            eps = 0.000001
            b1 = beta1 #0.9 # 0.95,  0.99 , 0.999
            b2 = beta2 #0.999 # 0.95,  0.99 , 0.999 ### decay rate if RMSprop
            mW = b1*mW + (1-b1)*dW
            mV = b1*mV + (1-b1)*dV
            mU = b1*mU + (1-b1)*dU
            vW =  b2*vW + (1-b2)* dW**2 
            vV = b2*vV + (1-b2)* dV**2 
            vU =  b2*vU + (1-b2)* dU**2
            if(method=='RMSprop'):
                w += eta* dW / (np.sqrt(vW) +eps)
                v += eta* dV / (np.sqrt(vV) +eps)
                u += eta* dU / (np.sqrt(vU) +eps)
            if(method=='normal'):
                w += eta* dW 
                v += eta* dV 
                u += eta* dU 
            if (method=='Adam'):
                w += eta* mW / (np.sqrt(vW) +eps)
                v += eta* mV / (np.sqrt(vV) +eps)
                u += eta* mU / (np.sqrt(vU) +eps)
            ## error: calculate different loss functions here: quadratic, log-cosh, huber
            yt = y_
            error = err(y,yt,dev=False)
            step_error = np.append(step_error,error)
            if rec :
                dreamstep = MLP2(G,w,v,u,f,f2)
                dream = np.append(dream,dreamstep)
            if ((testx !=[])and(testy !=[])):
                predtest = MLP2(testx,w,v,u,f,f2)
                error2 = err(testy,predtest,False)
                step_error2 = np.append(step_error2,error2)
    else:
        step_error = np.array([])
        step_error2 = np.array([])
        x1 = np.vstack((xt.T,np.ones(xt.shape[0]))).T 
        X1 = np.copy(x1)
        yn = np.copy(y)   
        if w == [] : w = np.random.uniform(-1.,1,(node1,x1.shape[1]))
        if v == [] : v = np.random.uniform(-1.,1,(node2,node1+1))
        if u == [] : u = np.random.uniform(-1.,1,(outneurons,node2+1))
        cW = np.ones(w.shape)
        cV = np.ones(v.shape)
        cU = np.ones(u.shape)
        mW = np.random.uniform(0.,1,w.shape)
        mV = np.random.uniform(0.,1,v.shape)
        mU = np.random.uniform(0.,1,u.shape)
        vW = np.random.uniform(0.,1,w.shape)
        vV = np.random.uniform(0.,1,v.shape)
        vU = np.random.uniform(0.,1,u.shape)
        stat = time.perf_counter()
        a=0
        for i in range(steps):
            if stoch != 0.:
                j = np.random.choice(range(X1.shape[0]),int(X1.shape[0]*stoch),replace=False)
                x1 = X1[j]
                y = yn[j]
            if (i >= a+np.rint(steps/10.)): print('------------',i,'----of----',steps,'------ iterations in ',time.perf_counter()-stat,'s');a+=np.rint(steps/10.)
            mask0 = np.random.choice([0, 1], size=(x1.shape[1],), p=[1-drop1, drop1])
            mask1 = np.random.choice([0, 1], size=(node1,), p=[1-drop2, drop2])
            mask2 = np.random.choice([0, 1], size=(node2,), p=[1-drop2, drop2])
            H = (np.dot(mask0*x1,w.T)).T
            s = (mask1*f(H).T).T
            s1 = np.vstack((s,np.ones(s.shape[1])))
            H2 = (np.dot(s1.T,v.T))
            s2 = (mask2*f(H2.T).T).T
            s2 = np.vstack((s2,np.ones(s2.shape[1]).T))
            H3 = np.dot(s2.T,u.T)
            y_ = f2(H3)
            d_u = (err(y,y_,dev=True)*f2(H3,True)).T
            dU = -1.0/xt.shape[0] * (np.dot(d_u,s2.T)) - reg(lam,u)
            d_v = np.dot(u.T[1:],d_u)*(mask2*f(H2,True)).T
            dV = -1.0/xt.shape[0] * (np.dot(d_v,s1.T)) - reg(lam,v)
            d_w = np.dot(v.T[1:],d_v)*(mask1*f(H,True).T).T 
            dW = -1.0/xt.shape[0] * (np.dot(d_w,mask0*x1)) - reg(lam,w)
            ############# adaptive 'learning rate' #########
            ### Adam  // good with tanh, relu not so. init mX,vX with 1 is better than 0
            eps = 1e-10
            b1 = beta1 #0.9 # 0.95,  0.99 , 0.999
            b2 = beta2 #0.999 # 0.95,  0.99 , 0.999 ### decay rate if RMSprop
            mW = b1*mW + (1-b1)*dW
            mV = b1*mV + (1-b1)*dV
            mU = b1*mU + (1-b1)*dU
            vW =  b2*vW + (1-b2)* dW**2 
            vV = b2*vV + (1-b2)* dV**2 
            vU =  b2*vU + (1-b2)* dU**2
            if(method=='RMSprop'):
                w += eta* dW / (np.sqrt(vW) +eps)
                v += eta* dV / (np.sqrt(vV) +eps)
                u += eta* dU / (np.sqrt(vU) +eps)
            if(method=='normal'):
                w += eta* dW 
                v += eta* dV 
                u += eta* dU 
            if (method=='Adam'):
                w += eta* mW / (np.sqrt(vW) +eps)
                v += eta* mV / (np.sqrt(vV) +eps)
                u += eta* mU / (np.sqrt(vU) +eps)
            ## error: calculate different loss functions here: quadratic, log-cosh, huber
            yt = y_
            error = err(y,yt,dev=False)
            #messi += '\n'+ (error)
            step_error = np.append(step_error,error) 
            if ((testx !=[])and(testy !=[])):
                predtest = MLP2(testx,w,v,u,f,f2)
                error2 = err(testy,predtest,False)
                step_error2 = np.append(step_error2,error2)
    if rec : 
        a = int(steps)
        b = np.sqrt(G.shape[0]).astype(int)
        dream = dream.reshape(a,b,b)
    return drop1*w,drop2*v,drop2*u,step_error,step_error2,dream
def MMLP3(xt,y,out,node1=4,node2=3,node3=5,steps=500,f=f_tanh,f2=f_iden,err=qef,
    method='Adam',beta1=0.9,beta2=0.99,eta=0.05,reg=L2,lam=0.,testx=[],
    testy=[],w=[],v=[],u=[],z=[],rec=False,stoch=0.,drop1=1.,drop2=1.):  
    ### training 
    outneurons = out
    dream = np.array([])
    if rec:
        ############# 2D MESHGRID ############
        R = np.linspace(-2, 2, 128, endpoint=True)
        A,B = np.meshgrid(R,R)
        G = [] 
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                G += [[A[i][i],A[i][j]]]
        G = np.array(G)
    if( out == 1 ):
        step_error = np.array([])
        step_error2 = np.array([])
        x1 = np.vstack((xt.T,np.ones(xt.shape[0]))).T 
        X1 = np.copy(x1)
        yn = np.copy(y) 
        if w == []: w = np.random.uniform(-1.,1,(node1,x1.shape[1]))
        if v == []: v = np.random.uniform(-1.,1,(node2,node1+1))
        if u == []: u = np.random.uniform(-1.,1,(node3,node2+1)) 
        if z == []: z = np.random.uniform(-1.,1,node3+1)
        cW = np.ones(w.shape)
        cV = np.ones(v.shape)
        cU = np.ones(u.shape)
        mW = np.random.uniform(0.,1,w.shape)
        mV = np.random.uniform(0.,1,v.shape)
        mU = np.random.uniform(0.,1,u.shape)
        mZ = np.random.uniform(0.,1,z.shape)
        vW = np.random.uniform(0.,1,w.shape)
        vV = np.random.uniform(0.,1,v.shape)
        vU = np.random.uniform(0.,1,u.shape)
        vZ = np.random.uniform(0.,1,z.shape)
        stat = time.perf_counter()
        a=0
        for i in range(steps):

            if stoch != 0.:
                j = np.random.choice(range(X1.shape[0]),int(X1.shape[0]*stoch),replace=False)
                x1 = X1[j]
                y = yn[j]

            if (i >= a+np.rint(steps/10.)): print('------------',i,'----of----',steps,'------ iterations in ',time.perf_counter()-stat,'s');a+=np.rint(steps/10.)
            mask0 = np.random.choice([0, 1], size=(x1.shape[1],), p=[1-drop1, drop1])
            mask1 = np.random.choice([0, 1], size=(node1,), p=[1-drop2, drop2])
            mask2 = np.random.choice([0, 1], size=(node2,), p=[1-drop2, drop2])
            mask3 = np.random.choice([0, 1], size=(node3,), p=[1-drop2, drop2])
            H = (np.dot(mask0*x1,w.T)).T
            s = (mask1*f(H).T).T
            s1 = np.vstack((s,np.ones(s.shape[1])))
            H2 = (np.dot(s1.T,v.T))
            s2 = (mask2*f(H2.T).T).T
            s2 = np.vstack((s2,np.ones(s2.shape[1]).T))
            H3 = np.dot(s2.T,u.T)
            s3 = (mask3*f(H3))
            s3 = np.vstack((s3.T,np.ones(s3.T.shape[1]).T))
            H4 = np.dot(s3.T,z.T)
            y_ = (f2(H4))   
            dZ = -1.0/xt.shape[0] * (np.dot(err(y,y_,dev=True),s3.T)) - reg(lam,z)
            dz = np.diag(err(y,y_,dev=True))
            dU = -1.0/xt.shape[0] * np.dot(np.diag(z[1:]),(mask3*f(H3.T,True).T).T).dot(dz).dot(s2.T) - reg(lam,u)
            du = np.diag(np.dot(np.diag(z[1:]),(mask3*f(H3.T,True).T).T).dot(dz).T.sum(axis=1))
            dV = -1.0/xt.shape[0] * np.dot(np.diag(u.T[1:].sum(axis=1)),(mask2*f(H2.T,True).T).T).dot(du).dot(s1.T) - reg(lam,v)
            dv = np.diag(np.dot(np.diag(u.T[1:].sum(axis=1)),(mask2*f(H2.T,True).T).T).dot(du).T.sum(axis=1))
            dW = -1.0/xt.shape[0] * np.dot(np.diag(v.T[1:].sum(axis=1)),(mask1*f(H,True).T).T).dot(dv).dot(mask0*x1) - reg(lam,w)
            ############# adaptive 'learning rate' #########
            ### Adam  // good with tanh, relu not so. init mX,vX with 1 is better than 0
            eps = 0.00001
            b1 = beta1 #0.9 # 0.95,  0.99 , 0.999
            b2 = beta2 #0.95 # 0.95,  0.99 , 0.999 ### decay rate if RMSprop
            mW = b1*mW + (1-b1)*dW
            mV = b1*mV + (1-b1)*dV
            mU = b1*mU + (1-b1)*dU
            mZ = b1*mZ + (1-b1)*dZ
            vW =  b2*vW + (1-b2)* dW**2 
            vV = b2*vV + (1-b2)* dV**2 
            vU =  b2*vU + (1-b2)* dU**2
            vZ =  b2*vZ + (1-b2)* dZ**2
            if(method=='RMSprop'):
                w += eta* dW / (np.sqrt(vW) +eps)
                v += eta* dV / (np.sqrt(vV) +eps)
                u += eta* dU / (np.sqrt(vU) +eps)
                z += eta* dZ / (np.sqrt(vZ) +eps)
            if(method=='normal'):
                w += eta* dW 
                v += eta* dV 
                u += eta* dU 
                z += eta* dZ 
            if (method=='Adam'):
                w += eta* mW / (np.sqrt(vW) +eps)
                v += eta* mV / (np.sqrt(vV) +eps)
                u += eta* mU / (np.sqrt(vU) +eps)
                z += eta* mZ / (np.sqrt(vZ) +eps)
            ## error: calculate different loss functions here: quadratic, log-cosh, huber
            yt = y_
            error = err(y,yt,dev=False)
            step_error = np.append(step_error,error)
            if rec :
                dreamstep = MLP3(G,w,v,u,z,f,f2)
                dream = np.append(dream,dreamstep)
            if ((testx !=[])and(testy !=[])):
                predtest = MLP3(testx,w,v,u,z,f,f2)
                error2 = err(testy,predtest,False)
                step_error2 = np.append(step_error2,error2)
    else:
        step_error2 = np.array([])
        step_error = np.array([])
        x1 = np.vstack((xt.T,np.ones(xt.shape[0]))).T    
        X1 = np.copy(x1)
        yn = np.copy(y)

        '''
        ##### XAVIER INTITIALIZATION #####
         ## RELU   
         w_init =   np.random.multivariate_normal(0.,[2./x1.shape[1]],(node1,x1.shape[1])) 
         ## tanh   
         w_init =   np.random.multivariate_normal(0.,[2./(x1.shape[1]+outneurons)],(node1,x1.shape[1])) 
        '''

        if w == []: w = np.random.uniform(-1.,1,(node1,x1.shape[1]))
        if v == []: v = np.random.uniform(-1.,1,(node2,node1+1))
        if u == []: u = np.random.uniform(-1.,1,(node3,node2+1)) 
        if z == []: z = np.random.uniform(-1.,1,(outneurons,node3+1))
        cW = np.ones(w.shape)
        cV = np.ones(v.shape)
        cU = np.ones(u.shape)
        mW = np.random.uniform(0.,1,w.shape)
        mV = np.random.uniform(0.,1,v.shape)
        mU = np.random.uniform(0.,1,u.shape)
        mZ = np.random.uniform(0.,1,z.shape)
        vW = np.random.uniform(0.,1,w.shape)
        vV = np.random.uniform(0.,1,v.shape)
        vU = np.random.uniform(0.,1,u.shape)
        vZ = np.random.uniform(0.,1,z.shape)
        stat = time.perf_counter()
        a=0
        for i in range(steps):

            if stoch != 0.:
                j = np.random.choice(range(X1.shape[0]),int(X1.shape[0]*stoch),replace=False)
                x1 = X1[j]
                y = yn[j]
            if (i >= a+np.rint(steps/10.)): print('------------',i,'----of----',steps,'------ iterations in ',time.perf_counter()-stat,'s');a+=np.rint(steps/10.)
            mask0 = np.random.choice([0, 1], size=(x1.shape[1],), p=[1-drop1, drop1])
            mask1 = np.random.choice([0, 1], size=(node1,), p=[1-drop2, drop2])
            mask2 = np.random.choice([0, 1], size=(node2,), p=[1-drop2, drop2])
            mask3 = np.random.choice([0, 1], size=(node3,), p=[1-drop2, drop2])
            H = (np.dot(mask0*x1,w.T)).T
            s = (mask1*f(H).T).T
            s1 = np.vstack((s,np.ones(s.shape[1])))
            H2 = (np.dot(s1.T,v.T))
            s2 = (mask2*f(H2.T).T).T
            s2 = np.vstack((s2,np.ones(s2.shape[1]).T))
            H3 = np.dot(s2.T,u.T)
            s3 = (mask3*f(H3))
            s3 = np.vstack((s3.T,np.ones(s3.T.shape[1]).T))
            H4 = np.dot(s3.T,z.T)
            y_ = (f2(H4))   
            d_z = (err(y,y_,dev=True)*f2(H4,True)).T
            dZ = -1.0/xt.shape[0] * (np.dot(d_z,s3.T)) - reg(lam,z)
            d_u = np.dot(z.T[1:],d_z)*(mask3*f(H3,True)).T
            dU = -1.0/xt.shape[0] * (np.dot(d_u,s2.T)) - reg(lam,u)
            d_v = np.dot(u.T[1:],d_u)*(mask2*f(H2,True)).T
            dV = -1.0/xt.shape[0] * (np.dot(d_v,s1.T)) - reg(lam,v)
            d_w = np.dot(v.T[1:],d_v)*(mask1*f(H,True).T).T 
            dW = -1.0/xt.shape[0] * (np.dot(d_w,mask0*x1)) - reg(lam,w)
            ############# adaptive 'learning rate' #########
            ### Adam  // good with tanh, relu not so. init mX,vX with 1 is better than 0
            eps = 0.00001
            b1 = beta1 #0.9 # 0.95,  0.99 , 0.999
            b2 = beta2 #0.95 # 0.95,  0.99 , 0.999 ### decay rate if RMSprop
            mW = b1*mW + (1-b1)*dW
            mV = b1*mV + (1-b1)*dV
            mU = b1*mU + (1-b1)*dU
            mZ = b1*mZ + (1-b1)*dZ
            vW =  b2*vW + (1-b2)* dW**2 
            vV = b2*vV + (1-b2)* dV**2 
            vU =  b2*vU + (1-b2)* dU**2
            vZ =  b2*vZ + (1-b2)* dZ**2
            if(method=='RMSprop'):
                w += eta* dW / (np.sqrt(vW) +eps)
                v += eta* dV / (np.sqrt(vV) +eps)
                u += eta* dU / (np.sqrt(vU) +eps)
                z += eta* dZ / (np.sqrt(vZ) +eps)
            if(method=='normal'):
                w += eta* dW 
                v += eta* dV 
                u += eta* dU 
                z += eta* dZ 
            if (method=='Adam'):
                w += eta* mW / (np.sqrt(vW) +eps)
                v += eta* mV / (np.sqrt(vV) +eps)
                u += eta* mU / (np.sqrt(vU) +eps)
                z += eta* mZ / (np.sqrt(vZ) +eps)
            ## error: calculate different loss functions here: quadratic, log-cosh, huber
            yt = y_
            error = err(y,yt,dev=False)
            step_error = np.append(step_error,error)
            if ((testx !=[])and(testy !=[])):
                predtest = MLP3(testx,w,v,u,z,f,f2)
                error2 = err(testy,predtest,False)
                step_error2 = np.append(step_error2,error2)
    if rec : 
        a = int(steps)
        b = np.sqrt(G.shape[0]).astype(int)
        dream = dream.reshape(a,b,b)         
    return drop1*w,drop2*v,drop2*u,drop2*z,step_error,step_error2,dream
 
###################### GUI activity ############################################################################################################################
# function called by pressing the buttons
def home(btn):
    app.setMessage("mess",hometext)
def cho(choose):
    if choose=="A":
        a = app.openBox(title=None, dirName=None, fileTypes=None, asFile=False)
        app.setEntry("Training data: ", str(a))
    if choose=="B":
        a = app.openBox(title=None, dirName=None, fileTypes=None, asFile=False)
        app.setEntry("Training label: ", str(a))
    if choose=="C":
        a = app.openBox(title=None, dirName=None, fileTypes=None, asFile=False)
        app.setEntry("Test data / Img: ", str(a))
    if choose=="D":
        a = app.openBox(title=None, dirName=None, fileTypes=None, asFile=False)
        app.setEntry("Test label: ", str(a))
def press(btn):
    if btn=="Quit":
        app.stop()
    else:
        # get/set random seed
        npseed = int(app.getEntry("seed: "))
        np.random.seed(npseed)

        print('-[NEW RUN]----- MMLP started at --- ',datetime.datetime.now().isoformat(),' ------------------ ')
        start_import = time.perf_counter()
        P,t,G,G_max = create_data()
        Ph,th = create_data(False)
        autoencoder = False
        if (app.getCheckBox("Autoencoder")):
            autoencoder = True
        clas = False
        if (app.getCheckBox("classification")):
            clas = True
        G_ = G # meshgrid
        TRAINDAT = P # training data
        TRAINLAB = t # training label
        TESTDAT = Ph # test data
        TESTLAB = th

        ################################################# check for apples input ####################################################################################################
        if (app.getEntry("Training data: ") == "apples"):    
            p1 = np.zeros([100,2])
            p2 = np.zeros([100,2])
            mu1 = np.array([-.25,-.25])
            mu2 = np.array([.25,.25])
            var = np.diag([.25,.5])
            vari = np.array([[0.314,-.125],[-.125,.334]])
            for i in range(p1.shape[0]):
                    p1[i] = np.random.multivariate_normal(mu1,vari*.4)
                    p2[i] = np.random.multivariate_normal(mu2,vari*.4)
            TRAINDAT = np.concatenate((p1,p2),0) 
            for i in range(p1.shape[0]):
                    p1[i] = np.random.multivariate_normal(mu1,vari*.4)
                    p2[i] = np.random.multivariate_normal(mu2,vari*.4)
            TESTDAT = np.concatenate((p1,p2),0)
            t = np.ones(200)
            t[100:] = 0
            TESTLAB = t
            TRAINLAB = t



        ################ 1-out-of-c-code ######################################################################
        if (clas):
            iterate =  (np.unique(t)) 
            halle = []
            halle2 = []
            for i in iterate:
                halle +=  [np.where(t==i,1,0)]
                halle2 +=  [np.where(TESTLAB==i,1,0)]
            expY = np.array(halle)
            expY = np.vstack((expY[:100],expY[100:]))
            yx = np.array([np.argmax(x) for x in expY.T])
            TRAINLAB = expY.T # training label

            expYh = np.array(halle2)
            expYh = np.vstack((expYh[:100],expYh[100:]))
            yxh = np.array([np.argmax(x) for x in expYh.T])
            yh = expYh # test label
            TESTLAB = yh.T

            # set outneurons according to classes 
            outneurons = TRAINLAB.shape[1]
            #TRAINLAB= y
        else:
            outneurons = 1

        ################################################ check if data input #####################################################################################################
        if ( (app.getEntry("Test data / Img: ") != "") and (app.getEntry("Training data: ") != "")): 
            import csv
            with open(app.getEntry("Test data / Img: "), 'r') as f:
                reader = csv.reader(f)
                dat = list(reader)
            dat = np.array(dat)
            dat = dat.T
            dat = dat.astype(float)
            
            with open(app.getEntry("Training data: "), 'r') as f:
                reader = csv.reader(f)
                X = list(reader)
            X = np.array(X)
            X = X.T
            X = X.astype(float)
            
            with open(app.getEntry("Training label: "), 'r') as f:
                reader = csv.reader(f)
                y = list(reader)
            y = np.array(y)
            y = y.T            
            if not autoencoder:
                y = y[:,0]   # important line for multioutput regression CHANGEX
            y = y.astype(float)
            #print ('train label shape ---- shape ',y.shape) ### here it's doing what it should-> shape 1500/64
            TRAINDAT = X # training data
            TESTDAT = dat # test data
            TRAINLAB = y 
            #'''#  get multi output regression right CHANGEX
            if autoencoder:
                try:
                    #print ('trainlab shape ',TRAINLAB.shape)
                    outneurons = TRAINLAB.shape[1]
                    #print ('outneurons shape stuff ',outneurons)
                except ValueError:
                    print('oh nonononono not again....')
                
            #'''
            if ( (app.getEntry("Test label: ") != "") ):
                
                with open(app.getEntry("Test label: "), 'r') as f:
                    readerx = csv.reader(f)
                    TESTLAB = list(readerx)
                TESTLAB = np.array(TESTLAB)
                TESTLAB = TESTLAB.T
                TESTLAB = TESTLAB.astype(float)
                TESTLAB = TESTLAB.reshape(-1)
                TESTLAB_ = np.copy(TESTLAB)                
            else:
                TESTLAB = []              
            if (clas):
                eiterate =  (np.unique(TESTLAB)) 
                ehalle = []
                for i in eiterate:
                    ehalle +=  [np.where(TESTLAB==i,1,0)]
                
                eexpY = np.array(ehalle)
                eexpY = np.vstack((eexpY[:100],eexpY[100:]))                    
                TESTLAB = eexpY.T # test label

                eiterate =  (np.unique(TRAINLAB)) 
                ehalle = []
                for i in eiterate:
                    ehalle +=  [np.where(TRAINLAB==i,1,0)]
                
                eexpY = np.array(ehalle)
                eexpY = np.vstack((eexpY[:100],eexpY[100:]))                    
                TRAINLAB = eexpY.T # test label

                outneurons = TRAINLAB.shape[1]
                #print ('classi testcaste ',TRAINLAB.shape)
            else:
                if not autoencoder: outneurons = 1    #CHANGEX 
                #print('testcase .... code work in progress')
            if ( (app.getEntry("Test label: ") == "") ): TESTLAB = []
        ############################################ testing for image input ################################################################################################################
        if ( (app.getEntry("Test data / Img: ") != "") and (app.getEntry("Training data: ") == "")):
            size = 32, 32
            for infile in glob.glob(app.getEntry("Test data / Img: ")):
                file, ext = os.path.splitext(infile)
                im = Image.open(infile)
                im.thumbnail(size, Image.ANTIALIAS)
                im.save("temp64.png", "png")
            img = matplotlib.image.imread('temp64.png')
            R = np.linspace(-1., 1., img.shape[0], endpoint=True)
            A,B = np.meshgrid(R,R)
            C = [] 
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    C += [[A[i][i],A[i][j]]]
            C = np.array(C)
            C = C.reshape((img.shape[0],img.shape[0],2))
            C = C.reshape(img.shape[0]*img.shape[0],2)
            img = img[:,:,1]
            img = img.reshape(-1)                                               
            TRAINDAT = C # training data
            TESTDAT = C # test data
            TRAINLAB = img # training label
            TESTLAB = []    
        end_import = time.perf_counter()
        time_import = end_import-start_import
        print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> ','loading data DONE in ',time_import,' seconds')

        ################################################ get config ########################################################################################
        hiddenL = int(app.getOptionBox("Layer")) # nr of hidden layers
        node1 = int(app.getEntry("layer 1 nodes"))   # nr of nodes in 1st hidden layer
        node2 = int(app.getEntry("layer 2 nodes")  ) # nr of nodes in 2nd hidden layer
        node3 = int(app.getEntry("layer 3 nodes") )# nr of nodes in 3rd hidden layer
        f_xx = eval(app.getOptionBox("Transfer function")  )    # activation function (for all neurons)
        f2 = eval(app.getOptionBox("Activation function")     )   # output function 
        erf  = eval(app.getOptionBox("Error function")      )   # error/loss function (qef: quadratic error fct,   phl: pseudo Huber loss)  ##### add the linerity parameter #####
        bprg = str(app.getOptionBox("Gradient descent ") )    # backpropagation/grad descent algorithm ('Adam','RMSprop','normal')
        beta1 = float(app.getEntry("b1") )       # ---->  smoothing parameter (Adam)
        beta2 = float(app.getEntry("b2")  )       # ---->  decay rate (RMSprop,Adam)
        eta = float(app.getEntry("learning rate"))          # initial learning rate
        iterations = int(app.getEntry("iterations"))  # nr of training steps / frames
        regularizer = float(app.getEntry("regularization")) # lambda for L2 weights decay
        sgd = float(app.getEntry("SGD mini-batch"))
        rglz = eval(app.getOptionBox(" "))
        drout1 = 1.-np.array(eval(app.getEntry("dropout")))[0]
        drout2 = 1.-np.array(eval(app.getEntry("dropout")))[1]

        messi = ''
        msg = ''
        msg += '\n'+('##### MLP configuration #####')
        msg += '\n'+('--seed: '+str(npseed)) 
        msg += '\n'+('Transfer function: '+str(str(f_xx)[10:17]))        
        msg += '\n'+('Activation function: '+str(str(f2)[10:17]))
        msg += '\n'+('Error/Loss function: '+str(str(erf)[10:14]))
        msg += '\n'+('Hidden layer: '+str(hiddenL))
        if hiddenL == 1:
            msg += '\n'+('# nodes in 1st hlayer: '+str(node1))
        if hiddenL == 2:
            msg += '\n'+('# nodes in 1st hlayer: '+str(node1))
            msg += '\n'+('# nodes in 2nd hlayer: '+str(node2))
        if hiddenL == 3:
            msg += '\n'+('# nodes in 1st hlayer: '+str(node1))    
            msg += '\n'+('# nodes in 2nd hlayer: '+str(node2))
            msg += '\n'+('# nodes in 3rd hlayer: '+str(node3))
        msg += '\n'+('Learning steps: '+str(iterations))
        msg += '\n'+('Gradient descent: '+str(bprg))
        if  sgd != 0. :
            msg += '\n'+('SGD mini-batch: '+str(sgd))
        if (bprg=='Adam'):
            msg += '\n'+('beta1: '+str((beta1)))
        msg += '\n'+('beta2: '+str(beta2))
        msg += '\n'+('learning rate: '+str(eta))
        msg += '\n'+(str(app.getOptionBox(" "))+' regularization: '+str(regularizer))
        if (drout1 != 1.)or(drout2 != 1.):
            msg += '\n'+('input dropout: '+str((1-drout1)*100)+str(' %'))
            msg += '\n'+('layer dropout: '+str((1-drout2)*100)+str(' %'))
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  PCA / LDA for dim reduction  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #'''
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!! NEW STUFF!! HOT!! GET IT NOW!!  (also do LDA and use svd for PCA ffs 11elf!)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # PCA
        if (app.getCheckBox("  ")):
            scale = int(app.getEntry("PCA"))
            msg += '\n'+('----- PCA reduced data from '+str(TRAINDAT.shape[1])+' dim to '+str(scale)+' dim' )
            TESTDAT = np.array(TESTDAT)
            a = np.copy(TRAINDAT).mean(0)
            b = np.copy(TESTDAT).mean(0)
            c = np.copy(TESTDAT).mean(0)
            TRAINDAT = TRAINDAT - TRAINDAT.mean(0)
            TESTDAT = TESTDAT - TESTDAT.mean(0)
            C = (1./TRAINDAT.shape[0]) * np.dot(TRAINDAT.T,TRAINDAT)
            eW,eV = np.linalg.eig(C)
            ind = (np.argsort(eW))[::-1]
            eW = eW[ind].real 
            eV = eV[ind].real
            ax = plt.figure()
            a = ax.add_subplot(111)
            #plt.bar(range(eW.shape[0]-1),eW[1:]/np.linalg.norm(eW[1:]),0.3)
            compsum = np.array([np.sum(eW[x+1:]) for x in range(eW.shape[0]-1)])
            plt.bar(range(eW.shape[0]),eW/np.sum(eW),edgecolor=None) #compsum/compsum[0]
            #print (compsum.shape)
            compsum = np.append(compsum,[0.])
            #print (compsum.shape)
            print ('PCA:  dropping %f percent explained variance'%(compsum[scale-1]*100.0/compsum[0]),' by reducing from ',eW.shape[0],' to ', scale,' dimensions')
            plt.bar(scale,0.5*np.max(eW/np.sum(eW)),color='red',width=.1,edgecolor=None)
            plt.annotate('- %f percent'%(compsum[scale-1]*100.0/compsum[0]),xy=(scale,.5*np.max(eW/np.sum(eW))),xytext=(scale-1,.6*np.max(eW/np.sum(eW))))
            plt.title('scree plot')
            plt.xlabel('components')
            plt.ylabel('contribution')
            plt.xlim(0,eW.shape[0]-1)
            plt.ylim(0,np.max(eW/np.sum(eW)))
            ax.show()
            TRAINDAT = np.dot(TRAINDAT,eV[:,:scale])# + P.mean(0)[:scale]
            TESTDAT = np.dot(TESTDAT,eV[:,:scale])# + D.mean(0)[:scale]
            
            '''
            P = P - P.mean(0)
            D = D - D.mean(0)
            TESTDAT = TESTDAT - TESTDAT.mean(0)
            U,L,V = numpy.linalg.svd(P,full_matrices=0, compute_uv=1)
            P = U.dot(np.diag(L)).dot(V[:,:scale])
            #P = recon + P.mean(0)[:scale]            
            U,L,V = numpy.linalg.svd(D,full_matrices=0, compute_uv=1)
            D = U.dot(np.diag(L)).dot(V[:,:scale])
            #D = recon + D.mean(0)[:scale]
            U,L,V = numpy.linalg.svd(TESTDAT,full_matrices=0, compute_uv=1)
            TESTDAT = U.dot(np.diag(L)).dot(V[:,:scale])
            #TESTDAT = recon + TESTDAT.mean(0)[:scale]
            ### watch out-> mean is not added again, therefore blue shade around digits !!!!!! fixed?
            #''' and None
        # FISHER PROJECTION 
        '''
        # compute mean of each class:
        if (app.getCheckBox(" f ")):
            scale = int(app.getEntry("PCA"))
            #print ('Variables for FISHER:  ',np.min(P),P.shape,yx.shape,yx[:10])
            meanVec = np.zeros((outneurons,P.shape[1]))
            Sw = np.zeros((outneurons,P.shape[1],P.shape[1]))
            Sb = np.zeros((outneurons,P.shape[1],P.shape[1]))
            for i in range(outneurons):
                iP = (P[yx==i])
                print ('iP,P shape ',iP.shape, P.shape)
                print ('iP,P  mean shape ',iP.mean(0).shape, P.mean(0).shape)
                meanVec[i] = iP.mean(0) ######## looks good so far .....
                print ('meanVec shape',meanVec.shape)
                Sw[i] =     (np.dot((iP-iP.mean(0)).T,(iP-iP.mean(0))))
                Sb[i] =  (iP.shape[0])  *   (np.dot((iP.mean(0)-P.mean(0)).T,(iP.mean(0)-P.mean(0))))
                print ('Sw,Sb shape ', Sw.shape,Sb.shape)
            #meanC = meanVec.mean(0)  # should be also P.mean(0), shouldn't it?
            print (Sw.shape,Sb.shape)
            Sw = Sw.sum(0) #*  (1./outneurons)
            Sb = Sb.sum(0) #*  (1./outneurons)
            print (Sw.shape,Sb.shape)
            ##get pro (jection)
            pro = np.dot(np.linalg.pinv(Sw),Sb)
            print ('pro shape: ',pro.shape)
            eW,eV = np.linalg.eig(pro)
            print (eV.shape, eW.shape)
            ind = (np.argsort(eW))[::-1]
            eW = eW[ind].real 
            eV = eV[ind].real
            print (eV.shape, eW.shape)
            GGGG = int(app.getEntry("PCA"))
            P = np.dot(P,eV[:,:GGGG])
            D = np.dot(D,eV[:,:GGGG])
            TESTDAT = np.dot(TESTDAT,eV[:,:GGGG])
            #gggg = np.linalg.inv(Sw)  ###### DUUUUDE, WHY NO INVERSE with multi-class pictures?!
            #pro = np.dot(np.linalg.inv(Sw),Sb)
            #pro = pro / (pro ** 2).sum() ** .5
            #P = np.dot(P,pro[:,:36])
            #print ('new P  ', P.shape)
            #pro_ =  (np.linalg.inv((pro.T).dot(Sw).dot(pro))).dot((pro.T).dot(Sb).dot(pro)) 
            #print ('pro_ shape',pro_.shape)
        ''' and None

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  PCA / LDA for dim reduction  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!







        ############################## load weights ####################################################################################################
        if app.getCheckBox("load weights") :
            wx = np.load('w.npy')
            vx = np.load('v.npy')
            if hiddenL >= 2:ux = np.load('u.npy')
            if hiddenL == 3:zx = np.load('z.npy')
        else:
            wx = []
            vx = []
            ux = []
            zx = []


        ############################## draw data randomly ####################################################################################################
        '''
        ### toDO : add some noise to avoid overrepresentation
        if (app.getCheckBox('sample uniformly from data')):            
            Pnew = []
            tnew = []
            for i in range(TRAINDAT.shape[0]):
                j = np.random.randint(0,TRAINDAT.shape[0])
                Pnew += [TRAINDAT[j]]
                tnew += [TRAINLAB[j]]
            TRAINDAT = np.array(Pnew)
            TRAINLAB = np.array(tnew)
        '''
        # CHANGEX
        #print ('outneurons shape before multi output ', outneurons)
        ############################################# multilabel calculations #######################################################################################################
        if(outneurons != 1):
            start_multi = time.perf_counter()              
            ################# 1 Layer #########
            if (hiddenL==1):
                w,v,er,er2,dream = MMLP1(TRAINDAT,TRAINLAB,outneurons,
                                 node1,
                                 iterations,
                                 f_xx,
                                 f2,
                                 erf,
                                 bprg,
                                 beta1,
                                 beta2,
                                 eta,
                                 rglz,regularizer,
                                 TESTDAT,TESTLAB,wx,vx,stoch=sgd,drop1=drout1,drop2=drout2)
                erg = MLP1(TRAINDAT,w,v,f_xx,f2)
                np.save('w.npy',w)
                np.save('v.npy',v)
                print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> weights are saved as: w.npy,v.npy')
                testy = MLP1(TESTDAT,w,v,f_xx,f2)
                yp = np.array([np.argmax(x) for x in testy])
                messi += '\n'+ ('Testset predictions : \n'+str(yp[:]))
            ################# 2 Layer #########    
            if (hiddenL==2):
                w,v,u,er,er2,dream = MMLP2(TRAINDAT,TRAINLAB,outneurons,
                                 node1,
                                 node2,
                                 iterations,
                                 f_xx,
                                 f2,
                                 erf,
                                 bprg,
                                 beta1,
                                 beta2,
                                 eta,
                                 rglz,
                                 regularizer,
                                 TESTDAT,TESTLAB,wx,vx,ux,stoch=sgd,drop1=drout1,drop2=drout2)
                erg = MLP2(TRAINDAT,w,v,u,f_xx,f2)
                np.save('w.npy',w)
                np.save('v.npy',v)
                np.save('u.npy',u)
                print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> weights are saved as: w.npy,v.npy,u.npy')
                testy = MLP2(TESTDAT,w,v,u,f_xx,f2)
                yp = np.array([np.argmax(x) for x in testy])
                messi += '\n'+ ('Testset predictions : \n'+str(yp[:]))
            ################# 3 Layer #########           
            if (hiddenL==3):    
                w,v,u,z,er,er2,dream = MMLP3(TRAINDAT,TRAINLAB,outneurons,
                                 node1,
                                 node2,
                                 node3,
                                 iterations,
                                 f_xx,
                                 f2,
                                 erf,
                                 bprg,
                                 beta1,
                                 beta2,
                                 eta,
                                 rglz,regularizer,
                                 TESTDAT,TESTLAB,wx,vx,ux,zx,stoch=sgd,drop1=drout1,drop2=drout2)
                erg = MLP3(TRAINDAT,w,v,u,z,f_xx,f2)
                np.save('w.npy',w)
                np.save('v.npy',v)
                np.save('u.npy',u)
                np.save('z.npy',z)
                print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> weights are saved as: w.npy,v.npy,u.npy,z.npy')
                testy = MLP3(TESTDAT,w,v,u,z,f_xx,f2)
                yp = np.array([np.argmax(x) for x in testy])
                messi += '\n'+ ('Testset predictions : \n'+str(yp[:]))
            if(app.getEntry("Test label: ") != "") :
                messi += '\n'+ ('Test accuracy: '+str((np.where(yp==TESTLAB_)[0].shape[0] / TESTDAT.shape[0])))
                messi += '\n'
            end_multi = time.perf_counter() 
            time_multi = end_multi-start_multi
            print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> multi output calculations DONE in ',time_multi,' seconds')
            pred = np.array([np.argmax(x) for x in erg])
            messi += '\n'+ ('Training predictions: \n '+str(pred[:35]) + '...')
            rlbl = np.array([np.argmax(x) for x in TRAINLAB])
            messi += '\n'+ ('Trainingset label   : \n '+str(rlbl[:35]) + '...')
            messi += '\n'+ ('Training accuracy: '+str(np.where(pred==rlbl)[0].shape[0] / TRAINDAT.shape[0]))
            messi += '\n'
            messi += '\n'
            ################ save output ######################################################################
            #CHANGEX
            #print ('params before saving ',TRAINLAB.shape[1] , outneurons )
            #'''
            if autoencoder :
                print ('Autoencoder images saved as ae.txt ')
                with open("ae.txt", "w") as files:
                    total = ''
                    for j in range (testy.shape[1]):
                        for i in range(testy.shape[0]):
                            total += str(testy[i,j]) + ","
                        total = total[:-1]
                        total += '\n'
                    total = total[:-1]
                    files.write(total)
                plot_imgs(testy,yp)  
            #'''
            else:
                with open("MMLP_predictions.txt", "w") as files:
                        total = ''
                        for i in range(yp.shape[0]):
                            total += str(yp[i]) + ","
                        total = total[:-1]
                        files.write(total)
                print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> classification label saved as "MMLP_predictions.txt')
            ### re-project
            if (app.getCheckBox("  ")):
                TESTDAT = np.dot(TESTDAT,eV[:,:scale].T)
                TESTDAT += c
        
            plot_er(er,er2,msg)
            if (app.getCheckBox("analytics")):
                plot_imgs(TESTDAT,yp)
        ################ draw MLP ######################################################################
        if (app.getCheckBox("Net")): 
                in_neurons = np.linspace(0,1,TRAINDAT.shape[1])
                in_neurons -= in_neurons.mean()
                h1_neurons = np.linspace(-1,1,node1)
                h2_neurons = np.linspace(-1,1,node2)
                h3_neurons = np.linspace(-1,1,node3)
                out_neurons = np.linspace(0,1,outneurons)
                out_neurons -= out_neurons.mean()
                vv = plt.figure(figsize=(20,10))
                plt.scatter(np.ones(TRAINDAT.shape[1]),in_neurons,label='input: %d'%TRAINDAT.shape[1])
                plt.scatter(np.ones(node1)*2,h1_neurons,label='hidden1: %d'%node1)
                #in - h1
                for i in range(TRAINDAT.shape[1]):
                    plt.plot([.5,1],[in_neurons[i],in_neurons[i]],c='black')
                    for j in range(node1):
                        plt.plot([1,2],[in_neurons[i],h1_neurons[j]], linewidth=.2,c='red')
                if hiddenL == 1:
                    #h1 - out
                    for i in range(node1):
                        for j in range(outneurons):
                            plt.plot([2,5],[h1_neurons[i],out_neurons[j]], linewidth=.2,c='orange')
                if hiddenL == 2:
                    plt.scatter(np.ones(node2)*3,h2_neurons,label='hidden2: %d'%node2)
                    #h1 - h2        
                    for i in range(node1):
                        for j in range(node2):
                            plt.plot([2,3],[h1_neurons[i],h2_neurons[j]], linewidth=.2,c='green')
                    #h2 - out
                    for i in range(node2):
                        for j in range(outneurons):
                            plt.plot([3,5],[h2_neurons[i],out_neurons[j]], linewidth=.2,c='orange')
                if hiddenL == 3:
                    plt.scatter(np.ones(node2)*3,h2_neurons,label='hidden2: %d'%node2)
                    plt.scatter(np.ones(node3)*4,h3_neurons,label='hidden3: %d'%node3)
                    #h1 - h2        
                    for i in range(node1):
                        for j in range(node2):
                            plt.plot([2,3],[h1_neurons[i],h2_neurons[j]], linewidth=.2,c='green')
                    #h2 - h3
                    for i in range(node2):
                        for j in range(node3):
                            plt.plot([3,4],[h2_neurons[i],h3_neurons[j]], linewidth=.2,c='blue')
                    #h3 - out
                    for i in range(node3):
                        for j in range(outneurons):
                            plt.plot([4,5],[h3_neurons[i],out_neurons[j]], linewidth=.2,c='orange')
                plt.scatter(np.ones(outneurons)*5,out_neurons,label='output: %d'%outneurons)
                for j in range(outneurons):
                        plt.plot([5,5.5],[out_neurons[j],out_neurons[j]],c='black')
                plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom='off',      # ticks along the bottom edge are off
                    top='off',         # ticks along the top edge are off
                    labelbottom='off') # labels along the bottom edge are off
                plt.tick_params(
                    axis='y',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    left='off',         # ticks along the bottom edge are off
                    right='off',         # ticks along the top edge are off
                    labelleft='off')    # labels along the left edge are off
                plt.legend()
                vv.show()

        ''' 
        # plot something even for classificatin. somewhat stupid..
        if ((P.shape[1] == 2)and(outneurons != 1)): 
            outneurons = 1
            t = yx
            messi += '\n'+ ('(single output neuron): ')
        '''

        if (outneurons==1):
            if (app.getCheckBox("Dream")):
                video = True
            else:
                video = False
            start_one = time.perf_counter()
            
            ######################################## single output calculations ######################################################################################################################### 
            if hiddenL == 1:
                w,v,er,er2,dream =  MMLP1(TRAINDAT,TRAINLAB,1,node1,iterations,f_xx,f2,erf,bprg,beta1,beta2,eta,rglz,regularizer,TESTDAT,TESTLAB,w=wx,v=vx,rec=video,stoch=sgd,drop1=drout1,drop2=drout2)
                np.save('w.npy',w)
                np.save('v.npy',v)
                print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> weights are saved as: w.npy,v.npy')
                if (app.getCheckBox("analytics")):
                    G_lab = MLP1(G_,w,v,f_xx,f2)                 
                pred  = MLP1(TESTDAT,w,v,f_xx,f2)        
            if hiddenL == 2:
                w,v,u,er,er2,dream = MMLP2(TRAINDAT,TRAINLAB,1,node1,node2,iterations,f_xx,f2,erf,bprg,beta1,beta2,eta,rglz,regularizer,TESTDAT,TESTLAB,w=wx,v=vx,u=ux,rec=video,stoch=sgd,drop1=drout1,drop2=drout2)
                np.save('w.npy',w)
                np.save('v.npy',v)
                np.save('u.npy',u)
                print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> weights are saved as: w.npy,v.npy,u.npy')
                if (app.getCheckBox("analytics")): 
                    G_lab = MLP2(G_,w,v,u,f_xx,f2)           
                pred  = MLP2(TESTDAT,w,v,u,f_xx,f2)  
            if hiddenL == 3:
                w,v,u,z,er,er2,dream = MMLP3(TRAINDAT,TRAINLAB,1,node1,node2,node3,iterations,f_xx,f2,erf,bprg,beta1,beta2,eta,rglz,regularizer,TESTDAT,TESTLAB,w=wx,v=vx,u=ux,z=zx,rec=video,stoch=sgd,drop1=drout1,drop2=drout2)
                np.save('w.npy',w)
                np.save('v.npy',v)
                np.save('u.npy',u)
                np.save('z.npy',z)
                print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> weights are saved as: w.npy,v.npy,u.npy,z.npy')
                if (app.getCheckBox("analytics")):
                    G_lab = MLP3(G_,w,v,u,z,f_xx,f2)     
                pred  = MLP3(TESTDAT,w,v,u,z,f_xx,f2)
            end_one = time.perf_counter()
            time_one = end_one-start_one
            print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> single output calculations DONE in ',time_one,' seconds')
            with open("MMLP_predictions.txt", "w") as files:
                    total = ''
                    for i in range(pred.shape[0]):
                        total += str(pred[i]) + ","
                    total = total[:-1]
                    files.write(total)
            print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> regression label saved as "MMLP_predictions.txt')
            messi += '\n'+ ('Training error:\n'+str((er[-1]))[:6])
            if (er2 != []):
                    messi += '\n'+ ('Test error:\n'+str((er2[-1]))[:6])
            plot_er(er,er2,msg)
            if (app.getCheckBox("analytics")):
                plotting(TRAINDAT,G_,TRAINLAB,G_lab,er)                
            if (app.getCheckBox("Image")):
                if hiddenL == 1:
                    G_lab = MLP1(G_max,w,v,f_xx,f2)
                if hiddenL == 2:
                    G_lab = MLP2(G_max,w,v,u,f_xx,f2)
                if hiddenL == 3:
                    G_lab = MLP3(G_max,w,v,u,z,f_xx,f2)
                fig4 = plt.figure(figsize=(10,10))
                fig4.add_subplot(111)
                plt.imshow(G_lab.reshape(1000,1000),cmap=str(app.getEntry("colormap")))
                plt.axis('off')
                fig4.show()
            
            #showing aproximated imgae  
            if ( (app.getEntry("Test data / Img: ") != "") and (app.getEntry("Training data: ") == "")):       
                fig5 = plt.figure(figsize=(10,5))
                fig5.add_subplot(121)
                plt.title('input')
                plt.imshow(img.reshape(int(np.sqrt(img.shape[0])),int(np.sqrt(img.shape[0]))),cmap=str(app.getEntry("colormap")))
                plt.axis('off')
                fig5.add_subplot(122)
                plt.title('output')
                plt.imshow(pred.reshape(int(np.sqrt(pred.shape[0])),int(np.sqrt(pred.shape[0]))),cmap=str(app.getEntry("colormap")))
                plt.axis('off')
                fig5.show()
       
            ##################################### dreaming  #################################################################################################################################  
            if (app.getCheckBox("Dream")):
                start_dream = time.perf_counter()
                trace = er
                ims = []
                ebs = []
                fig = plt.figure(figsize=(8,9))
                Grr = gridspec.GridSpec(6, 1)
                a = fig.add_subplot(Grr[:5, :])                
                if (app.getCheckBox("Error bar")):
                    b = fig.add_subplot(Grr[5, :])
                    b.set_xlim(0,trace.shape[0]-1)
                    b.set_yticks([np.min(trace),np.max(trace)],minor=False)
                    b.set_xticks([0,trace.shape[0]],minor=True)
                    b.set_axis_bgcolor('black')
                    b.grid(True)
                    b.plot(range(trace.shape[0]),trace,linewidth=.5,color='white',zorder=1,antialiased=True)
                j = 0
                for i in dream:
                    if (app.getCheckBox("analytics")):
                        im = a.scatter(G[:,0],G[:,1],s=16,alpha=.6,c=i.reshape(-1),cmap=str(app.getEntry("colormap")),edgecolor='None',animated=True)\
                        ;a.scatter(TRAINDAT[:,0],TRAINDAT[:,1],s=30,alpha=1,c=TRAINLAB,cmap=str(app.getEntry("colormap")),edgecolor='None')\
                        ;a.set_xlim([-2,2])\
                        ;a.grid(True)\
                        ;a.set_ylim([-2,2])
                    else:
                        im = a.imshow(i, animated=True,cmap=str(app.getEntry("colormap")))
                    ims.append([im])
                    j += 1
                a.axis('off')
                ani = animation.ArtistAnimation(fig, ims, interval=1000./(int(app.getEntry("fps"))), blit=True,
                                                repeat_delay=2000)
                if (app.getCheckBox(" ")):
                    ani.save(app.getEntry("save as"))
                if (app.getCheckBox("Error bar")): 
                    for i in range(trace.shape[0]):
                        eb = b.scatter(range(i),trace[:i],s=5,marker='.',animated=True,color='white',zorder=2)\
                        ;b.set_xlim(0,trace.shape[0]-1)\
                        ;b.set_yticks([np.min(trace),np.max(trace)],minor=False)\
                        ;b.set_xticks([0,trace.shape[0]],minor=True)\
                        ;b.set_axis_bgcolor('black')                        
                        ebs.append([eb])
                    ani2 = animation.ArtistAnimation(fig, ebs, interval=1000./(int(app.getEntry("fps"))), blit=True,repeat_delay=2000)             
                end_dream = time.perf_counter()
                time_dream = end_dream-start_dream
                print('---###---STATUS---',datetime.datetime.now().isoformat(),'---> ','dreaming DONE in ',time_dream,' seconds')
                print('------ MMLP finished after ',(time.perf_counter()-start_import),'sec at --- ',datetime.datetime.now().isoformat(),' ------------------ ')
                plt.show()
            else: print('------ MMLP finished after ',(time.perf_counter()-start_import),'sec at --- ',datetime.datetime.now().isoformat(),' ------------------ ')
        else: print('------ MMLP finished after ',(time.perf_counter()-start_import),'sec at --- ',datetime.datetime.now().isoformat(),' ------------------ ')
        
        if autoencoder:
            app.setMessage("mess",'Autoencoder done!\nData saved as:\n--> ae.txt')
        else:
            app.setMessage("mess",messi)
        #print(messi)
        app.setEntry("seed: ", np.random.randint(1,10000))  # set new seed for next run              
        
    
################################  GUI  ############################################################################################################################################################
hometext = "### MMLP v0.31 ###\n(follow the introductions or just press RUN)\n\
INPUT data has to be:\n\
 - *.txt  comma delimited text file - rows=dimensions(d), cols=samples(n)\n\
 - or IMG (has to be square, will be downscaled to 32x32) \n\
Without input, 3-class data is used, but try binary data by typing 'apple'\n\
in 'Training data'.\n\
- PCA: reduces data to _ dimensions, will be backprojected for plotting \n\
CONFIG\n\
 Error function:\n\
  - qef : quadratic erorr function\n\
  - phl : pseudo huber loss (delta = 1.)\n\
  - bce : binary cross entropy (needs 0/1 sigmoid as activation i.e. f_lgtr)\n\
 Transfer/Activation function:\n\
  - f_tanh : tanh\n\
  - f_atan : atan\n\
  - f_lgtr : logistic transfer function\n\
  - f_sp   : softplus (~not robust, do better)\n\
  - f_relu : rectifier\n\
  - f_bi   : bent identity\n\
  - f_iden : identity function\n\
  - f_bin  : 0/1 binary function\n\
  - f_rint : rounded identity\n\
  - f_stoch: stochastic tranfser function\n\
 load weights: load w.npy,v.npy,u.npy,z.npy as initial weights\n\
 Layer: number of hidden layer, nodes in hidden layer\n\
 Gradient descent:\n\
  - SGD mini-batch: size of mini-batch (0. , 1.) [1= shuffle all, 0= no SGD]\n\
  - Adam   : b1, b2    (try b1: .7-.99  b2: .8-.999)\n\
  - RMSprop: b1\n\
  - iteration: learnign steps (try 10 - 10000)\n\
  - learning rate: (try 0.01,.05,.005)\n\
  - regularization: weight-decay lambda (try 0.001,.0001,.00001)\n\
  - dropout: [input,hlayer] (0:= no dropout)\n\
PLOT:\n\
 - Image: shows 1000x1000 grid from 2D regression / IMG regression\n\
 - Net :  shows MLP layout\n\
 - analytics: shows errorbar, 2D / IMG data over 128x128 grid\n\
 - colormap: colormap for plotting (rainbow,gray,winter,cool,hot,..)\n\
DREAM creates an animation over the iterations \n\
 - fps    : well...fps\n\
 - error bar: adds error bar to dream output" 
app=gui("MMLP", "850x700" )#, "450x700" 
app.setTitle("MMLP v0.31")
app.setGuiPadding(4,4)
app.setIcon("mmlp.ico")
app.setBg("black")
app.setFont(14)
##############################
app.startPanedFrame("p1")

app.addLabelEntry("seed: ",1,1,1)
app.setEntry("seed: ", np.random.randint(1,10000))
#app.setEntryTooltip("seed: ", 'test123, hallo')
app.addLabel("data", "Data", 2, 0,1)
app.addCheckBox('Autoencoder',2,1,1) 
app.addLabelEntry("Training data: ",3,1,1)
app.setEntry("Training data: ", "MNIST_train.txt")
app.addLabelEntry("Training label: ",4,1,1)
app.setEntry("Training label: ", "MNIST_label.txt")
app.addLabelEntry("Test data / Img: ",5,1,1)
app.setEntry("Test data / Img: ", "MNIST_test.txt")
app.addLabelEntry("Test label: ",6,1,1)
app.setEntry("Test label: ", "MNIST_testy.txt")
app.addNamedButton("..","A", cho, 3, 2, 1)     
app.addNamedButton("..","B", cho, 4, 2, 1)  
app.addNamedButton("..","C", cho, 5, 2, 1)
app.addNamedButton("..","D", cho, 6, 2, 1)  

app.addCheckBox("classification",7,1,1)
app.setCheckBox("classification")
app.addLabelEntry("PCA",8,1,1)
app.addCheckBox("  ",8,2,1)
app.setCheckBox("  ")
app.setEntry("PCA", 49)

app.addLabel("config", "Config", 9, 0,1)             
app.addLabelOptionBox("Error function", ["qef", "phl","bce"],9,1,1)
app.addLabelOptionBox("Transfer function", ["f_tanh","f_iden",  "f_atan", "f_relu", "f_lgtr", "f_bi", "f_sp", "f_bin","f_rint","f_stoch"],10,1,1)            
app.setOptionBox("Transfer function",3) 
app.addLabelOptionBox("Activation function", ["f_iden", "f_tanh", "f_atan", "f_relu", "f_lgtr", "f_bi", "f_sp", "f_bin","f_rint","f_stoch"],11,1,1)            
app.setOptionBox("Activation function",4)

app.addCheckBox("load weights",12,1,1)             
app.addLabelOptionBox("Layer", [1,2,3],13,1,1)   
app.setOptionBox("Layer",1)   
app.addLabelEntry("layer 1 nodes",14,1,1)
app.setEntry("layer 1 nodes", 147)
app.addLabelEntry("layer 2 nodes",15,1,1)
app.setEntry("layer 2 nodes", 49)
app.addLabelEntry("layer 3 nodes",16,1,1)
app.setEntry("layer 3 nodes", 8)

#app.addLabel("bprg", "Backpropagation", 17, 1,1)  
app.addLabelEntry("SGD mini-batch", 18, 1,1)
app.setEntry("SGD mini-batch", 0.3)
#app.addCheckBox("_",18,2,1)
app.addLabelOptionBox("Gradient descent ", ["Adam", "RMSprop","normal"],17,1,1)
app.addLabelEntry("b1",19,1,1)
app.setEntry("b1", 0.85)
app.addLabelEntry("b2",20,1,1)
app.setEntry("b2", 0.9)
app.addLabelEntry("iterations",21,1,1)
app.setEntry("iterations", 1500)
app.addLabelEntry("learning rate",22,1,1)
app.setEntry("learning rate", 0.005)
app.addLabelEntry("regularization",23,1,1)
app.setEntry("regularization", 0.001)
app.addLabelOptionBox(" ", ["L2", "L1"],23,0,0)  #,"MX"
app.addLabelEntry("dropout",24,1,1)
app.setEntry("dropout","0, 0")
app.addLabel("plot","Plot",25,0,1)  
app.addCheckBox("Image",25,1,1)
app.addCheckBox("Net",26,1,1)
#app.setCheckBox("Net") 
app.addCheckBox("analytics",27,1,1) 
#app.setCheckBox("analytics")
app.addLabelEntry("colormap",28,1,1)
app.setEntry("colormap", "rainbow") 
   
app.addCheckBox("Dream",28,0,1)     
app.addLabelEntry("fps",30,1,1)
app.setEntry("fps", 10)
app.addCheckBox("Error bar",31,1,1)
app.addCheckBox(" ",32,2,1)
app.addLabelEntry("save as",32,1,1)
app.setEntry("save as", "MMLP_dream.mp4") 
app.addButtons(["Run", "Quit"], press, 33, 1, 1) # Row 3,Column 0,Span 2  
#app.stopPanedFrame()
app.startPanedFrame("p2")
app.setFont(10)
app.addButtons(["start page"], home, 0, 0, 1)
app.addMessage("mess",hometext,1,0,0) 

app.stopPanedFrame() 
app.stopPanedFrame()
#app.registerEvent(mess_up)

app.go()
