#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle


from collections import namedtuple


Param = namedtuple('Param', 'R0 DI I0 HospitalisationRate HospiterIters')

def sir(par, distr, flow, alpha, iterations, inf):
    
    r = flow.shape[0]
    n = flow.shape[1]
    N = distr[0].sum() # total population, we assume that N = sum(flow)
    #print(r)
    Svec = distr[0].copy()
    Ivec = np.zeros(n)
    Rvec = np.zeros(n)
    
    if par.I0 is None:
        initial = np.zeros(n)
        # randomly choose inf infections
        for i in range(inf):
            loc = np.random.randint(n)
            if (Svec[loc] > initial[loc]):
                initial[loc] += 1.0
                
    else:
        initial = par.I0
    assert ((Svec < initial).sum() == 0)
 
    Svec = Svec - initial
    Ivec = Ivec + initial

    res = np.zeros((iterations, 4))
    res[0,:] = [Svec.sum(), Ivec.sum(), Rvec.sum(), 0]
    
    newI_ = np.zeros((iterations,n))
    
    realflow = flow.copy() # copy!
    #print(realflow)
    #print("flow = " ,flow)
    # The two lines below normalise the flows and then multiply them by the alpha values. 
    # This is actually the "wrong" the way to do it because alpha will not be a *linear* measure 
    # representing lockdown strength but a *nonlinear* one.
    # The normalisation strategy has been chosen for demonstration purposes of numpy functionality.
    # (Optional) can you rewrite this part so that alpha remains a linear measure of lockdown strength? :)
    
    #print(realflow)
    realflow = realflow / realflow.sum(axis=2)[:,:, np.newaxis] 
    #realflow = alpha * realflow    
    realflow = np.nan_to_num(realflow, copy = True)
    
    #realflow = realflow.astype(int)
   
    history = np.zeros((iterations, 4, n))
    history[0,0,:] = Svec
    history[0,1,:] = Ivec
    history[0,2,:] = Rvec
    
    eachIter = np.zeros(iterations + 1)
    
    #run simulation
    for iter in range(0, iterations - 1):
        realOD = realflow[iter % r]
        d = distr[iter % r] + 1
   
        if ((d>N+1).any()): #assertion!
            print("Miracle, we have a problem!")
            return res, history

        newI = Svec * Ivec / d * par.R0 / par.DI
        newR = Ivec / par.DI
        
        
        Svec = Svec - newI
        
        Ivec = Ivec + newI - newR
        
        Rvec = Rvec + newR
        
        sumSIR = Svec + Ivec + Rvec 
       
        #sumSIR = np.where(sumSIR == 0, 1, sumSIR) 
        
        Svec = (Svec 
               + np.matmul(Svec.reshape(1,n), realOD)/sumSIR
               - Svec * realOD.sum(axis=1)/sumSIR
                )
      
        Ivec = (Ivec 
               + np.matmul(Ivec.reshape(1,n), realOD)/sumSIR
               - Ivec * realOD.sum(axis=1)/sumSIR
                )
      
        Rvec = (Rvec 
               + np.matmul(Rvec.reshape(1,n), realOD)/sumSIR
               - Rvec * realOD.sum(axis=1)/sumSIR
               )
        
        Svec = np.where(Svec < 0, 0, Svec) 
        Ivec = np.where(Ivec < 0, 0, Ivec) 
        Rvec = np.where(Rvec < 0, 0, Rvec) 

        res[iter + 1,:] = [Svec.sum(), Ivec.sum(), Rvec.sum(), 0]

        eachIter[iter + 1] = newI.sum()
        
        res[iter + 1, 3] = eachIter[max(0, iter - par.HospiterIters) : iter].sum() * par.HospitalisationRate
        
        history[iter + 1,0,:] = Svec
        history[iter + 1,1,:] = Ivec
        history[iter + 1,2,:] = Rvec
        newI_[iter+1,:] = newI
    return res, history,newI_
                
    


# In[ ]:




