#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import numpy as np

def sir(par, distr_pc, flow_pc, distr_pt, flow_pt, iterations, inf_pc):
    
    r = flow_pc.shape[0]
    n = flow_pc.shape[1]
    
    N_pc = distr_pc[0].sum()  # total population, we assume that N = sum(flow) 
    Svec_pc = distr_pc[0].copy()
    Ivec_pc = np.zeros(n)
    Rvec_pc = np.zeros(n)
    
    N_pt = distr_pt[0].sum()  # total population, we assume that N = sum(flow) 
    Svec_pt = distr_pt[0].copy()
    Ivec_pt = np.zeros(n)
    Rvec_pt = np.zeros(n)
    
    if par.I0 is None:
        initial = np.zeros(n)
        # randomly choose inf infections
        for i in range(inf_pc):
            loc = np.random.randint(n)
            if (Svec_pc[loc] > initial[loc]):
                initial[loc] += 1.0
                
    else:
        initial = par.I0
    assert ((Svec_pc < initial).sum() == 0)
    
    Svec_pc = Svec_pc - initial
    Ivec_pc = Ivec_pc + initial
    
    res_pc = np.zeros((iterations, 4))
    res_pc[0,:] = [Svec_pc.sum(), Ivec_pc.sum(), Rvec_pc.sum(), 0]
    realflow_pc = flow_pc.copy() # copy!
    
    res_pt = np.zeros((iterations, 4))
    res_pt[0,:] = [Svec_pt.sum(), Ivec_pt.sum(), Rvec_pt.sum(), 0]
    realflow_pt = flow_pt.copy() # copy!

    
    newI_pc_rp = np.zeros((iterations,n))
    newI_pt_rp = np.zeros((iterations,n))
    
    # The two lines below normalise the flows and then multiply them by the alpha values. 
    # This is actually the "wrong" the way to do it because alpha will not be a *linear* measure 
    # representing lockdown strength but a *nonlinear* one.
    # The normalisation strategy has been chosen for demonstration purposes of numpy functionality.
    # (Optional) can you rewrite this part so that alpha remains a linear measure of lockdown strength? :)
    realflow_pc = realflow_pc / realflow_pc.sum(axis=2)[:,:, np.newaxis]    
    realflow_pc = np.nan_to_num(realflow_pc, copy = True)
    
    realflow_pt = realflow_pt / realflow_pt.sum(axis=2)[:,:, np.newaxis]    
    realflow_pt = np.nan_to_num(realflow_pt, copy = True)
    
    
    history_pc = np.zeros((iterations, 4, n))
    history_pc[0,0,:] = Svec_pc
    history_pc[0,2,:] = Ivec_pc
    history_pc[0,3,:] = Rvec_pc
    
    eachIter_pc = np.zeros(iterations + 1)
    
    history_pt = np.zeros((iterations, 4, n))
    history_pt[0,0,:] = Svec_pt
    history_pt[0,2,:] = Ivec_pt
    history_pt[0,3,:] = Rvec_pt
    
    eachIter_pt = np.zeros(iterations + 1)
    
    # run simulation
    for iter in range(0, iterations - 1):
        realOD_pc = realflow_pc[iter % r]
        
        realOD_pt = realflow_pt[iter % r]
        
        d_pc = distr_pc[iter % r] + 1
        
        d_pt = distr_pt[iter % r] + 1
        if ((d_pc>N_pc+1).any()): #assertion!
            print("Miracle, we have a problem!")
            return res_pc, history_pc,res_pt, history_pt
        # N =  S + I + R
        
        #print(min(Svec_pt))
        p_pc = (Ivec_pc + Ivec_pt) / (d_pc + d_pt) * par.R0 / par.DI
        p_pt =  Ivec_pt/d_pt * par.R0 / par.DI
        
        newI_pc = Svec_pc * p_pc
        newR_pc = Ivec_pc / par.DI
        
        newI_pt = Svec_pt * (p_pc+p_pt)
        newR_pt = Ivec_pt / par.DI
        
        Svec_pc = Svec_pc - newI_pc
        Ivec_pc = Ivec_pc + newI_pc - newR_pc      
        Rvec_pc = Rvec_pc + newR_pc
        
        
        sumSIR_pc = Svec_pc + Ivec_pc + Rvec_pc
        sumSIR_pt = Svec_pt + Ivec_pt + Rvec_pt
        
        Svec_pc = (Svec_pc 
               + np.matmul(Svec_pc.reshape(1,n), realOD_pc)/sumSIR_pc
               - Svec_pc * realOD_pc.sum(axis=1)/sumSIR_pc
                )
        Ivec_pc = (Ivec_pc 
               + np.matmul(Ivec_pc.reshape(1,n), realOD_pc)/sumSIR_pc
               - Ivec_pc * realOD_pc.sum(axis=1)/sumSIR_pc
                )
        Rvec_pc = (Rvec_pc 
               + np.matmul(Rvec_pc.reshape(1,n), realOD_pc)/sumSIR_pc
               - Rvec_pc * realOD_pc.sum(axis=1)/sumSIR_pc
                )
        
        Svec_pt = Svec_pt.astype(float)
        
        Svec_pt = Svec_pt - newI_pt
        Ivec_pt = Ivec_pt + newI_pt - newR_pt
        Rvec_pt = Rvec_pt + newR_pt
        
        
        Svec_pt = (Svec_pt 
               + np.matmul(Svec_pt.reshape(1,n), realOD_pt)/sumSIR_pt
               - Svec_pt * realOD_pt.sum(axis=1)/sumSIR_pt
                )
        Ivec_pt = (Ivec_pt 
               + np.matmul(Ivec_pt.reshape(1,n), realOD_pt)/sumSIR_pt
               - Ivec_pt * realOD_pt.sum(axis=1)/sumSIR_pt
                )
        Rvec_pt = (Rvec_pt 
               + np.matmul(Rvec_pt.reshape(1,n), realOD_pt)/sumSIR_pt
               - Rvec_pt * realOD_pt.sum(axis=1)/sumSIR_pt
                )
        
        Svec_pc = np.where(Svec_pc < 0, 0, Svec_pc) 
        Ivec_pc = np.where(Ivec_pc < 0, 0, Ivec_pc) 
        Rvec_pc = np.where(Rvec_pc < 0, 0, Rvec_pc) 
        
        Svec_pt = np.where(Svec_pt < 0, 0, Svec_pt) 
        Ivec_pt = np.where(Ivec_pt < 0, 0, Ivec_pt) 
        Rvec_pt = np.where(Rvec_pt < 0, 0, Rvec_pt) 
            
           
        res_pc[iter + 1,:] = [Svec_pc.sum(), Ivec_pc.sum(), Rvec_pc.sum(), 0]
        eachIter_pc[iter + 1] = newI_pc.sum()
        res_pc[iter + 1, 3] = eachIter_pc[max(0, iter - par.HospiterIters) : iter].sum() * par.HospitalisationRate
        
        history_pc[iter + 1,0,:] = Svec_pc
        history_pc[iter + 1,1,:] = Ivec_pc
        history_pc[iter + 1,2,:] = Rvec_pc
        
        res_pt[iter + 1,:] = [Svec_pt.sum(), Ivec_pt.sum(), Rvec_pt.sum(), 0]
        eachIter_pt[iter + 1] = newI_pt.sum()
        res_pt[iter + 1, 3] = eachIter_pt[max(0, iter - par.HospiterIters) : iter].sum() * par.HospitalisationRate
         
        history_pt[iter + 1,0,:] = Svec_pt
        history_pt[iter + 1,1,:] = Ivec_pt
        history_pt[iter + 1,2,:] = Rvec_pt
        

        newI_pc_rp[iter+1,:] = newI_pc
        newI_pt_rp[iter+1,:] = newI_pt
    return res_pc, history_pc,res_pt, history_pt,newI_pc_rp,newI_pt_rp

