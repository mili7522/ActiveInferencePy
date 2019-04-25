import numpy as np
from utils import logist, spm_norm, spm_softmax
from MDP_prelims import MDP_prelims
import copy

def spm_MDP_VB_LC(MDP):
    """
    # A simplified version of spm_MDP_VB (see https://www.fil.ion.ucl.ac.uk/spm/)
    # amended to include calculation of state-action prediction error and model
    # decay. Also includes 'environmental' A and B matrices (A_ENV, B_ENV), 
    # which represent the 'real' environment and are used to generate 
    # observations for the agent / work out the agent's actual location in the
    # real world. 
    
    # Original code spm_MDP_VB  Copyright (C) 2005 Wellcome Trust Centre for
    # Neuroimaging Karl Friston
    
    # Amendments:  Anna Sales 2018, University of Bristol.
    
    # Inputs: an vector of structs, with one struct per trial. MDP structs hold
    # all the parameters relevant for the model during one trial, e.g. a,b,d
    # parameters, states, policies etc.
    
    # Returns: the same struct, but with a complete record of all calculations,
    # updates, actions and observations during each trial.
    """ 
    ## if there are multiple trials ensure that parameters are updated
    #--------------------------------------------------------------------------
    
    if isinstance(MDP, list):
        OUT = copy.deepcopy(MDP)
        for i in range(len(MDP)):       
            # update concentration parameters
            #------------------------------------------------------------------
            if i > 1:
                if hasattr(OUT[i - 1],'a'): MDP[i].a = OUT[i - 1].a
                if hasattr(OUT[i - 1],'b'): MDP[i].b = OUT[i - 1].b
                if hasattr(OUT[i - 1],'c'): MDP[i].c = OUT[i - 1].c
                if hasattr(OUT[i - 1],'d'): MDP[i].d = OUT[i - 1].d
                if hasattr(OUT[i - 1],'beta'): MDP[i].beta = OUT[i - 1].beta
                if hasattr(OUT[i - 1],'SAPEall'): MDP[i].SAPEall= OUT[i - 1].SAPEall
            # solve this trial (send this MDP down to the main code below)
            #------------------------------------------------------------------
            OUT[i] = spm_MDP_VB_LC(MDP[i])  
        MDP = OUT 
        return MDP
    
    # get preliminaries needed to start the trial - e.g. initial values of state 
    # and location, current versions of A,B,D based on a,b,d, values of precision. 
    V, T, No, Np, Ns, Nu, A , qA, B, qB, rB, sB, d, qD, Vo, H, alpha, beta, s, o, P, x, X, u ,a, qbeta, gu, A_ENV, B_ENV = MDP_prelims(MDP)
    
    ## solve
    #==========================================================================
    Ni    = MDP.Ni                         # number of VB iterations
    xn    = np.zeros((Ni,Ns,T,T,Np)) + 1/Ns # history of state updates
    un    = np.zeros((Np,T*Ni))             #    policy updates
    p     = range(Np)   # number of allowable policies
    X_t = np.zeros(list(X.shape) + [T])
#    SAPE = np.zeros(T - 1)
    SAPE = []
    
    for t in range(T):    #Do updates, pick action, get observations for every time point in the trial

        if t > 0:
          
            pol_OK = [v == a[t - 1] for v in V[t - 1,:]]  #include allowable policies only
#            [~,p] = find(pol_OK)
            p = np.flatnonzero(pol_OK)
         
        # Get state updates over all times in task (tau) past and future
        # Variational updates (hidden states) under sequential policies
        #======================================================================
        F = np.zeros((Np,T))
        #NB 'G'  the expected free energy of policies over future time points,
        #is denoted by 'Q' in this code.
        
        for k in p:  # State updates for each policy, over each time point. 
            x[:,:,k] = spm_softmax(np.log(x[:,:,k]) / 2)  #reset.
            for i in range(Ni): # Do Ni iterations of the state update equations (and calculate F / components of Q at the same time)
                px = x[:,:,k]  # store state probabilities for each time, for each policy
                for j in range(T): 
                    # current state
                    #----------------------------------------------------------
                    qx = np.log(x[:,j,k]).reshape((-1,1))
                    
                    # transition probabilities 
                    #------------------------------------------------------
                    if k > Np - 1:
                        fB = sH
                        bB = rH
                    else:
                        if j > 0:
                            fB = sB[V[j - 1, k]]
                        if j < T - 1:
                            bB = rB[V[j, k]]
                    
                    # evaluate free energy and gradients (v = dFdx)
                    #----------------------------------------------------------
                    v = qx
                    if j <= t: v = v - qA[o[j],:].reshape((-1,1))
                    if j == 0: v = v - qD
                    if j > 0: v = v - np.log(np.dot(fB, x[:,j - 1,k].reshape(-1,1)))
                    vF = v 
                    if j < T - 1: v = v - np.log(np.dot(bB, x[:,j + 1,k].reshape(-1,1)))
                    
                                
                    # free energy and belief updating
                    #----------------------------------------------------------
                    F[k,j]  = np.dot(-x[:,j,k].reshape((1,-1)), vF)    ## Free energy of policies at each time point (F(pi,tau))
                    px[:,j] = spm_softmax(qx - v/Ni).ravel() ##  update equation for states.
                
                # hidden state updates
                #--------------------------------------------------------------
                x[:,:,k] = px    #probs of states (rows) over each policy (k, sheets), over each time (cols)     
                
                
        ##  Get expected (future) FE over policies (negative path integral of free energy of policies (Q)
        #======================================================================
        Q = np.zeros((Np,T))
    
        for k in p:  #for each policy
            
            for j in range(T):
                qx = np.dot(A, x[:,j,k])
                Q[k,j] = np.dot(qx.T, Vo[:,j] - np.log(qx)) + np.dot(H, x[:,j,k]) #Expected free energy of k-th policy at time t=j   
    
        # Calculate Q, F as sum over time - total free energy in past/future. 
        F = np.sum(F, 1, keepdims = True)
        Q = np.sum(Q, 1, keepdims = True)
       
        ## Get policy probability and precision, pi / beta and gamma
           
        for i in range(Ni):
            
            # policy (u)
            #------------------------------------------------------------------
            #TODO: No i in this for loop
            qu = spm_softmax( np.dot(gu[t], Q[p]) + F[p] )  #pi, probability of each policy
            pu = spm_softmax( np.dot(gu[t], Q[p]) ) # pi_0 
            v = qbeta - beta + np.dot((qu - pu).T, Q[p])   #update equation for beta
            
            # precision (gu) 
         
            qbeta = qbeta - v/2 ## UPDATE = OLD BETA + ERROR
            gu[t] = 1/qbeta


            u[p,t] = qu.ravel()  #store history of values of policy prob. 
      
        
        # Bayesian model averaging of hidden states over policies
        
        for i in range(T):
            X[:,i] = np.dot(np.squeeze(x[:,i,:]), u[:,t].reshape((-1,1))).ravel()
            X_t[:,i,t] = X[:,i]
        
        # Calculate the state-action prediction error as a KL divergence
        # between successive BMA distributions.
        
        if t > 0:
            St_lg_change = np.log(X_t[:,:,t]) - np.log(X_t[:,:,t-1]) 
            SAPE.append(np.sum(X_t[:,:,t] * St_lg_change))

          
        # action selection and observations
    
        if t < T - 1:
            # posterior expectations about (remaining) actions (q)
            if len(p) > 1:
                q = np.unique(V[t,p]) #make sure if only picks allowable actions.
            else:
                q = V[t,p]
            
            v = np.log( np.dot(A, X[:,t + 1].reshape((-1,1))) )
            
            for j in q:
                qo = np.dot(np.dot(A, B[j]), X[:,t].reshape((-1,1)))
                P[j,t] = np.dot((v - np.log(qo)).T, qo) + 16
            
            # action selection     
            P[:,t] = spm_softmax(np.dot(alpha, P[:,t]))
            a[t] = np.argmax(P[:,t], axis = 0)  #deterministic
              
            # Use environment matrices to work out where agent ends up in the
            # real world.
          
#            s[t + 1] = find(rand < cumsum(B_ENV{a(t)}(:,s(t))),1)
            s[t + 1] = np.flatnonzero(np.random.uniform() < np.cumsum(B_ENV[a[t],:,s[t]]))[0]
    
            
            # Use environmental matrices to get an observation from the real
            # world.
    
            o[t + 1] = np.flatnonzero(np.random.uniform() < np.cumsum(A_ENV[:,s[t + 1]]))[0]
      
            # save outcome and state sampled
            #------------------------------------------------------------------
            gu[t + 1]   = gu[t]
            
    
    ## End of trial. Now do updates to concentration parameters.
    
    #Calculate model decay factor (here denoted 'df') based on logistic function using prediction errors
    
    # These are mean values worked out for GNG / EE tasks from 100 trials with
    # df =16.
    if T == 3:   #GNG 
        mean = 1
    elif T == 2:  #EE
        mean = 1.8


    if hasattr(MDP, 'df_set') and (MDP.df_set is not None):  #If we're forcing the agent to use a fixed df.
        df = MDP.df_set 
        df_settings = {'vals': df}    #store the values of df that the MDP used.
    else:   #Or use SAPE to calculate it from a logistic function.
        min_d = 2
        max_d = 32
        grad_d = 8 
        df = logist(np.max(SAPE), grad_d, max_d, min_d, mean)   
        df_settings = {'grad_d': grad_d, 'max_d': max_d, 'min_d': min_d, 'mean': mean}     #store the values of df that the MDP used.
        
    ##
    
    for t in range(T):
       
    # update concentration parameters - use model decay calculated above.
    #----------------------------------------------------------------------
        if hasattr(MDP, 'a'):
            decay = np.zeros(MDP.a.shape)
        
        if hasattr(MDP,'a'):
            i = MDP.a > 0
            tmp = np.zeros((No,1))
            tmp[o[t],0] = 1
            da = np.dot(tmp, X[:,t].reshape((1,-1)))
            dec_weights = np.tile(da[o[t], :], (No,1))
            dec_weights[~i] = 0   #don't change things that are already 0.
            mask = np.ones(MDP.a.shape, dtype = bool)
            mask[o[t], :] = 0  #only want to decay elements in row for observation seen, as per outer product in update equation 
            MDP.a[mask] = MDP.a[mask] - dec_weights[mask] * ( (MDP.a[mask] - 1) / df )  #decay
            MDP.a[i] = MDP.a[i] + da[i]     #increment
        
       
        if hasattr(MDP,'b') and t > 0:
        
            for k in range(Np):
                v           = V[t - 1, k]
                i           = MDP.b[v] > 0
                db          = np.dot(np.dot(u[k, t - 1], x[:,t,k].reshape((-1,1))), x[:,t - 1,k].reshape((1,-1)))
                MDP.b[v][i] = MDP.b[v][i] + db[i] - (MDP.b[v][i] -1)/df         

    
    # initial hidden states:
    #--------------------------------------------------------------------------
    if hasattr(MDP,'d'):
        i        = MDP.d > 0
        MDP.d[i] = MDP.d[i] + X[i.ravel(),0] - (MDP.d[i] - 1)/df    
        
    
    ## assemble results and place in MDP structure
    MDP.P   = P              # probability of action at time 1,...,T - 1
    MDP.Q   = x              # conditional expectations over N hidden states
    MDP.X   = X              # Bayesian model averages
    MDP.X_t = X_t              # BMA at each time point over all times in a trial.
    MDP.R   = u              # conditional expectations over policies
    MDP.o   = o              # outcomes at 1,...,T
    MDP.s   = s              # states at 1,...,T
    MDP.u   = a              # action at 1,...,T 
    MDP.SAPEall = MDP.SAPEall + SAPE
    MDP.w   = gu             # posterior expectations of precision (policy)
    MDP.C   = Vo             # utility
    MDP.A = A                  # this is the A matrix that has been used throughout this MDP (next time it'll be updated from a)
    MDP.A_ENV = A_ENV          # enivornmental A_ENV
    MDP.Ni = Ni               # number of iterations
    MDP.SAPE = SAPE            # state action prediction error based on changes to BMA states
    MDP.df = df               # decay factor used in trial
    MDP.beta = 1/gu[T-1]       # carry forward beta
    MDP.dfsettings = df_settings
    MDP.B = B
    
    
    return MDP