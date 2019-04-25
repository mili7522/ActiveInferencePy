import numpy as np
from utils import spm_norm, spm_softmax
from scipy.special import psi

class MDP_class(object):
    pass

np.set_printoptions(precision=4, suppress=True)  # Set print options similar to Matlab

def MDP_prelims(MDP):
    """
    Sort out preliminaries in separate code here to make main AI code less lengthy
    
    Set up and preliminaries - get things out of the MDP structure that the AI
    code will use
    """
    
    V = MDP.V  # allowable policies (T - 1, Np)
    
    # number of transitions, policies and states
    #--------------------------------------------------------------------------
    T = MDP.V.shape[0] + 1          # number of transitions
    Np = MDP.V.shape[1]             # number of allowable policies
    Ns = MDP.B.shape[1]             # number of hidden states
    Nu = MDP.B.shape[0]             # number of hidden controls

    # some small constants to keep the code stable, and stop it trying to take
    # logs of zero.
    
    p0  = np.exp(-8)                      # smallest probability 
    q0  = 1/16                            # smallest probability
    
    
    # parameters of generative model and policies
    #==========================================================================
    
    A  = MDP.A + p0            #this A is the default one assigned to the MDP by the game code's 'deal' - update later.
    No = A.shape[0]            # number of outcomes
    
    A_ENV = MDP.A_ENV                  # this doesn't need p0 added on as it never gets log'd etc. Doing that will allow impossible things to happen...
    A          = spm_norm(A)           # normalise 
    A_ENV      = spm_norm(A_ENV)       # normalise
        
    # parameters (concentration parameters): a and A
    #--------------------------------------------------------------------------
    if hasattr(MDP,'a'):       #overrides the above for A if a is provided  (enables learning on A)
        qA = MDP.a + q0  
        qA = psi(qA) - np.dot(np.ones((No,1)), psi(np.sum(qA, axis = 0, keepdims = True)))
        
        #Make the log probabilities produce normalised probability
        #distros: 
        qA = np.log(spm_softmax(qA))
        A = spm_norm(spm_softmax(qA))
        
    else:
        qA = np.log(spm_norm(A))
     
    # transition probabilities (priors)
    #--------------------------------------------------------------------------
    B = np.empty_like(MDP.B)
    B_ENV = np.empty_like(MDP.B_ENV)
    sB = np.empty_like(MDP.B)
    rB = np.empty_like(MDP.B)
    qB = np.empty_like(MDP.B)
    for i in range(Nu):  #one B matrix for each policy
        
        B[i] = MDP.B[i] + p0  #as above for A, will be overwritten if b is enabled
        B[i] = spm_norm(B[i])
           
        B_ENV[i] = MDP.B_ENV[i]
        B_ENV[i] = spm_norm(B_ENV[i])
            
        # parameters (concentration parameters): b and B
        #----------------------------------------------------------------------
        if hasattr(MDP,'b'):  #learning on b
            b     = MDP.b[i] + q0
            sB[i] = spm_norm(b)
            rB[i] = spm_norm(b.T)
            qB[i] = psi(b) - np.dot(np.ones((Ns,1)), psi(np.sum(b, axis = 0, keepdims = True)))
            qB[i] = np.log(spm_softmax(qB[i]))      
            B[i]  = spm_norm(spm_softmax(qB[i]))
        else:
            b     = MDP.B[i] + p0
            sB[i] = spm_norm(b)
            rB[i] = spm_norm(b.T)
            qB[i] = np.log(b)
    
     
    # priors over initial hidden states - d and D
    #--------------------------------------------------------------------------
    if hasattr(MDP,'d'):
        d  = MDP.d + q0  
        qD = psi(d) - np.dot(np.ones((Ns,1)), psi(np.sum(d, axis = 0, keepdims = True))) 
        qD = np.log(spm_softmax(qD))
    elif hasattr(MDP,'D'):
        d  = MDP.D + q0
        qD = np.log(spm_norm(d))
    else:
        d  = np.ones((Ns,1))
        qD = psi(d) - np.dot(np.ones((Ns,1)), psi(np.sum(d, axis = 0, keepdims = True)))
    
    ## - ---------------------------------------------------------------
    # prior preferences (log probabilities) : C
    #--------------------------------------------------------------------------
    try:
        Vo = MDP.C
    except:
        Vo = np.zeros((No,1))  #set flat if not provided.
        
    
    # assume constant preferences, if only final states are specified
    #--------------------------------------------------------------------------
    if Vo.shape[1] != T:
        Vo = np.dot(Vo[:,-1].reshape((-1,1)), np.ones((1,T)))
    
    Vo    = np.log(spm_softmax(Vo))
    H     = np.sum(spm_softmax(qA) * qA, axis = 0, keepdims = True)  #-H as defined in the paper
                                   # precision defaults
    #--------------------------------------------------------------------------
    try:
        alpha = MDP.alpha
    except:
        alpha = 16
    try:
        beta  = MDP.beta
    except:
        beta  = 1
     
    # initial states and outcomes
    #--------------------------------------------------------------------------
    s  = np.zeros(T, dtype = int)
    o  = np.zeros(T, dtype = int) 
    try:
        if isinstance(MDP.s, int):     # initial state (index)
            s[0] = MDP.s
        else:
            s[0] = MDP.s[0]
    except:
        s[0] = 0
    
    try:
        if isinstance(MDP.o, int):     # initial outcome (index)
            o[0] = MDP.o
        else:
            o[0] = MDP.o[0]
    except:
#        o = find(rand < cumsum(A_ENV(:,s)), 1)
        o[0] = np.flatnonzero(np.random.uniform() < np.cumsum(A_ENV[:,s[0]]))[0]
    
    P  = np.zeros((Nu, T - 1))               # posterior beliefs about control
    x  = np.zeros((Ns, T, Np)) + 1 / Ns         # expectations of hidden states | policy
    X  = np.zeros((Ns, T))                   # expectations of hidden states
    u  = np.zeros((Np, T))               # expectations of policy
    a  = np.zeros((T - 1), dtype = int)               # action (index)
        
    # initialise priors over states
    #--------------------------------------------------------------------------
    for k in range(Np):
        x[:,0,k] = spm_softmax(qD).ravel()

     
    # expected rate parameter
    #--------------------------------------------------------------------------
    qbeta = beta                       # initialise rate parameters
    gu    = np.zeros(T) + 1 / qbeta      # posterior precision (policy)
     

    return V, T, No, Np, Ns, Nu,A , qA, B, qB, rB, sB, d, qD, Vo, H, alpha, beta, s, o, P, x, X, u ,a, qbeta, gu, A_ENV, B_ENV