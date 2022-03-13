import numpy as np
from copy import copy, deepcopy


def vi_exp(MDP,tol,maxIter,dis,):

    # Initialize
    Ns = MDP.getNumStates()
    maxIter = int(maxIter)
    V = np.zeros((Ns, 1))

    # print np.size(MDP.P, axis=0)
    Pol = np.zeros((Ns, 1))
    # Do Value Iteration
    for i in range(maxIter):
        V_prev = deepcopy(V)
        for s in range(Ns):
            a = MDP.getActions(s)
            Q = MDP.getReward(s, a) + np.dot(MDP.nextStateProb(s, a), V_prev)
            V[s] = np.max(Q)  # maximum element of each column
            Pol[s] = np.argmax(Q)  # maximum element of each column

        err = np.max(np.absolute(V - V_prev))
        if err < tol:
            print('Error threshold reached at iteration: ', i, err)
            return (V, Pol, err)

    return (V, Pol, err)
