def ACTIONS(name = None):
    if 'Taxi' in name:
        return ('\u2193', '\u2191', '\u2192', '\u2190', 'P', 'D')
    elif 'Frozen' in name:
        return ('\u2190', '\u2193', '\u2192', '\u2191')
    else: # GridWorld
        return ('\u2193', '\u2192', '\u2191', '\u2190')
        
import numpy as np

class SuccessorRepresentation():
    """
    Successor Representation Class
    
    Based on:
        "Improving Generalization for Temporal Difference Learning: The Successor Representation", Peter Dayan, 1993, http://www.gatsby.ucl.ac.uk/~dayan/papers/sr93.pdf
        
    Parameters
    ============
        P     : dict
                State-action transition dynamics {state : {action : [(p, state_prime, r, done)]}}.
                
        gamma : float
                Discount rate.
                
        theta : float
                Convergence threshold.
                
        cost  : float
                State transition cost.
    """
    def __init__(self, P, gamma = 0.999, theta = 1e-12, cost = 1):
        self.P     = P
        self.gamma = gamma
        self.theta = theta
        self.cost  = cost
        
        self.A = A = np.zeros((len(P),) * 2)
        As         = set()
        
        for i in P:
            for a, ls in P[i].items():
                As |= {a}
                for (_p, j, _, _) in ls:
                    A[[i,j], [j,i]] = 1
        
        self.nA    = len(As)
                    
        S = np.zeros_like(A)
        D = [[dic[action][0][1] for action in dic] for i, dic in P.items()] # Too crude - need to consider stochastic envs i.e. p != 1
        n = len(S)
        e = np.inf
        while e > theta:
            O = S.copy()
            S = np.eye(n) + gamma * S[:,D].mean(axis = -1)
            e = np.absolute(S - O).max()
                
        self.M = S - np.eye(n)

    def value_iterate(self, r, p = 1, max_iters = -1):

        V = np.zeros(len(self.M))
        R = V.copy() + r - self.cost
        # for k, v in r.items():
        #     R[k] += v
        c = 0
        e = np.inf
        while c != max_iters and e > self.theta:
            c  += 1
            old = V.copy()
            for state, dic in self.P.items():
                if state in r:
                    continue
                nA   = len(dic)
                prob = np.eye(nA) * p + (1 - p) * self.nA
                rV   = []
                for action, ls in dic.items():
                    value = 0
                    for (_p, state_prime, _, t) in ls:
                        term   = state_prime in r
                        temp   = R[state_prime] + self.gamma * V[state_prime] * (1 - term)
                        value += _p * temp
                    rV.append(value)
                product  = prob @ rV
                idx      = product.argmax()
                V[state] = max(product)
            e = sum(np.fabs(old - V))
        return (V, R) if c != max_iters else (None, R)
    
    def compute_policy(self, env, r, p = 1, max_iters = -1, as_string = True):
        V, R    = self.value_iterate(r, p, max_iters)
        actions = ACTIONS(env.spec.id)
        P       = np.empty_like(V, dtype = str if as_string else int)
        if isinstance(V, np.ndarray):
            for i in self.P:
                if i in r:
                    P[i] = 'T' if as_string else -1
                    continue
                best = -np.inf
                for action in self.P[i]:
                    value = 0
                    for (p, j, *args) in self.P[i][action]:
                        value += p * (R[j] + self.gamma * V[j])
                    if value > best:
                        best = value
                        act  = action
                P[i] = actions[act] if as_string else act
        return P
