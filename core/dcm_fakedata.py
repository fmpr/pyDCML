import numpy as np
import pandas as pd

def generate_fake_data_wide(N, T, J, true_alpha, true_beta, true_Omega):
    ###
    #Generate data
    ###

    NT = N * T
    NTJ = NT * J

    L = len(true_alpha) #no. of fixed paramters
    K = len(true_beta) #no. of random parameters

    print("Generating fake data...")
    xFix = np.random.rand(NTJ, L)
    xRnd = np.random.rand(NTJ, K)

    betaInd_tmp = true_beta + \
    (np.linalg.cholesky(true_Omega) @ np.random.randn(K, N)).T
    beta_tmp = np.kron(betaInd_tmp, np.ones((T * J,1)))

    eps = -np.log(-np.log(np.random.rand(NTJ,)))

    vDet = xFix @ true_alpha + np.sum(xRnd * beta_tmp, axis = 1)
    v = vDet + eps

    vDetMax = np.zeros((NT,))
    vMax = np.zeros((NT,))

    chosen = np.zeros((NTJ,), dtype = 'int64')

    for t in np.arange(NT):
        l = t * J; u = (t + 1) * J
        altMaxDet = np.argmax(vDet[l:u])
        altMax = np.argmax(v[l:u])
        vDetMax[t] = altMaxDet
        vMax[t] = altMax
        chosen[l + altMax] = 1

    error = np.sum(vMax == vDetMax) / NT * 100
    print("Error:", error)

    indID = np.repeat(np.arange(N), T * J)
    obsID = np.repeat(np.arange(NT), J)
    altID = np.tile(np.arange(J), NT)  

    num_alternatives = altID.max() + 1
    num_resp = indID.max() + 1

    # convert long format to wide format
    xs = []
    ys = []
    for ind in range(num_resp):
        #print("------------------ individual:", ind)
        ind_ix = np.where(indID == ind)[0]
        #print("ind_ix:", ind_ix)
        ind_xs = []
        ind_ys = []
        for n in np.unique(obsID[ind_ix]):
            #print("--------- observation:", n)
            obs_ix = np.where(obsID == n)[0]
            #print("obs_ix:", obs_ix)

            # get attributes (x)
            x = [[] for i in range(num_alternatives)]
            #print("altID:", altID[obs_ix])
            for alt in range(num_alternatives):
                if alt in altID[obs_ix]:
                    x[alt].append(np.hstack([xFix[obs_ix][alt], xRnd[obs_ix][alt]]))
                else:
                    x[alt].append(np.zeros(L+K))
            x = np.hstack(x)[0]
            #print("x:", x)
            ind_xs.append(x)

            # get choice (y)
            y = np.argmax(chosen[obs_ix])
            #print("y:", y)
            ind_ys.append(y)

        xs.append(np.array(ind_xs))
        ys.append(np.array(ind_ys))

    alt_availability = np.ones((N,T,J))
    alt_attributes = np.array(xs)
    true_choices = np.array(ys)
        
    attr_names = []
    for alt in range(num_alternatives):
        for l in range(L):
            attr_names += ['ALT%d_XF%d' % (alt+1,l+1)]
        for k in range(K):
            attr_names += ['ALT%d_XR%d' % (alt+1,k+1)]

    indID = np.array([n*np.ones(T) for n in range(N)]).astype(int).reshape(N*T,1)
    menuID = np.array([np.arange(T) for n in range(N)]).astype(int).reshape(N*T,1)
    
    df = pd.DataFrame(data=alt_attributes.reshape(N*T,(K+L)*J), columns=attr_names)
    df['choice'] = true_choices.reshape(N*T,1)
    df['indID'] = indID
    df['menuID'] = menuID
    df['obsID'] = np.arange(N*T).astype(int)
    df['ones'] = np.ones(N*T).astype(int)
    
    return df
    