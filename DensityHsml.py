import numpy as np
import hope
from scipy import spatial
from time import time
hope.config.optimize=True

def GetHsml(coords, des_ngb=32):
    tree = spatial.cKDTree(coords)
    neighbor_dists, neighbors = tree.query(coords, des_ngb)
    hsml = np.empty(len(coords))
#    t = time()
    GetHsmlWork(neighbors, neighbor_dists, hsml, des_ngb, len(hsml))
    return hsml

@hope.jit
def GetHsmlWork(neighbors, neighbor_dists, hsml, des_ngb, N):
    n_ngb = 0.0
    for i in xrange(N):
        upper = 1e-100
        lower = 1e100
        d=0.0

        upper = neighbor_dists[i,des_ngb-1]*2
        lower = neighbor_dists[i, 1]
        error = 1e100

        while error > 1:
            midpoint = (upper+lower)/2
            
#            n_ngb = NNgb(neighbor_dists[i,:], midpoint, des_ngb)
            n_ngb=0.0
            q = 0.0
            for j in xrange(des_ngb):
                q = neighbor_dists[i, j]/midpoint
                if q <= 0.5:
                    n_ngb += (1 - 6*q**2 + 6*q**3)
                elif q <= 1.0:
                    n_ngb += (2*(1-q)**3)
            n_ngb *= 32./3
            if n_ngb > des_ngb:
                upper = midpoint
            else:
                lower = midpoint
            error = np.fabs(n_ngb-des_ngb)
        hsml[i] = midpoint

#@hope.jit
#def NNgb(dists, i, h, des_ngb):


#    return n_ngb
#    for i in xrange(num_pts):
#        density = 0.0
#        fval[:] = 0.0
#        w = 0.0
#        h = hsml[i]

    
#coords = np.load("/home/mgrudic/glass.npy")
#r = np.sum(coords**2, axis=1)**0.5
#coords = coordsB[r.argsort()][:1000]
#r = r[r.argsort()][:1000]

#r, coords = r/(r.max()), coords/(r.max())
#coords = np.load("coords.npy")
#r = np.sum(coords**2, axis=1)
#hsml = GetHsml(np.ones(1e3), coords)
#np.savetxt("hsml.txt", np.c_[hsml[r.argsort()],])
#print (32.0 / 1000)**(1./3), hsml.mean(), np.sum(hsml**3)/32
#np.savetxt("hsml.txt", np.c_[np.loadtxt("hsml.txt"), hsml])

