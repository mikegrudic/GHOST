import numpy as np
import hope
from scipy import spatial
from time import time
hope.config.optimize=True

def GetHsml(coords, des_ngb=32):
    tree = spatial.cKDTree(coords)
    neighbor_dists, neighbors = tree.query(coords, des_ngb)
    if len(coords) < des_ngb:
        return np.ones(len(coords))*neighbor_dists.max()
    hsml = np.empty(len(coords))
    GetHsmlWork(neighbors, neighbor_dists, hsml, des_ngb, len(hsml))
    return hsml

@hope.jit
def GetHsmlWork(neighbors, neighbor_dists, hsml, des_ngb, N):
    n_ngb = 0.0
    for i in xrange(N):
        upper = neighbor_dists[i,des_ngb-1]/0.6
        lower = neighbor_dists[i,1]
        error = 1e100
        count = 0
        while error > 1:
            h = (upper + lower)/2
            n_ngb=0.0
            dngb=0.0
            q = 0.0
            for j in xrange(des_ngb):
                q = neighbor_dists[i, j]/h
                if q <= 0.5:
                    n_ngb += (1 - 6*q**2 + 6*q**3)
                elif q <= 1.0:
                    n_ngb += 2*(1-q)**3
            n_ngb *= 32./3
            if n_ngb > des_ngb:
                upper = h
            else:
                lower = h
            error = np.fabs(n_ngb-des_ngb)
            
        hsml[i] = h
