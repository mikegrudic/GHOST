import hope
import numpy as np
hope.config.optimize = True

@hope.jit
def DepositDataToGrid(data, coords, N, hsml, gridres, rmax, griddata):
    norm = 1.8189136353359467 # 40 / (7 pi) for 2D
    grid_dx = 2*rmax/(gridres-1)
    shift_coords = coords[:] + rmax
    
    gxmin = np.int_((shift_coords[:,0] - hsml[:])/grid_dx + 0.5)
    gxmax = np.int_((shift_coords[:,0] + hsml[:])/grid_dx)
    gymin = np.int_((shift_coords[:,1] - hsml[:])/grid_dx + 0.5)
    gymax = np.int_((shift_coords[:,1] + hsml[:])/grid_dx)
    for i in xrange(N):
        h = hsml[i]
        mh2 = data[i,:]/h**2
    
        if gxmin[i] < 0:
            gxmin[i] = 0
        if gxmax[i] > gridres - 1:
            gxmax[i] = gridres - 1
        if gymin[i] < 0:
            gymin[i] = 0
        if gymax[i] > gridres - 1:
            gymax[i] = gridres - 1

        for gx in xrange(gxmin[i], gxmax[i]+1):
            for gy in xrange(gymin[i], gymax[i]+1):
                q = np.sqrt((shift_coords[i,0] - gx*grid_dx)**2 + (shift_coords[i,1] - gy*grid_dx)**2)/h
                if q <= 0.5:
                    griddata[gy, gx,:] += (1 - 6*q**2 + 6*q**3) * mh2
                elif q <= 1.0:
                    griddata[gy, gx,:] += (2*(1-q)**3) * mh2

    griddata[:] = norm*griddata[:]

@hope.jit
def DepositDataToPoint(data, coords, N, hsml, X):
    norm = 1.8189136353359467 # 40 / (7 pi) for 2D

    Xdata = 0.0
    for i in xrange(N):
        h = hsml[i]
        mh2 = data[i]/h**2
        x = coords[i,0] - X[0]
        y = coords[i,1] - X[1]
        q = np.sqrt(x**2 + y**2)/h
        if q <= 0.5:
            Xdata += (1 - 6*q**2 + 6*q**3) * mh2
        elif q <= 1.0:
            Xdata += (2*(1-q)**3) * mh2                

    return norm*Xdata

@hope.jit
def DepositDataToGrid3D(data, coords, N, hsml, gridres, rmax, griddata):
    norm =  2.5464790894703255 #8/np.pi for 3D
    grid_dx = 2*rmax/(gridres-1)
    zSqr = coords[:,2]*coords[:,2]
    hsml_plane = np.sqrt(hsml[:]*hsml[:] - zSqr)
    shift_coords = coords[:,:2] + rmax
    
    gxmin = np.int_((shift_coords[:,0] - hsml_plane[:])/grid_dx + 0.5)
    gxmax = np.int_((shift_coords[:,0] + hsml_plane[:])/grid_dx)
    gymin = np.int_((shift_coords[:,1] - hsml_plane[:])/grid_dx + 0.5)
    gymax = np.int_((shift_coords[:,1] + hsml_plane[:])/grid_dx)
    
    for i in xrange(N):
        h = hsml[i]
        mh3 = data[i,:]/h**3
        z2 = zSqr[i]
    
        if gxmin[i] < 0:
            gxmin[i] = 0
        if gxmax[i] > gridres - 1:
            gxmax[i] = gridres - 1
        if gymin[i] < 0:
            gymin[i] = 0
        if gymax[i] > gridres - 1:
            gymax[i] = gridres - 1
        for gx in xrange(gxmin[i], gxmax[i]+1):
            for gy in xrange(gymin[i], gymax[i]+1):
                q = np.sqrt((shift_coords[i,0] - gx*grid_dx)**2 + (shift_coords[i,1] - gy*grid_dx)**2 + z2)/h
                if q <= 0.5:
                    griddata[gy, gx,:] += (1 - 6*q**2 + 6*q**3) * mh3
                elif q <= 1.0:
                    griddata[gy, gx,:] += (2*(1-q)**3) * mh3

    griddata[:] = norm*griddata[:]

