#!/usr/bin/env python
from sys import argv, stdout
import os
from PlotSettings import *
from glob import glob
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
import numpy as np
import color_maps
from scipy import spatial
from joblib import Parallel, delayed, cpu_count
from matplotlib.colors import LogNorm
import data_field_defs
import hope
hope.config.optimize = True

G = 4.3e4

filenames = glob(argv[1])
rmax = float(argv[2])


if len(argv) > 3:
    plane = argv[3]
else:
    plane = 'z'
if len(argv) > 4:
    type_toplot = int(float(argv[4])+0.5)
else:
    type_toplot = 0

nums = np.int64([fn.split('_')[1].split('.')[0] for fn in filenames])
filenames = np.array(filenames)[np.argsort(nums)]

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

class SnapData:
    def __init__(self, name):
        f = h5py.File(name, "r")
        header_toparse = f["Header"].attrs
        self.time = header_toparse["Time"]
        
        particle_counts = header_toparse["NumPart_ThisFile"]
        
        self.field_data = [{}, {}, {}, {}, {}]
        r = {}
        
        for i, n in enumerate(particle_counts):
            if n==0: continue
#            if not i in fields_toplot.keys(): continue
            if i==5: continue

            pname = {0:"Gas", 1:"DM", 2:"Disk", 3:"Bulge", 5:"BH", 4:"Stars"}[i]
            
            ptype = f["PartType%d" % i]
            X = np.array(ptype["Coordinates"])
            r[i] = np.sqrt(np.sum(X[:,:2]**2, axis=1))
            filter = r[i] <= 1.01*np.sqrt(2)*rmax #only need to look at particles that are in the window
            
            for key in ptype.keys():
                self.field_data[i][key] = np.array(ptype[key])[filter]
            r[i] = r[i][filter]
            if not "SmoothingLength" in ptype.keys():
                if verbose: print "Computing smoothing length for %s..." % pname.lower()
                self.field_data[i]["SmoothingLength"] = np.max(spatial.cKDTree(self.field_data[i]["Coordinates"]).query(self.field_data[i]["Coordinates"], n_ngb)[0], axis = 1)
        f.close()

        if verbose: print "Reticulating splines..."        

        X, Y = np.linspace(-rmax,rmax,gridres), np.linspace(-rmax,rmax,gridres)
        grid_dx = 2*rmax/(gridres-1)
        X, Y = X-grid_dx/2, Y-grid_dx/2
        self.X, self.Y = np.meshgrid(X,Y)

        self.r = r
        self.num = name.split("_")[1].split('.')[0]
        
    def ProjectionData(self, ptype=0, plane='z'):
        if len(self.field_data[ptype].keys())==0:
            return None
        coords = self.field_data[ptype]["Coordinates"]
        masses = self.field_data[ptype]["Masses"]
        hsml = self.field_data[ptype]["SmoothingLength"]
        vel = self.field_data[ptype]["Velocities"]

        grid_dx = 2*rmax/(gridres-1)
        #floor smoothing length at the Nyquist wavelength to avoid aliasing
        hsml = np.clip(hsml, grid_dx, 1e100)

        if plane != 'z':
            x, y, z = coords.T
            coords = {"x": np.c_[y,z,x], "y": np.c_[x,z,y]}[plane]
            vx, vy, vz = vel.T
            vel = {"x": np.c_[vy,vz,vx], "y": np.c_[vx,vz,vy]}[plane]

        field_data = [masses,]

        data_index = {"SurfaceDensity": 0,}

        i = 1        
        if "SigmaV" in fields_toplot[ptype] or "Q" in fields_toplot[ptype]:
            vzSqr = vel[:,2]**2
            field_data.append(masses*vzSqr)
            data_index["SigmaV"] = i
            i+=1
        if ptype==0:
            if "Q" in  fields_toplot[ptype]:
                omega = np.abs((vel[:,0] * coords[:,1] - vel[:,1] * coords[:,0])/self.r[ptype]**2)
                field_data.append(masses*omega)
                data_index["Q"] = i
                i+=1
            if "SFDensity" in fields_toplot[ptype]:
                sfr = self.field_data[0]["StarFormationRate"]
                field_data.append(sfr)
                data_index["SFDensity"] = i
                i += 1

        if verbose: print "Summing projection kernels..."
        griddata = np.zeros((gridres, gridres, len(field_data)))

        coords2d = coords[:,:2]

        DepositDataToGrid(np.vstack(field_data).T, coords2d, len(coords), hsml, gridres, rmax, griddata)

        outdict = {}

        outdict["SurfaceDensity"] = griddata[:,:,0] * 1e4
        if "SigmaV" in fields_toplot[ptype]:
            outdict["SigmaV"] = np.sqrt(griddata[:,:,data_index["SigmaV"]]/griddata[:,:,0])
        if "KineticEnergy" in fields_toplot[ptype]:
            outdict["KineticEnergy"] = 0.5 * 1e4 * griddata[:,:,data_index["SigmaV"]]
        if "Q" in fields_toplot[ptype]:
            outdict["Q"] = np.sqrt(griddata[:,:,1]/griddata[:,:,0]) * (griddata[:,:,data_index["Q"]]/griddata[:,:,0]) / G / np.pi / griddata[:,:,0]
        if "SFDensity" in fields_toplot[ptype]:
            outdict["SFDensity"] = griddata[:,:,data_index["SFDensity"]] / 1e6

        return outdict

    def SliceData(self, ptype=0, plane='z'):
        if len(self.field_data[ptype].keys())==0:
            return None
        coords = self.field_data[ptype]["Coordinates"]
        masses = self.field_data[ptype]["Masses"]
        hsml = self.field_data[ptype]["SmoothingLength"]
        vel = self.field_data[ptype]["Velocities"]

        if plane != 'z':
            x, y, z = coords.T
            coords = {"x": np.c_[y,z,x], "y": np.c_[x,z,y]}[plane]
            vx, vy, vz = vel.T
            vel = {"x": np.c_[vy,vz,vx], "y": np.c_[vx,vz,vy]}[plane]

        grid_dx = 2*rmax/(gridres-1)

        #floor hsml at the Nyquist wavelength to avoid aliasing
        hsml = np.clip(hsml, 2*grid_dx, 1e100)

        filter = np.abs(coords[:,2]) < hsml
        coords, masses, hsml = coords[filter], masses[filter], hsml[filter]

        field_data = [masses,]
        if "Temperature" in fields_toplot[ptype]:
            field_data.append(masses * self.field_data[ptype]["InternalEnergy"][filter] * 19964.9789829/401.27)

        if verbose: print "Summing slice kernels..."
        griddata = np.zeros((gridres, gridres, len(field_data)))

        DepositDataToGrid3D(np.vstack(field_data).T, coords, len(coords), hsml, gridres, rmax, griddata)

        outdict = {}
        outdict["Density"] = griddata[:,:,0] * 6.768e-22
        outdict["NumberDensity"] = outdict["Density"] * 5.97e23
        if "Temperature" in fields_toplot[ptype]:
            outdict["Temperature"] = griddata[:,:,1]/griddata[:,:,0]

        return outdict

def Make2DPlots(data, plane='z', show_particles=False):
    X, Y = data.X, data.Y
    for type in fields_toplot:
        projdata = data.ProjectionData(type, plane)
#    if sum([f in slice_fields for f in fields]):
        slicedata = data.SliceData(type, plane)
        
        for field in fields_toplot[type]:
            plotname = "%s_%s_PartType%s_r%g_%s.png"%(field, data.num, type, rmax, plane)
            if field in proj_fields:
                if projdata==None: continue
                Z = projdata[field]
            else:
                if slicedata==None: continue
                Z = slicedata[field]
            
            zlabel = field_labels[field]

            if verbose: print "Saving %s..."%plotname
    
            fig = plt.figure()
            ax = fig.add_subplot(111, axisbg='black')
            ax.set_aspect('equal')
            plot = ax.pcolormesh(X, Y, Z, norm=LogNorm(field_limits[field][0],field_limits[field][1]), antialiased=AA)
            bar = plt.colorbar(plot, pad=0.0)
            bar.set_label(zlabel)
            if show_particles:
                X = data.field_data[type]["Coordinates"]
                ax.scatter(X[:,0], X[:,1], s=1)
            ax.set_xlim([-rmax,rmax])
            ax.set_ylim([-rmax,rmax])
            ax.set_xlabel("$x$ $(\mathrm{kpc})$")
            ax.set_ylabel("$y$ $(\mathrm{kpc})$")
            plt.title("$t = %g\\mathrm{Myr}$"%(data.time*979))
            plt.savefig(plotname, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
    if verbose: print "kthxbai"    

#Parallel(n_jobs=6)(delayed(MakePlot)(name) for name in filenames)
for f in filenames:
    print f
    Make2DPlots(SnapData(f), plane=plane)

