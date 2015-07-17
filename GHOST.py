#!/usr/bin/env python
"""
GHOST: Gadget Hdf5 Output Slice and rayTrace

  |\____
(:o ___/
  |/
  
Usage:
GHOST.py <files> ... [options]
GHOST.py <files> ... --CoreSigma [options]

Options:
    -h --help         Show this screen.
    --rmax=<kpc>      Maximum radius of plot window [default: 1.0]
    --plane=<x,y,z>   Slice/projection plane [default: z]
    --c=<cx,cy,cz>    Coordinates of plot window center [default: 0.0,0.0,0.0]
    --cmap=<name>     Name of colormap to use [default: algae]
    --verbose         Verbose output
    --antialiasing    Using antialiasing when sampling the grid data for the actual plot. Costs some speed.
    --gridres=<N>     Resolution of slice/projection grid [default: 400]
    --neighbors=<N>   Number of neighbors used for smoothing length calculation [default: 32]
    --np=<N>          Number of processors to run on. [default: 1]
    --periodic        Must use for simulations in a periodic box.
    --CoreSigma       Make a plot of central surface density vs. time
    --imshow          Make an image without any axes
"""

import matplotlib as mpl
from PlotSettings import *
mpl.use('Agg')
#mpl.rcParams['font.size']=12
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
#from SnapData import *
import h5py
import numpy as np
from yt.visualization import color_maps
import option_d
from scipy import spatial
from matplotlib.colors import LogNorm
import re
import hope
from docopt import docopt
hope.config.optimize = True

arguments = docopt(__doc__)
filenames = arguments["<files>"]
rmax = float(arguments["--rmax"])
plane = arguments["--plane"]
center = np.array([float(c) for c in re.split(',', arguments["--c"])])
verbose = arguments["--verbose"]
AA = arguments["--antialiasing"]
n_ngb = int(arguments["--neighbors"])
gridres = int(arguments["--gridres"])
nproc = int(arguments["--np"])
periodic = arguments["--periodic"]
CoreSigma = arguments["--CoreSigma"]
colormap = arguments["--cmap"]
imshow = arguments["--imshow"]

font = ImageFont.truetype("LiberationSans-Regular.ttf", gridres/12)

if n_ngb > 1:
    from joblib import Parallel, delayed, cpu_count

G = 4.3e4

nums = np.int_([fn.split('_')[1].split('.')[0] for fn in filenames])
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
    norm =  2*2.5464790894703255 #8/np.pi for 3D
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
        print(name)
        f = h5py.File(name, "r")
        header_toparse = f["Header"].attrs
        box_size = header_toparse["BoxSize"]
        self.time = header_toparse["Time"]
        
        particle_counts = header_toparse["NumPart_ThisFile"]
        
        self.field_data = [{}, {}, {}, {}, {}, {}]
        r = {}
        
        for i, n in enumerate(particle_counts):
            if n==0: continue
            if len(fields_toplot[i]) == 0 : continue
            if i==5: continue

            pname = {0:"Gas", 1:"DM", 2:"Disk", 3:"Bulge", 5:"BH", 4:"Stars"}[i]
            
            ptype = f["PartType%d" % i]
            X = np.array(ptype["Coordinates"]) - center
            if periodic: X = (X + box_size/2)%box_size - box_size/2
            r[i] = np.sqrt(np.sum(X[:,:2]**2, axis=1))
            filter = np.max(np.abs(X), axis=1) <= 1e100
            
            for key in ptype.keys():
                self.field_data[i][key] = np.array(ptype[key])[filter]
            r[i] = r[i][filter]

            self.field_data[i]["Coordinates"] = X[filter]
            if not "SmoothingLength" in ptype.keys():
                if verbose: print("Computing smoothing length for %s..." % pname.lower())
                self.field_data[i]["SmoothingLength"] = np.max(spatial.cKDTree(self.field_data[i]["Coordinates"]).query(self.field_data[i]["Coordinates"], n_ngb)[0], axis = 1)
        f.close()

        if verbose: print("Reticulating splines...")        

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
        vel = self.field_data[ptype]["Velocities"]

        if plane != 'z':
            x, y, z = coords.T
            coords = {"x": np.c_[y,z,x], "y": np.c_[x,z,y]}[plane]
            vx, vy, vz = vel.T
            vel = {"x": np.c_[vy,vz,vx], "y": np.c_[vx,vz,vy]}[plane]

        filter = np.max(np.abs(coords[:,:2]),1) <= 1.1*rmax
        coords, vel = coords[filter], vel[filter]
        hsml = self.field_data[ptype]["SmoothingLength"][filter]
        masses = self.field_data[ptype]["Masses"][filter]

        grid_dx = 2*rmax/(gridres-1)
        #floor smoothing length at the Nyquist wavelength to avoid aliasing
        hsml = np.clip(hsml, grid_dx, 1e100)

        field_data = [masses,]

        data_index = {"SurfaceDensity": 0,}

        i = 1        
        if "SigmaV" in fields_toplot[ptype] or "Q" in fields_toplot[ptype] or "KineticEnergy" in fields_toplot[ptype]:
            vzSqr = vel[:,2]**2
            field_data.append(masses*vzSqr)
            data_index["SigmaV"] = i
            i+=1
        if ptype==0:
            if "Q" in  fields_toplot[ptype]:
                omega = np.abs((vel[:,0] * coords[:,1] - vel[:,1] * coords[:,0])/self.r[ptype][filter]**2)
                field_data.append(masses*omega)
                data_index["Q"] = i
                i+=1
            if "SFDensity" in fields_toplot[ptype]:
                sfr = self.field_data[0]["StarFormationRate"][filter]
                field_data.append(sfr)
                data_index["SFDensity"] = i
                i += 1
            if "MagEnergySurfaceDensity" in fields_toplot[ptype]:
                B = self.field_data[0]["MagneticField"][filter]
                sigmaB = np.sum(B**2/2, axis=1) * 2.938e55
                field_data.append(sigmaB)
                data_index["MagEnergySurfaceDensity"] = i
                i+= 1
                

        if verbose: print("Summing projection kernels for type %d..."% ptype)
        griddata = np.zeros((gridres, gridres, len(field_data)))

        coords2d = coords[:,:2]

        DepositDataToGrid(np.vstack(field_data).T, coords2d, len(coords), hsml, gridres, rmax, griddata)

        outdict = {}

        outdict["SurfaceDensity"] = griddata[:,:,0] * 1e4
        if "SigmaV" in fields_toplot[ptype]:
            outdict["SigmaV"] = np.sqrt(griddata[:,:,data_index["SigmaV"]]/griddata[:,:,0])
        if "KineticEnergy" in fields_toplot[ptype]:
            outdict["KineticEnergy"] = 0.5 * griddata[:,:,data_index["SigmaV"]] * 1.988e47
        if "Q" in fields_toplot[ptype]:
            outdict["Q"] = np.sqrt(griddata[:,:,data_index["SigmaV"]]/griddata[:,:,0]) * (griddata[:,:,data_index["Q"]]/griddata[:,:,0]) / G / np.pi / griddata[:,:,0]
        if "MagEnergySurfaceDensity" in fields_toplot[ptype]:
            outdict["MagEnergySurfaceDensity"] = griddata[:,:,data_index["MagEnergySurfaceDensity"]] / 1e6
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

#        hsml = 2*~hsml
        #floor hsml at the Nyquist wavelength to avoid aliasing
        filter = np.abs(coords[:,2]) < hsml
        coords, masses, hsml = coords[filter], masses[filter], hsml[filter]

        hsml_plane = np.sqrt(hsml**2 - coords[:,2]**2)
        hsml[hsml_plane < grid_dx] = np.sqrt(grid_dx**2 + coords[:,2][hsml_plane < grid_dx]**2)

        field_data = [masses,]
        if "JeansMass" in fields_toplot[ptype] or "Temperature" in fields_toplot[ptype]:
            gamma = 5.0/3.0
            x_H = 0.76
            a_e = self.field_data[ptype]['ElectronAbundance'][filter]
            mu = 4.0 / (3.0 * x_H + 1.0 + 4.0 * x_H * a_e)
            field_data.append(masses*self.field_data[ptype]["InternalEnergy"][filter]*1e10*mu*(gamma-1)*1.211e-8)
        if ptype==0 and "Density" in fields_toplot[ptype] or "NumberDensity" in fields_toplot[ptype] or "JeansMass" in fields_toplot[ptype]:
            field_data.append(masses*self.field_data[ptype]["Density"][filter])

        if verbose: print("Summing slice kernels for type %d..."%ptype)
        griddata = np.zeros((gridres, gridres, len(field_data)))

        DepositDataToGrid3D(np.vstack(field_data).T, coords, len(coords), hsml, gridres, rmax, griddata)

        outdict = {}
        if "Density" in fields_toplot[ptype] or "NumberDensity" in fields_toplot[ptype] or "JeansMass" in fields_toplot[ptype]:
            outdict["Density"] = griddata[:,:,2]/griddata[:,:,0] * 6.768e-22
            outdict["NumberDensity"] = outdict["Density"] * 5.97e23
        if "Temperature" in fields_toplot[ptype]:
            outdict["Temperature"] = griddata[:,:,1]/griddata[:,:,0]
            outdict["Temperature"][griddata[:,:,1]==0] = np.nan
        if "JeansMass" in fields_toplot[ptype]:
            outdict["JeansMass"] = 45*outdict["Temperature"]**1.5 * outdict["NumberDensity"]**(-0.5)

        return outdict

    def CentralSurfaceDensity(self):
        sigma = 0.0
        if verbose: print("Computing central surface density")
        for i in xrange(6):
            if i > 0: break
            #if len(self.field_data[i].keys())==0: continue
            else:
                s0 = DepositDataToPoint(self.field_data[i]["Masses"], self.field_data[i]["Coordinates"], len(self.field_data[i]["Masses"]), self.field_data[i]["SmoothingLength"], np.array([0.0,0.0]))
                sigma += s0
        return sigma

def Make2DPlots(data, plane='z', show_particles=False):
    X, Y = data.X, data.Y
    for type in fields_toplot:
        if sum([f in proj_fields for f in fields_toplot[type]]):
            projdata = data.ProjectionData(type, plane)
        if sum([f in slice_fields for f in fields_toplot[type]]):            
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

            if verbose: print("Saving %s..."%plotname)
    
            fig = plt.figure()
            ax = fig.add_subplot(111, axisbg='black')
            ax.set_aspect('equal')
            if imshow:
                Z[Z==0] = Z[Z>0].min()
                
                mpl.image.imsave(plotname, np.log10(Z), cmap=colormap, vmin=np.log10(field_limits[field][0]), vmax=np.log10(field_limits[field][1]))
                F = Image.open(plotname)
                draw = ImageDraw.Draw(F)
                draw.line(((gridres/16, 7*gridres/8), (gridres*5/16, 7*gridres/8)), fill="#FFFFFF", width=6)
                draw.text((gridres/16, 7*gridres/8 + 5), "%gpc"%(rmax*250), font=font)
                F.save(plotname)
                F.close()
            else:
                plot = ax.pcolormesh(X, Y, Z, norm=LogNorm(field_limits[field][0],field_limits[field][1]), antialiased=AA, cmap=colormap)
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
    if verbose: print("kthxbai")

def MakePlot(f):
    data = SnapData(f)
    Make2DPlots(data, plane)

# Here we actually run the code
if CoreSigma:
    t = []
    sigma = []
    for f in filenames:
        print f
        data = SnapData(f)
        t.append(data.time*979)
        sigma.append(data.CentralSurfaceDensity() * 1e4)
        print t[-1], sigma[-1]
    plt.plot(t, sigma)
    plt.savefig("CoreSigma")
else:
    if nproc > 1 and len(filenames) > 1:
        Parallel(n_jobs=nproc)(delayed(MakePlot)(f) for f in filenames)
    else:
        [MakePlot(f) for f in filenames]
    print("Done!")

