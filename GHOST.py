#!/usr/bin/env python
"""
GHOST: Gadget Hdf5 Output Slice and rayTrace

  |\____
(:o ___/
  |/
  
Usage:
    GHOST.py [<files>]... [options]

Options:
    -h --help         Show this screen.
    --rmax=<kpc>      Maximum radius of plot window [default: 1.0]
    --plane=<x,y,z>   Slice/projection plane [default: z]
    --c=<cx,cy,cz>    Coordinates of plot window center [default: 0.0,0.0,0.0]
    --cmap=<name>     Name of colormap to use [default: cubehelix]
    --verbose         Verbose output
    --antialiasing    Using antialiasing when sampling the grid data for the actual plot. Costs some speed.
    --gridres=<N>     Resolution of slice/projection grid [default: 400]
    --neighbors=<N>   Number of neighbors used for smoothing length calculation [default: 32]
    --np=<N>          Number of processors to run on. [default: 1]
    --periodic        Must use for simulations in a periodic box.
    --imshow          Make an image without any axes instead of a matplotlib plot
    --metallicity=<Z> If specified, will print this information on the plot             
    --sfeff=<sf>      If specified, will print this information on the plot             
    --mingastemp=<gt> If specified, will print this information on the plot
    --uv=<Nx>         If specified, will print this information on the plot                          
    --cooling=<N>     If specified, will print this information on the plot
    --time=<T>        If sepcified, with only analyze files within a short bit of this timestep 
    --phase           Do a temperature vs density phase diagram instead of plot
    --stars           Count stars instead of plot
    --accrete         If enabled the stars can accrete and so are black hole particles
"""

import matplotlib as mpl
from PlotSettings import *
mpl.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import h5py
import numpy as np
from numpy import genfromtxt
from yt.visualization import color_maps
from scipy import spatial
from matplotlib.colors import LogNorm
import re
import os
import DensityHsml
import hope
from docopt import docopt
from GridDeposit import *
import glob

arguments = docopt(__doc__)
filenames = arguments["<files>"]
if not filenames:
    filenames=glob.glob('snapshot_*.hdf5')
rmax = float(arguments["--rmax"])
plane = arguments["--plane"]
center = np.array([float(c) for c in re.split(',', arguments["--c"])])
verbose = arguments["--verbose"]
AA = arguments["--antialiasing"]
n_ngb = int(arguments["--neighbors"])
gridres = int(arguments["--gridres"])
nproc = int(arguments["--np"])
periodic = arguments["--periodic"]
colormap = arguments["--cmap"]
imshow = arguments["--imshow"]
metallicity = arguments["--metallicity"]
sfeff = arguments["--sfeff"]
uv = arguments["--uv"]
mingastemp = arguments["--mingastemp"]
cooling = arguments["--cooling"]
target_time = arguments["--time"]
analyze_stars = arguments["--stars"]
analyze_phase = arguments["--phase"]
accrete = arguments["--accrete"]
font = ImageFont.truetype("LiberationSans-Regular.ttf", gridres/12)
G = 4.3e4

class SnapData:
    def __init__(self, name):
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
            if i==5 and not accrete: continue
            pname = {0:"Gas", 1:"DM", 2:"Disk", 3:"Bulge", 5:"BH", 4:"Stars"}[i]
            
            ptype = f["PartType%d" % i]
            X = np.array(ptype["Coordinates"]) - center
            if periodic: X = (X + box_size/2)%box_size - box_size/2
            r[i] = np.sqrt(np.sum(X[:,:2]**2, axis=1))
            myfilter = np.max(np.abs(X), axis=1) <= 1e100
            
            for key in ptype.keys():
                #if key == "Temperature":
                #    print "Warning, this was crashing so I am skipping the Temperature key!"
                #    continue
                self.field_data[i][key] = np.array(ptype[key])[myfilter]

            r[i] = r[i][myfilter]

            self.field_data[i]["Coordinates"] = X[myfilter]
            if not "Masses" in ptype.keys():
                self.field_data[i]["Masses"] = f["Header"].attrs["MassTable"][i] * np.ones_like(self.field_data[i]["Coordinates"][:,0])
            if "Temperature" in fields_toplot[i]:

                """
                Temperature = mean_molecular_weight * (gamma-1) * InternalEnergy / k_Boltzmann  
                #k_Boltzmann is the Boltzmann constant  
                gamma = 5/3 (adiabatic index)  
                mean_molecular_weight = mu*proton_mass (definition)  
                mu = (1 + 4*y_helium) / (1+y_helium+ElectronAbundance)  (ignoring small corrections from the metals)  
                ElectronAbundance defined below (saved in snapshot)  
                y_helium = helium_mass_fraction / (4*(1-helium_mass_fraction))  
                helium_mass_fraction is the mass fraction of helium (metallicity element n=1 described below)
                """
                gamma = 5.0/3.0
                x_H = 0.76
                if 'ElectronAbundance' in self.field_data[i].keys():
                    a_e = self.field_data[i]['ElectronAbundance'][myfilter]
                else:
                    a_e = 0.0
                mu = 4.0 / (3.0 * x_H + 1.0 + 4.0 * x_H * a_e)
                self.field_data[i]['Temperature']=self.field_data[i]["Masses"][myfilter]*self.field_data[i]["InternalEnergy"][myfilter]*1e10*mu*(gamma-1)*1.211e-8
            if not "SmoothingLength" in ptype.keys():
                if "AGS-Softening" in ptype.keys():
                    self.field_data[i]["SmoothingLength"] = np.array(ptype["AGS-Softening"])
                else:
                    if verbose: print("Computing smoothing length for %s..." % pname.lower())
                    self.field_data[i]["SmoothingLength"] = DensityHsml.GetHsml(self.field_data[i]["Coordinates"])
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

        myfilter = 0*np.max(np.abs(coords[:,:2]),1) <= 1.1*rmax
        coords, vel = coords[myfilter], vel[myfilter]
        hsml = self.field_data[ptype]["SmoothingLength"][myfilter]
        masses = self.field_data[ptype]["Masses"][myfilter]

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
                omega = np.abs((vel[:,0] * coords[:,1] - vel[:,1] * coords[:,0])/self.r[ptype][myfilter]**2)
                field_data.append(masses*omega)
                data_index["Q"] = i
                i+=1
            if "SFDensity" in fields_toplot[ptype]:
                sfr = self.field_data[0]["StarFormationRate"][myfilter]
                field_data.append(sfr)
                data_index["SFDensity"] = i
                i += 1
            if "MagEnergySurfaceDensity" in fields_toplot[ptype]:
                B = self.field_data[0]["MagneticField"][myfilter]
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

#        hsml = 2*hsml
        #floor hsml at the Nyquist wavelength to avoid aliasing
        myfilter = np.abs(coords[:,2]) < hsml
        coords, masses, hsml = coords[myfilter], masses[myfilter], hsml[myfilter]

        hsml_plane = np.sqrt(hsml**2 - coords[:,2]**2)
        hsml[hsml_plane < grid_dx] = np.sqrt(grid_dx**2 + coords[:,2][hsml_plane < grid_dx]**2)

        field_data = [masses,]
        data_index = {"Density": 0,}
        i = 1
        if "JeansMass" in fields_toplot[ptype] or "Temperature" in fields_toplot[ptype]:
            gamma = 5.0/3.0
            x_H = 0.76
            if 'ElectronAbundance' in self.field_data[ptype].keys():
                a_e = self.field_data[ptype]['ElectronAbundance'][myfilter]
            else:
                a_e = 0.0
            mu = 4.0 / (3.0 * x_H + 1.0 + 4.0 * x_H * a_e)
            field_data.append(masses*self.field_data[ptype]["InternalEnergy"][myfilter]*1e10*mu*(gamma-1)*1.211e-8)
            data_index["Temperature"] = i
            i+=1
        for B in "B_x","B_y","B_z":
            if B in fields_toplot[ptype]:
                field_data.append(masses * self.field_data[ptype]["MagneticField"][:,{"B_x":0, "B_y": 1, "B_z":2}[B]][myfilter])
                data_index[B] = i
                i += 1
        if "B" in fields_toplot[ptype]:
            field_data.append(masses * np.sum(self.field_data[ptype]["MagneticField"][myfilter]**2, axis=1)**0.5)
            data_index["B"] = i
            i += 1
        if ptype==0 and "Density" in fields_toplot[ptype] or "NumberDensity" in fields_toplot[ptype] or "JeansMass" in fields_toplot[ptype]:
            field_data.append(masses*self.field_data[ptype]["Density"][myfilter])
            
        if verbose: print("Summing slice kernels for type %d..."%ptype)
        griddata = np.zeros((gridres, gridres, len(field_data)))

        DepositDataToGrid3D(np.vstack(field_data).T, coords, len(coords), hsml, gridres, rmax, griddata)

        outdict = {}

        #regorganize this into an iteration over the keys of data_index
        if "Density" in fields_toplot[ptype] or "NumberDensity" in fields_toplot[ptype] or "JeansMass" in fields_toplot[ptype]:
            outdict["Density"] = griddata[:,:,-1]/griddata[:,:,0]# * 6.768e-22
            outdict["NumberDensity"] = outdict["Density"] * 404
        if "Temperature" in fields_toplot[ptype]:
            outdict["Temperature"] = griddata[:,:,1]/griddata[:,:,0]
            outdict["Temperature"][griddata[:,:,1]==0] = np.nan
        if "JeansMass" in fields_toplot[ptype]:
            outdict["JeansMass"] = 45*outdict["Temperature"]**1.5 * outdict["NumberDensity"]**(-0.5)
        for B in "B_x","B_y","B_z", "B":
            if B in fields_toplot[ptype]:
                outdict[B] = griddata[:,:,data_index[B]]/griddata[:,:,0]
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

def GetPlotTitle(t):
    extra_info=''

    if metallicity:
        extra_info += ', $Z_{\odot}=%s$' % metallicity
    if sfeff:
        extra_info += ', $SfEffPerFreeFall=%s$' % sfeff
    if mingastemp:
        extra_info += ', $MinGasTemp=%s$' % mingastemp
    if uv:
        extra_info += ', $UV=%s$' % uv
    if cooling:
        cooling_type = ''
        if cooling == '0':
            cooling_type='Tabular'
        elif cooling == '1':
            cooling_type = 'Atomic'
        elif cooling == '2':
            cooling_type = 'Atomic+H2+H2I+H2II'
        elif cooling == '3':
            cooling_type = 'Atomic+H2+H2I+H2II+DI+DII+HD'
        extra_info += ' %s'%cooling_type
    return "$t = %g\\mathrm{Myr}$%s"%(t*979,extra_info)

def Make2DPlots(data, plane='z', show_particles=False):
    if target_time:
        if not np.isclose(data.time,float(target_time)):
            return
    plot_title=GetPlotTitle(data.time)
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
            zmin, zmax = field_limits[field]
            if imshow:
                if  len(Z)>1:
                    if not np.prod(Z<=0):
                        Z[Z==0] = Z[Z>0].min()
                    
                if zmin > 0:
                    mpl.image.imsave(plotname, np.log10(np.abs(Z)), cmap=colormap, vmin=np.log10(field_limits[field][0]), vmax=np.log10(field_limits[field][1]))
                else:
                    mpl.image.imsave(plotname, Z, cmap="RdBu", vmin=field_limits[field][0], vmax=field_limits[field][1])
                F = Image.open(plotname)
                draw = ImageDraw.Draw(F)
                draw.line(((gridres/16, 7*gridres/8), (gridres*5/16, 7*gridres/8)), fill="#FFFFFF", width=6)
                draw.text((gridres/16, 7*gridres/8 + 5), "%gpc"%(rmax*250), font=font)
                F.save(plotname)
                F.close()
            else:
                if zmin > 0 and np.log10(np.abs(zmax)/np.abs(zmin)) > 2 and zmin != 0 and zmax != 0:
                    print X.shape, Y.shape, Z.shape
                    plot = ax.pcolormesh(X, Y, Z, norm=LogNorm(field_limits[field][0],field_limits[field][1]), antialiased=AA, cmap=colormap)
                else:
                    plot = ax.pcolormesh(X, Y, Z, vmin=zmin, vmax=zmax, antialiased=AA, cmap="RdBu")
                bar = plt.colorbar(plot, pad=0.0)
                bar.set_label(zlabel)
                if show_particles:
                    coords = data.field_data[type]["Coordinates"]
                    ax.scatter(coords[:,0], coords[:,1], s=1)
                ax.set_xlim([-rmax,rmax])
                ax.set_ylim([-rmax,rmax])
                ax.set_xlabel("$x$ $(\mathrm{kpc})$")
                ax.set_ylabel("$y$ $(\mathrm{kpc})$")
                plt.title(plot_title)
                plt.savefig(plotname, bbox_inches='tight')
            plt.close(fig)
            plt.clf()
    if verbose: print("kthxbai")

def MakePlot(f):
    data = SnapData(f)
    # This section I added so that the plots can be annotated with more relevent info
    Make2DPlots(data, plane)

def GetInfoFromDir(as_list=False):
    mydir=os.getcwd()
    if 'full1' in mydir or 'part1' in mydir:
        return mydir
    metallicity=re.findall('Metallicity(-?\d*\.?\d+([eE][-+]?\d+)?)',mydir)
    if metallicity:
        metallicity=metallicity[0]
    cooling=re.findall("Grack_(\d)",mydir)
    resolution=re.findall("Res(\d+)",mydir)
    mingas=re.findall("MinGasTemp(\d+)",mydir)
    sfeff=re.findall("SfEff(\d+\.\d+)",mydir)
    uv=re.findall("UV(\d+)x",mydir)
    ssf=re.findall("SSF(\d+_.*)/",mydir)
    if as_list:
        if not metallicity:
            print "warning setting metallicity to default 0.1"
            metallicity=[0.1]
        if not cooling:
            cooling=["normal"]
        else:
            cooling=['grack='+cooling[0]]
        if not mingas:
            mingas=[10]
        if not sfeff:
            sfeff=[1]
        if resolution[0]=='23':
            resolution=['low']
        elif resolution[0]=='50':
            resolution=['high']
        if not uv:
            uv=['0']
        if not ssf:
            ssf=['not_used']
        return [metallicity[0],cooling[0],resolution[0],mingas[0],sfeff[0],uv[0],ssf[0]]
    else:
        outinfo=[]
        if metallicity:
            outinfo+=['Z%s'%metallicity[0]]
        if cooling:
            outinfo+=['grack%s'%cooling[0]]
        if resolution:
            outinfo+=['res%s'%resolution[0]]        
        if mingas:
            outinfo+=['mingas=%s'%mingas[0]]
        if sfeff:
            outinfo+=['sfeff%s'% sfeff[0]]
        if uv:
            outinfo+=['uv%s'%uv[0]]
        if ssf:
            outinfo+=['ssf%s'%ssf[0]]
        return ', '.join(outinfo)

def convert_mass(m):
    unit_mass_in_solarmass_per_h=1.0e10
    return unit_mass_in_solarmass_per_h*m 


def AnalyzePhase(f,as_list=True):
    data = SnapData(f)
    if data.time==0.0:
        return
    if target_time:
        if not np.isclose(data.time,float(target_time)):
            return
    gas_idx = 0
    if (accrete or GetInfoFromDir(f)[-1]!='not_used') and data.field_data[5]:
        stars_idx = 5
    else:
        stars_idx =4


    plt.clf()
    gamma = 5.0/3.0
    x_H = 0.76
    a_e = 0.0
    mu = 4.0 / (3.0 * x_H + 1.0 + 4.0 * x_H * a_e)
    temp=data.field_data[gas_idx]["InternalEnergy"]*1e10*mu*(gamma-1)*1.211e-8

    n, bins, patches = plt.hist(temp, 50, facecolor='green', alpha=0.75) #normed=1 gives probability
    plt.xlabel("Temperature [Kelvin]")
    plt.ylabel("$N_{gas}$")
    plt.xscale("log")
    plt.xlim([10**2,10**7])
#    plt.yscale("log")
    info=GetInfoFromDir(as_list=True) 
    z=info[0]
    plt.title(GetPlotTitle(data.time)+ ' $Z/Z_\odot=%s$'%z)
    #l = plt.plot(bins, y, 'r--', linewidth=1)
    plt.savefig("%s-%f_gastemphist.png"%(GetInfoFromDir(),data.time),dpi=400)

    info+=[np.average(data.field_data[gas_idx]["Masses"]*temp)]
    print 'MassWeight,',','.join(map(str,info))

    #print "GAS SFE",data.time
    #print "ave",np.average(data.field_data[gas_idx]["StarFormationRate"])
    #print "min",np.min(data.field_data[gas_idx]["StarFormationRate"])
    #print "max",np.max(data.field_data[gas_idx]["StarFormationRate"])


    plt.plot(temp,data.field_data[gas_idx]["Density"],'ro')
    plt.xlabel("InternalEnergy [Kelvin]") # particle internal energy (specific energy per unit mass in code units). units are physical
    plt.ylabel("Density [units ?]")# Density (P['rho']): [N]-element array, code-calculated gas density, at the coordinate position of the particle 

    # plt.yscale('log')
    # plt.xscale('log')
    plt.title(GetPlotTitle(data.time))

    #l = plt.plot(bins, y, 'r--', linewidth=1)
    plt.savefig("%s-%f_temp_dens.png"%(GetInfoFromDir(),data.time),dpi=400)


def AnalyzeStars(f,as_list=True):
    data = SnapData(f)
    try:
	hopf=open(f.replace("snapshot","bound").replace("hdf5","dat"))
	hopf=genfromtxt(hopf)
        try:
            biggest_hop_mass=hopf[0][0]
        except:
            biggest_hop_mass=hopf[0]
    except:
	biggest_hop_mass="unknown"
    if data.time==0.0:
        return
    if target_time:
        if not np.isclose(data.time,float(target_time)):
            return
    if (accrete or GetInfoFromDir(f)[-1]!='not_used') and data.field_data[5]:
        stars_idx = 5
    else:
        stars_idx =4
    if not as_list:
        #len(data.field_data[stars_idx]["ParticleIDs"]),"stars","at t=%.3f"%data.time,"for params",GetInfoFromDir()+","," average mass [msol/h]=",np.average(data.field_data[stars_idx]["Masses"]),"max mass=",np.max(data.field_data[stars_idx]["Masses"])
        print "I found", len(data.field_data[stars_idx]["ParticleIDs"]),"stars","at t=%.3f"%data.time,"for params",GetInfoFromDir()+","," average mass [msol/h]=",np.average(data.field_data[stars_idx]["Masses"]),"max mass=",np.max(data.field_data[stars_idx]["Masses"])
    else:
        print  ','.join(map(str,
                            [len(data.field_data[stars_idx]["ParticleIDs"]),
                            "%.4f"%data.time]+
                            GetInfoFromDir(as_list=True)+
                            [np.average(data.field_data[stars_idx]["Masses"]),
                                np.max(data.field_data[stars_idx]["Masses"]),
                                np.sum(data.field_data[stars_idx]["Masses"]),
                                np.sum(data.field_data[0]["Masses"]),
				biggest_hop_mass]))
        #print convert_mass(np.sum(data.field_data[stars_idx]["Masses"])+ np.sum(data.field_data[0]["Masses"]))

    if accrete:
        # here we are going to histogram

        plt.clf()

        n, bins, patches = plt.hist(convert_mass(data.field_data[stars_idx]["Masses"]), 50, facecolor='green', alpha=0.75) #normed=1 gives probability
        plt.xlabel("Mass [$M_\odot/h$]")
        plt.ylabel("$N_{stars}$")
        plt.yscale("log")
        plt.title(GetPlotTitle(data.time))
        #l = plt.plot(bins, y, 'r--', linewidth=1)
        plt.savefig("%s-%f_starhist.png"%(GetInfoFromDir(),data.time),dpi=400)
    
def main():
    if nproc > 1:
        from joblib import Parallel, delayed, cpu_count

    if analyze_stars:
        function = AnalyzeStars
    elif analyze_phase:
        function = AnalyzePhase
    else:
        function = MakePlot

    if nproc > 1 and len(filenames) > 1:
        Parallel(n_jobs=nproc)(delayed(function)(f) for f in filenames)
    else:
        [function(f) for f in filenames]
    if not analyze_stars:
        print("Done!")


if __name__ == '__main__':
    main()

