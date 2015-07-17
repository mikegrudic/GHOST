#List of implemented fields.
proj_fields = "SurfaceDensity", "SigmaV", "KineticEnergy", "Q", "SFDensity", "MagEnergySurfaceDensity"
slice_fields = "Density", "NumberDensity", "Temperature", "MagEnergyDensity"

# Fields to actually plot
fields_toplot = {0: ["SurfaceDensity", "Temperature"],#["NumberDensity", "Temperature","SurfaceDensity", "Q", "SFDensity", "KineticEnergy", "SigmaV"],
                 1: [],
                 2: [],
                 3: [],
                 4: [],
                 5: []
                 }

# Colormap limits for each field
field_limits = {"SurfaceDensity": [1e-1, 1e6],
                "Temperature": [10, 1e7],
                "SigmaV": [1e-1, 1e4],
                "Density":[1e-30,1e-18],
                "NumberDensity":[1e0, 1e11],
                "KineticEnergy": [1e40, 1e51],
                "Q": [1e-1,1e3],
                "SFDensity" : [1e-8, 10],
                "MagEnergySurfaceDensity": [1e40, 1e51]}

#Plot labels for fields. Can contain TeX syntax
field_labels = {"SurfaceDensity": "$\\Sigma$ $(\mathrm{M_\odot}/\mathrm{pc}^2)$",
                "SigmaV": "$\\sigma_{zz}$ $(\mathrm{km/s})$",
                "KineticEnergy": "$\\frac{\mathrm{d}T}{\mathrm{d}A}$ $(\mathrm{erg} \, \mathrm{pc}^{-2})$",
                "NumberDensity": "$n$ $(\mathrm{cm}^{-3})$",
                "Density": "$\\rho$ $(\mathrm{g} \mathrm{cm}^{-3}$",
                "Temperature": "$T$ $(\mathrm{K})$",
                "Q": "$\\mathcal{Q}$",
                "SFDensity": "$\dot{\Sigma}_\star$ $(\\mathrm{M_\odot} \mathrm{yr}^{-1} \mathrm{pc}^{-2})$",
                "MagEnergySurfaceDensity": "$\\Sigma_{B}$ $(\mathrm{erg} \,\mathrm{pc}^{-2})$"
                                  }
