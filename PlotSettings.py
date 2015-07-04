#List of implemented fields.
proj_fields = "SurfaceDensity", "SigmaV", "KineticEnergy", "Q", "SFDensity"
slice_fields = "Density", "NumberDensity", "Temperature"

# Fields to actually plot
fields_toplot = {0: ["SigmaV", "KineticEnergy", "Q", "SurfaceDensity", "Temperature", "NumberDensity"],
                 1: ["SurfaceDensity",],
                 2: [],
                 3: [],
                 4: ["SurfaceDensity",]
                 }

# Colormap limits for each field
field_limits = {"SurfaceDensity": [1e-1, 1e7],
                "Temperature": [10, 1e7],
                "SigmaV": [1e-1, 1e4],
                "Density":[1e-30,1e-18],
                "NumberDensity":[1e0, 1e6],
                "KineticEnergy": [1e1, 1e7],
                "Q": [1e-1,1e3],
                "SFDensity" : [1e-8, 10]}

#Plot labels for fields. Can contain TeX syntax
field_labels = {"SurfaceDensity": "$\\Sigma$ $(\mathrm{M_\odot}/\mathrm{pc}^2)$",
                "SigmaV": "$\\sigma_{zz}$ $(\mathrm{km/s})$",
                "KineticEnergy": "$\\frac{\mathrm{d}T}{\mathrm{d}A}$ $(\mathrm{M_\odot} \mathrm{km}^2 {s}^{-2})$",
                "NumberDensity": "$n$ $(\mathrm{cm}^{-3})$",
                "Density": "$\\rho$ $(\mathrm{g} \mathrm{cm}^{-3}$",
                "Temperature": "$T$ $(\mathrm{K})$",
                "Q": "$\\mathcal{Q}$",
                "SFDensity": "$\dot{\Sigma}_\star$ $(\\mathrm{M_\odot} \mathrm{yr}^{-1} \mathrm{pc}^{-2})$"
                                  }
