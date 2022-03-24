"""
Plot target overdensities in RA, DEC of input catalog:
    ] python3 CATALOG_FILENAME.fits

The script saves the figure in PDF format

"""

__author__ = 'JK Krogager'

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
import os
import sys

catalog_fname = sys.argv[1]
targetcat = Table.read(catalog_fname)

ra = targetcat['RA']
dec = targetcat['DEC']
ra[ra > 300] -= 360.

# Plot target density:
plt.close('all')
fig = plt.figure(figsize=(10, 7.5))
plt.hexbin(ra, dec,
           C=np.zeros_like(ra),
           gridsize=50,
           mincnt=1,
           reduce_C_function=np.sum,
           cmap=plt.cm.Greys,
           vmin=-1, vmax=10,
           )

hexbin = plt.hexbin(ra, dec, gridsize=500, mincnt=15, reduce_C_function=np.sum)

objects = {
        'Sculptor': [[15.0391667, -33.7088888], [15.04, -32.]],
        'Fornax': [[39.9970833, -34.4491666], [40., -33.]],
        'NGC 55 group': [[3.9470833, -39.2572221], [-5, -41.]],
        'NGC 300 group': [[13.7226917, -37.6842174], [13.8, -40.]],
        'IC 1613 group': [[16.225, +2.133], [16.23, 0.5]],
        }

for name, locators in objects.items():
    pos, text_pos = locators
    plt.annotate(name, pos, text_pos)

plt.xlabel("R.A.  (deg)")
plt.ylabel("Decl.  (deg)")
plt.title("4G-PAQS : Survey Footprint")
plt.tight_layout()

fbase, ext = os.path.splitext(catalog_fname)
fig_fname = fbase+'_overdensity.pdf'
plt.savefig(fig_fname)

