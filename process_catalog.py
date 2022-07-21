"""
4G-PAQS Catalog Preparation

Run this script on the raw Gaia catalog to prepare for 4FS upload:
    ] python3 process_catalog.py CATALOG_FILENAME

Contact: J.-K. Krogager, K. E. Heintz
"""
__author__ = 'JK Krogager'

import astropy
from astropy.table import Table
from astropy import units as u
import astropy.coordinates as coord

import datetime
import healpy as hp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# ** [IMPORTANT!] **
# -- Parameters to update manually
#    Make sure that these match the values in the input catalog

MAG_TYPE = 'VEGA'  # VEGA or AB
SUBSURVEY_NAME = '0605_PAQS'
EPOCH = 2016.0
TARGET_CLASS = 'UNKNOWN'

RA_UNIT = 'deg'
DEC_UNIT = 'deg'
PMRA_UNIT = 'mas/yr'
PMDEC_UNIT = 'mas/yr'
MAG_UNIT = 'mag'
MAG_ERR_UNIT = 'mag'
REDDENING_UNIT = 'mag'

# ---------------------------------- 

def main(fname, dec_cut=10, use_crs=False):
    """
    Process the raw input catalog from `fname`.
    Use a cut in declination: DEC < `dec_cut`
    or use the S8-CRS footprint limit... (not implemented yet).
    """
    cat = Table.read(fname, format='fits')

    # Convert all table column names to upper case:
    for colname in cat.colnames:
        cat.rename_column(colname, colname.upper())

    # -- Cut coordinates:
    coords = coord.SkyCoord(cat['RA'], cat['DEC'])
    long_cut = coords.galactic.b < -62 * u.deg
    dec_criterion = cat['DEC'] < dec_cut
    cat = cat[long_cut]
    cat = cat[dec_criterion]
    N_removed_dec = np.sum(~dec_criterion)
    if N_removed_dec == 1:
        print(f" Removing %i target with declination > {dec_cut} deg" % N_removed_dec)
    else:
        print(f" Removing %i targets with declination > {dec_cut} deg" % N_removed_dec)
    

    # -- Define Output Filenames:
    now = datetime.datetime.now().strftime("%Y%m%dT%H%M%SZ")
    fig_output_fname = f"output/sky_density_{now}.pdf"
    cat_output_fname = f"output/S6_{now}_4GPAQS-target_catalog.fits"
    
    
    # -- Manual Fixes:
    cat['DATE_EARLIEST'] = np.zeros(len(cat)) * u.d
    cat['DATE_LATEST'] = np.zeros(len(cat)) * u.d
    cat['EXTENT_PARAMETER'] = np.zeros(len(cat)) * u.arcsec
    cat['EXTENT_INDEX'] = 0.
    cat['EXTENT_FLAG'] = 0
    cat['CADENCE'] = 0
    cat['RESOLUTION'] = 1
    cat['EPOCH'] = np.ones(len(cat)) * EPOCH * u.yr
    if 'SUBSURVEY' in cat.columns:
        cat.remove_column('SUBSURVEY')
    cat['SUBSURVEY'] = SUBSURVEY_NAME
    cat['CLASSIFICATION'] = TARGET_CLASS
    
    
    # -- Fix Mag Column Names:
    if 'GMAG' in cat.columns:
        cat.rename_column('GMAG', 'MAG')
        cat.rename_column('GMAG_ERR', 'MAG_ERR')

    if 'SOURCE_ID' in cat.colnames:
        cat.rename_column('SOURCE_ID', 'GAIA_ID')
    
    if 'MAG_TYPE' in cat.columns:
        cat.remove_column('MAG_TYPE')

    if MAG_TYPE == 'VEGA':
        magtype = np.zeros(len(cat), dtype='U11')
        magtype[:] = 'Gaia_G_Vega'
        cat['MAG_TYPE'] = magtype
    elif MAG_TYPE == 'AB':
        magtype = np.zeros(len(cat), dtype='U9')
        magtype[:] = 'Gaia_G_AB'
        cat['MAG_TYPE'] = magtype
    else:
        print(f"Unknown {MAG_TYPE=}")

    # Check that all three calibration magnitudes are present:
    try:
        cat.rename_column('RPMAG', 'CAL_MAG_RED')
        cat.rename_column('RPMAG_ERR', 'CAL_MAG_ERR_RED')
        cat['CAL_MAG_ID_RED'] = 'GAIA_GRP_AB_PSF'

        cat.rename_column('BPMAG', 'CAL_MAG_BLUE')
        cat.rename_column('BPMAG_ERR', 'CAL_MAG_ERR_BLUE')
        cat['CAL_MAG_ID_BLUE'] = 'GAIA_GBP_AB_PSF'

        cat['CAL_MAG_GREEN'] = cat['MAG']
        cat['CAL_MAG_ERR_GREEN'] = cat['MAG_ERR']
        cat['CAL_MAG_ID_GREEN'] = 'GAIA_G_AB_PSF'
    except Exception as e:
        print("Couldn't find all calibration magnitudes!")
        print(e)
        print("Moving on...")
    
    
    # -- Verify and/or Apply Units:
    column_names = ['RA', 'DEC', 'PMRA', 'PMDEC', 'REDDENING']
    column_units = [RA_UNIT, DEC_UNIT, PMRA_UNIT, PMDEC_UNIT, REDDENING_UNIT]
    for colname, cunit in zip(column_names, column_units):
        if cat[colname].unit is None:
            cat[colname] = cat[colname] * u.Unit(cunit)

    # Verify or apply magnitude units:
    for colname in cat.colnames:
        if 'MAG' in colname and cat[colname].unit is None:
            cat[colname] = cat[colname] * u.Unit('mag')
    
    
    # -- Define healpix indices and their coordinates
    nside = 32
    npix = 12288
    ipix = np.arange(0, npix)
    ra_hp, dec_hp = hp.pixelfunc.pix2ang(
        nside,
        ipix,
        nest=True,   # VERY IMPORTANT!
        lonlat=True
    )
    area_per_pixel = hp.nside2pixarea(nside)*(180/np.pi)**2
    ra_hp = ra_hp*u.deg
    dec_hp = dec_hp*u.deg
    tab_Healpix_idx = Table([ipix, ra_hp, dec_hp], names=('healpix_id', 'RA', 'DEC'))
    
    ra_rad = coord.Angle(ra_hp).wrap_at(180.0*u.degree).radian
    dec_rad = coord.Angle(dec_hp).radian
    
    ra = cat['RA']
    dec = cat['DEC']
    G_Vega = cat['MAG']
    FAINT = G_Vega >= 20.
    BRIGHT = G_Vega < 20.
    
    # -- Determine pixel indices for each subsurvey
    # IMPORTANT: note the definition for THETA and PHI
    #    These are the exact conventions within 4MOST
    
    # sub1 = ALL TARGETS
    ipix_sub1_all = hp.ang2pix(nside,
                               (np.pi/2. - dec*np.pi/180.),
                               (ra*np.pi/180.),
                               nest=True)
    ipix_sub1, numInPix_sub1 = np.unique(ipix_sub1_all, return_counts=True)
    
    # -- Define the target density for each pixel:
    target_N = np.zeros(npix)
    target_N[ipix_sub1] = numInPix_sub1
    target_N[target_N == 0.0] = np.nan
    
    rho_sub1 = np.zeros(npix)
    rho_sub1[ipix_sub1] = numInPix_sub1 / area_per_pixel
    rho_sub1[rho_sub1 == 0.0] = np.nan
    rho_sub1[rho_sub1 == 0.0] = np.nan
    
    med_N = np.median(numInPix_sub1 / area_per_pixel)
    sig_N = 2.5*np.median(np.abs(numInPix_sub1/area_per_pixel - med_N))
    
    overdense_pixels = (numInPix_sub1/area_per_pixel) > med_N + 4*sig_N
    num_overdense_pixels = np.sum(overdense_pixels)
    overdense_fraction = num_overdense_pixels / len(numInPix_sub1) * 100.
    # Use a slightly modified threshold to discard incomplete tiles:
    # sig_N/2.5*1.5 corresponds to ~1 Gaussian sigma.
    # Reject targets below 2-sigma off the median.
    # Visually this corresponds to the break in the distribution as well.
    incomplete_tiles = rho_sub1 < (med_N - sig_N/2.5*1.5*2)

    # Remove overdense regions from the plotting array:
    rho_sub1[ipix_sub1[overdense_pixels]] = np.nan
    # Remove incomplete tiles from the plotting array:
    rho_sub1[incomplete_tiles] = np.nan
    
    survey_pixels = np.sum(rho_sub1 > 0.)
    total_area = area_per_pixel * survey_pixels
    
    print(" Total survey area:  %.0f deg^2" % total_area)
    print(" Number of overdense piexls:  %i" % num_overdense_pixels)
    print(" Fraction of area in overdensity: %.1f %%" % overdense_fraction)
    
    
    # -- Remove dense regions from the catalog:
    dense_mask = sum(ipix_sub1_all == num for num in ipix_sub1[overdense_pixels])

    # -- Remove incomplete tiles from the catalog:
    dense_mask = sum(ipix_sub1_all == num for num in ipix_sub1[incomplete_tiles])
    
    # -- Remove the brightest targets above G < 15
    #    These are not consistent with the shape of the luminosity function
    toobright = G_Vega < 15.
    
    cat = cat[(dense_mask == 0) & ~toobright]
    print(" Removing overdense areas and incomplete tiles along the edge from the catalog")


    # -- Assign target names:
    names = list()
    for c in coords:
        name = 'PAQS_' + c.to_string('hmsdms', sep='', precision=2).replace(' ', '')
        names.append(name)
    cat.remove_column('NAME')
    names = np.array(names)
    cat['NAME'] = names
    
    
    # -- Provide a plot:
    plt.close('all')
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection="mollweide")
    ax1.title.set_text('Target Density')
    ax1.set_xlabel("RA (deg)")
    ax1.set_ylabel("Dec (deg)")
    ax1.grid(True, alpha=0.5)
    ax1.set_xticklabels(['14h', '16h', '18h', '20h', '22h',
                         '0h', '2h', '4h', '6h', '8h', '10h'])
    plt.scatter(
        ra_rad,
        dec_rad,
        c=rho_sub1,
        s=area_per_pixel,
        vmin=0,
        vmax=med_N+4*sig_N,
    )
    cbar1 = plt.colorbar(orientation="horizontal")
    cbar1.ax.set_xlabel("Target density  (deg$^{-2}$)")
    fig1.tight_layout()
    # -- Save the figure:
    fig1.savefig(fig_output_fname, bbox_inches='tight')
    print(f" Saved footprint figure: {fig_output_fname}")
    
    
    # -- Save the catalog:
    cat.write(cat_output_fname, overwrite=True)
    print(f" Saved new catalog: {cat_output_fname}")

    return cat_output_fname



def verify_catalog(catalog_fname, fmt='catalog_format'):
    """
    Check that all columns in the specified catalog format have been defined
    and that the units adhere to the data format specification.

    Returns `True` if the verification passed.
    """
    cat = Table.read(catalog_fname, format='fits')
    datatype = Table.read(fmt,
                          comment='#',
                          format='ascii.csv',
                          delimiter=';')
    
    errors = 0
    for item in datatype:
        colname = item['column']
        colunit = cat[colname].unit
        if colname not in cat.columns:
            print("ERROR - Missing Column: %s" % item['column'])
            errors += 1
    
        if colunit != item['unit']:
            print("ERROR - Incorrect unit of column: %s" % item['column'])
            print("        Found: %s, expected: %s" % (colunit, item['unit']))
            errors += 1

    if errors == 0:
        print(" Verification passed with no errors!")

    return errors == 0



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='4FS Catalog Tool for 4MOST-Gaia PAQS')
    parser.add_argument("filename", type=str, nargs='?', default='',
                        help="Input Gaia catalog")
    parser.add_argument("--cut", type=float, default=10,
                        help="Declination cut in decimal degrees. Targets above this limit will be removed.")
    parser.add_argument("-V", "--verify", action='store_true',
                        help="Run only the catalog verification.")
    parser.add_argument("--format", type=str, default='catalog_format',
                        help="Filename of Catalog Format Specification. \
                              Should have two columns: 'column' and 'unit' separated by `;`")
    parser.add_argument("--star", type=float, default=0.4,
                        help="Fraction of stellar contamination")
    parser.add_argument("--hibal", type=float, default=0.4,
                        help="Fraction of High BAL quasars")
    parser.add_argument("--lobal", type=float, default=0.02,
                        help="Fraction of Low BAL quasars")
    parser.add_argument("--id", type=str, default='highBAL',
                        help="Catalog ID to insert in catalog filename")
    parser.add_argument("--lib", type=str, default='lib',
                        help="Path to template library. Default=lib/")
    args = parser.parse_args()
    
    if args.verify:
        verify_catalog(args.filename, fmt=args.format)

    else:
        from templates import assign_template

        if not os.path.exists('output'):
            os.mkdir('output')

        catalog_name = main(args.filename, dec_cut=args.cut)
        catalog_name = assign_template(
                catalog_name,
                p_star=args.star,
                p_HiBAL=args.hibal,
                p_LoBAL=args.lobal,
                cat_id=args.id,
                temp_path=args.lib,
        )
        passed = verify_catalog(catalog_name, fmt=args.format)

