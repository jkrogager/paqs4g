# -*- coding: utf-8 -*-
"""
Update templates and rules
"""

import numpy as np
from scipy import stats
from glob import glob
import os
import sys

from astropy.io import fits
from astropy.table import Table

REDSHIFT_DIST_FNAME = 'Palanque_Delabrouille_zdist.dat'

def assign_template(gaia_fname, p_star=0.4, p_HiBAL=0.40, p_LoBAL=0.02, cat_id='highBAL', temp_path='lib', verbose=True):
    """
    Assign a random template from the library in `temp_path`, default = ./lib/
    A fraction of `p_star` is assigned a random stellar template according to observed
    distribution of stellar types by Heintz et al. (2020)

    The redshifts are drawn from Palanque-Delabrouille et al. (2016).
    Reddening and quasar spectral shapes are drawn from Krawczyk et al. (2015).

    The `cat_id` will be added to the catalog filename.
    Ex: gaia_fname='catalog.fits' -> output_fname='catalog_highBAL.fits'
    
    Alternative BAL mixture. For optical sample:
    p_HiBAL = 0.27, p_LoBAL = 0.003, cat_id = "lowBAL"

    Returns:
    catalog_output : the new catalog filename
    """

    #  * for radio sample:
    p_HiBAL = 0.40
    p_LoBAL = 0.02
    cat_id = "highBAL"
    # -- Load Gaia Target Catalog
    gaia_fname = sys.argv[1]
    gaia = Table.read(gaia_fname)
    
    # -- Create empty array for template filename
    N_gaia = len(gaia)
    temp_pattern = os.path.join(temp_path, '*.fits')
    all_templates = glob(temp_pattern)
    template_names = [fname.split('/')[-1] for fname in all_templates]
    max_tempname = np.max([len(fname) for fname in template_names])
    temp_format = 'U%i' % max_tempname
    template = np.zeros(N_gaia, dtype=temp_format)
    
    # -- Create empty arrays for REDSHIFT_ESTIMATE and REDSHIFT_ERROR
    redshift = np.zeros(N_gaia)
    redshift_error = np.zeros(N_gaia)
    
    
    
    cat_base, cat_ext = os.path.splitext(gaia_fname)
    catalog_output = "%s_%s.fits" % (cat_base, cat_id)
    
    
    # -- Stellar Contaminant:
    N_star = int(N_gaia * p_star)
    N_qso = N_gaia - N_star
    stars = np.zeros(len(gaia), dtype=bool)
    star_idx = np.random.choice(np.arange(N_gaia), N_star, replace=False)
    stars[star_idx] = True
    quasars = ~stars
    template_qso = template[quasars]
    
    
    # -- Draw random redshift sample from Palanque-Delabrouille et al. (2016):
    PD16 = np.loadtxt(REDSHIFT_DIST_FNAME)
    z_pdf = PD16[:, 1] / np.sum(PD16[:, 1])
    z_samples = np.random.choice(PD16[:, 0], size=5000, p=z_pdf)
    kde = stats.gaussian_kde(z_samples)
    z_dist = kde.resample(N_qso)[0]
    z_dist[z_dist > 5.] = 5.
    z_dist[z_dist < 0.] = 0.
    redshift[quasars] = z_dist
    
    
    # -- Krawczyk et al. (2015):
    # Normal distribution of power-law slopes
    alpha_dist = np.random.normal(0., 0.2, N_qso)
    alpha_dist[alpha_dist < -0.7] = -0.7
    alpha_dist[alpha_dist > 0.7] = 0.7
    
    # Exponential distribution of E(B-V)
    Av_dist = np.random.exponential(0.027, N_qso) * 3.1
    Av_dist[Av_dist > 1.2] = 1.2
    
    # -- Assign BAL features:
    # 0: no BAL;  1: HiBAL;  2: FeLoBAL
    BAL = np.random.choice([0, 1, 2], size=N_qso, p=[1.-p_HiBAL-p_LoBAL, p_HiBAL, p_LoBAL])
    
    
    number_BAL_models = [
            3,              # Number of HighBAL models
            10,             # Number of LowBAL models
            ]
    
    # -- Load Templates:
    #    z_range = np.arange(0., 4.5, 0.5) + 0.5/2
    #    alpha_range = np.array([-0.6, -0.4, -0.2, 0., 0.2, 0.4, 0.6])
    #    Av_range = np.array([0., 0.3, 0.6, 1.0])
    qso_pattern = os.path.join(temp_path, 'PAQS_quasar_*.fits')
    quasar_templates = glob(qso_pattern)
    for temp_fname in quasar_templates:
        temp_name = os.path.basename(temp_fname)
        with fits.open(temp_fname) as hdu:
            hdr = hdu[1].header
        z = hdr['REDSHIFT']
        Av = hdr['AV']
        alpha = hdr['ALPHA']
        if Av < 0.2:
            a_lower = 0.
            a_upper = 0.2
        elif np.abs(Av - 0.3) < 0.1:
            a_lower = 0.2
            a_upper = 0.4
        elif np.abs(Av - 0.6) < 0.1:
            a_lower = 0.4
            a_upper = 0.8
        elif np.abs(Av - 1.0) < 0.1:
            a_lower = 0.8
            a_upper = 1.2
        # Select subset of targets with matching
        subset = (np.abs(z_dist - z) <= 0.25)
        subset &= (np.abs(alpha_dist - alpha) <= 0.1)
        subset &= (Av_dist >= a_lower) & (Av_dist < a_upper)
        template_qso[subset] = temp_name
    
        if z < 0.5:
            continue
        
        # Loop over HiBAL and FeLoBAL:
        for bal_type in [1, 2]:
            BAL_subset = subset & (BAL == bal_type)
            # N_BAL_in_bin = len(template[quasars][subset & BAL])
            N_BAL_in_bin = int(np.sum(BAL_subset))
            if N_BAL_in_bin > 0:
                # Assign random BAL:
                if (Av < 0.7) & (np.abs(alpha) < 0.1):
                    BAL_number = number_BAL_models[bal_type-1]
                else:
                    BAL_number = 1
                random_BAL_models = np.random.randint(0, BAL_number, size=N_BAL_in_bin)
                placeholder = template_qso[BAL_subset]
                for bal_num in np.arange(BAL_number):
                    this_bal = random_BAL_models == bal_num
                    if bal_type == 1:
                        BAL_ID = 'HighBAL%i' % bal_num
                    else:
                        BAL_ID = 'FeLoBAL%i' % bal_num
                    bal_name = temp_name.replace('quasar', BAL_ID)
                    placeholder[this_bal] = bal_name
                # Assign placeholder back to parent array:
                template_qso[BAL_subset] = placeholder
    
    template[quasars] = template_qso
    
    # -- Stellar Templates:
    # star_pattern = os.path.join(temp_path, 'PAQS_star_*.fits')
    stellar_templates = ['PAQS_star_M5V.fits', 'PAQS_star_M0V.fits',
                         'PAQS_star_K5V.fits', 'PAQS_star_K0V.fits',
                         'PAQS_star_G5V.fits', 'PAQS_star_G0V.fits',
                         'PAQS_star_F5V.fits', 'PAQS_star_F0V.fits']
    for tname in stellar_templates:
        if tname not in template_names:
            msg = "Could not find template in library: %s" % tname
            raise FileNotFoundError(msg)

    P_star = np.array([0.5, 0.5, 1/8, 1/8, 1/8, 1/8, 1/4, 1/4])
    P_star = P_star / np.sum(P_star)
    random_stellar_templates = np.random.choice(stellar_templates, size=N_star, p=P_star)
    template[stars] = random_stellar_templates
    
    # -- Add template array to catalog
    gaia['TEMPLATE'] = template
    
    FAINT = gaia['MAG'] > 20.
    BRIGHT = gaia['MAG'] <= 20.
    ruleset = np.zeros(len(gaia), dtype='U14')
    ruleset[FAINT] = 'PAQS_LR_FAINT'
    ruleset[BRIGHT] = 'PAQS_LR_BRIGHT'
    gaia['RULESET'] = ruleset
    
    gaia['REDSHIFT_ESTIMATE'] = redshift
    gaia['REDSHIFT_ERROR'] = redshift_error
    
    gaia.write(catalog_output, overwrite=True)
    
    if verbose:
        print(f" Saved updated catalog: {catalog_output}")
    return catalog_output

