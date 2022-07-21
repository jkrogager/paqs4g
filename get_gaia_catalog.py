import datetime
from astroquery.gaia import Gaia

def get_gaia_catalog(catalog_fname):
    QUERY = """
    SELECT source_id as gaia_id, ra, dec, pmra, pmdec, pmra_error, pmdec_error,
        parallax AS plx, parallax/parallax_over_error AS plx_error,
        sqrt(power(pmra/pmra_error, 2) + power(pmdec/pmdec_error, 2)) AS PM_SIG,
        phot_g_mean_mag AS MAG,
        1.086/phot_g_mean_flux_over_error AS MAG_ERR,
        phot_bp_mean_mag AS BPMAG,
        1.086/phot_bp_mean_flux_over_error AS BPMAG_ERR,
        phot_rp_mean_mag AS RPMAG,
        1.086/phot_rp_mean_flux_over_error AS RPMAG_ERR,
        qso.classlabel_dsc_joint AS DSC_CLASS,
        qso.redshift_qsoc AS REDSHIFT_ESTIMATE,
		0.5*(qso.redshift_qsoc_upper - qso.redshift_qsoc_lower) AS REDSHIFT_ERROR
    FROM gaiadr3.gaia_source
    LEFT JOIN gaiadr3.qso_candidates AS qso USING (source_id)
    WHERE b < -62
        AND sqrt(power(pmra/pmra_error, 2) + power(pmdec/pmdec_error, 2)) < 2
        AND parallax_over_error < 3
        AND phot_g_mean_mag < 20.5
    """
    print("Connecting to Gaia server...")
    job = Gaia.launch_job_async(QUERY, dump_to_file=True,
                                output_format='fits', verbose=False,
                                output_file=catalog_fname)
    result = job.get_results()
    print("Done :)")
    return result


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    cat_fname = f'raw_catalog_{now}.fits'
    raw_cat = get_gaia_catalog(cat_fname)
    print("Saved catalog from Gaia database:")
    print(cat_fname)


