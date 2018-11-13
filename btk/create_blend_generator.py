"""The generator creates blend lists according to a given strategy"""
import numpy as np


def get_random_center_shift(Args, number_of_objects, maxshift_stamp_frac=0.1, minshift_stamp_frac=0.0, radial=False):
    """
    Returns a random shift from the center in x and y coordiantes
    - if radial=False, in a square of size maxshift*Args.stamp_size (in arcseconds).
    - if radial=True, in an annulus of radii between minshift_stamp_frac*Args.stamp_size and
    maxshift_stamp_frac*Args.stamp_size (in arcseconds).
    """
    maxshift = Args.stamp_size * maxshift_stamp_frac  # in arcseconds
    minshift = Args.stamp_size * minshift_stamp_frac  # in arcseconds
    if radial:
        r = np.sqrt(np.random.rand(number_of_objects)*(maxshift**2-minshift**2)+minshift**2);
        t = 2.*np.pi*np.random.rand(number_of_objects)
        dx = r * np.cos(t)
        dy = r * np.sin(t)
    else:
        dx = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
        dy = np.random.uniform(-maxshift, maxshift,
                           size=number_of_objects)
    return dx, dy


def random_sample(Args, catalog):
    """Randomly picks entries from input catlog that are brighter than 25.3
    mag in the i band. The centers are randomly distributed within 1/5 of the
    stamp size.
    """
    number_of_objects = np.random.randint(1, Args.max_number)
    q, = np.where(catalog['i_ab'] <= 25.3)
    blend_catalog = catalog[np.random.choice(q, size=number_of_objects)]
    blend_catalog['ra'], blend_catalog['dec'] = 0., 0.
    dx, dy = get_random_center_shift(Args, number_of_objects)
    blend_catalog['ra'] += dx
    blend_catalog['dec'] += dy
    return blend_catalog


def generate(Args, catalog, sampling_function=None):
    """Generates a list of blend catalogs of length Args.batch_size. Each blend
    catlog has overlapping objects between 1 and Args.max_number.
    """
    while True:
        blend_catalogs = []
        for i in range(Args.batch_size):
            if sampling_function:
                blend_catalog = sampling_function(Args, catalog)
            else:
                blend_catalog = random_sample(Args, catalog)
            np.testing.assert_array_less(len(blend_catalog) - 1, Args.max_number,
                                         "Number of objects per blend must be \
                                         less than max_number")
            blend_catalogs.append(blend_catalog)
        yield blend_catalogs
