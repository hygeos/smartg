

class DM_trunc(object):
    """ 
    Delta-M truncation

    Parameters
    ----------
    nb_streams : int
        Number of streams for the truncated phase function.
    integral_method : str, optional
        Integration method to use for computing the moments. Choices are:

        - 'lobatto' : use Lobatto quadrature (default)
        - 'trapezoidal' : use scypi.integrate.trapezoid method
        - 'simpson' : use scipy.integrate.simpson method
    """
    def __init__(self, nb_streams, integral_method='lobatto'):
        self.tr_method = 'DM'
        self.m_max = nb_streams
        self.integral_method = integral_method


class GT_trunc(object):
    """ 
    GT truncation, as in Iwabuchi and Suzuki (2009)

    Parameters
    ----------
    trunc_frac : float
        The truncature fraction
    integral_method : str, optional
        Integration method to use for computing the moments. Choices are:

        - 'lobatto' : use Lobatto quadrature (default)
        - 'trapezoid' : use scypi.integrate.trapezoid method
        - 'simpson' : use scipy.integrate.simpson method
    theta_tol : None | float, optional
        Search the truncated angle between 0 and theta_tol (in degrees). 
    theta_tr : None | float, optional
        Directly provide the truncated angle (in degrees). If provided, 
        trunc_frac and theta_tol are ignored.
    lobatto_optimization : bool, optional
        If True, use the optimized Lobatto quadrature for the integral.
        Reduce significantly the computional time in case theta_tr is not provided. 
    """
    def __init__(self, trunc_frac, integral_method='lobatto', theta_tol=None, theta_tr=None, 
                 lobatto_optimization=False):
        self.tr_method = 'GT'
        self.trunc_frac = trunc_frac
        self.integral_method = integral_method
        self.theta_tol = theta_tol
        self.theta_tr = theta_tr
        self.lobatto_optimization = lobatto_optimization