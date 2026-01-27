import numbers


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
    pha_scale_method : int, optional
        Scaling method to use for the truncated phase matrix. Choices are:

        - 1 : use Eq. 5 in Waquet et al. 2019 (Default)
        - 2 : use ARTDECO way, (same as 1, but with diffferent rescalling for F21 and F34)
    """
    def __init__(self, nb_streams, integral_method='lobatto', pha_scale_method=1):
        # check parameter values
        if ( isinstance(nb_streams, bool) or not 
             isinstance(nb_streams, numbers.Integral) or 
             nb_streams < 1 ):
            raise ValueError("The nb_streams parameter must be an integer >= 1.")
        integral_methods_ok = ['lobatto', 'trapezoid', 'simpson']
        if integral_method not in integral_methods_ok:
            raise ValueError(f"Choices for integral_method parameter are: {integral_methods_ok}.")
        if pha_scale_method not in [1, 2]:
            raise ValueError("Choices for pha_scale_method parameter are: 1 or 2.")

        self.tr_method = 'DM'
        self.m_max = nb_streams
        self.integral_method = integral_method
        self.pha_scale_method = pha_scale_method

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
    pha_scale_method : int, optional
        Scaling method to use for the truncated phase matrix. Choices are:

        - 1 : use Eq. 5 in Waquet et al. 2019 (Default)
        - 2 : use ARTDECO way, (same as 1, but with diffferent rescalling for F12 and F34)
    """
    def __init__(self, trunc_frac, integral_method='lobatto', theta_tol=None, theta_tr=None, 
                 lobatto_optimization=False, pha_scale_method=1):
        # check parameter values
        if (  ( isinstance(trunc_frac, bool) or not 
                isinstance(trunc_frac, numbers.Real) ) and
              ( 0. < trunc_frac < 1. )  ):
            raise ValueError("The trunc_frac parameter must be a scalar in the inteval ]0; 1[.")
        integral_methods_ok = ['lobatto', 'trapezoid', 'simpson']
        if integral_method not in integral_methods_ok:
            raise ValueError(f"Choices for integral_method parameter are: {integral_methods_ok}.")
        if theta_tol is not None:
            if (  ( isinstance(theta_tol, bool) or not 
                    isinstance(theta_tol, numbers.Real) ) and
                ( 0. < theta_tol < 180. )  ):
                raise ValueError("The theta_tol parameter must be a scalar in the inteval ]0; 180[.")
        if not isinstance(lobatto_optimization, bool):
            raise ValueError("The lobatto_optimization parameter must be a boolean.")
        if pha_scale_method not in [1, 2]:
            raise ValueError("Choices for pha_scale_method parameter are: 1 or 2.")
        
        self.tr_method = 'GT'
        self.trunc_frac = trunc_frac
        self.integral_method = integral_method
        self.theta_tol = theta_tol
        self.theta_tr = theta_tr
        self.lobatto_optimization = lobatto_optimization
        self.pha_scale_method = pha_scale_method