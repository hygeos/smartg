from smartg.smartg import Smartg
from smartg.atmosphere import AtmAFGL
from smartg.tools.smartg_view import smartg_view

from matplotlib import pyplot as plt

if __name__ == "__name__":
    """ Most basic SMARTG test """
    NBPHOTONS = 1e3
    m = Smartg().run(500., atm=AtmAFGL('afglms'), NBPHOTONS=NBPHOTONS)
    smartg_view(m)
    plt.savefig()