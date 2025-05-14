from smartg.smartg import Smartg
from smartg.atmosphere import AtmAFGL
from smartg.tools.smartg_view import smartg_view

from matplotlib import pyplot as plt

if __name__ == "__main__":
    
    print("Running SMARTG basic example")
    
    """ Most basic SMARTG test """
    NBPHOTONS = 1e8
    m = Smartg().run(500., THVDEG=45.,  atm=AtmAFGL('afglms'), NBPHOTONS=NBPHOTONS)
    fig = smartg_view(m)
    plt.savefig("fig.png")
    