import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pylab import subplots, setp

import sys
import imp
if sys.version[0] == '2':
    imp.reload(sys)
    sys.setdefaultencoding('utf8')
    import subprocess
else:
    from subprocess import check_output
    
# import sys
# sys.path.append('..')

from smartg.smartg import Smartg
from smartg.smartg import LambSurface, RoughSurface, CusForward
from smartg.atmosphere import AtmAFGL, AeroOPAC
from smartg.water import IOP_1

# from smartg import Smartg, reptran_merge
# from smartg import RoughSurface, LambSurface, FlatSurface, Environment
# from smartg import Profile, AeroOPAC, CloudOPAC, IOP_SPM, IOP_MM, IOP_AOS_WATER

from smartg.tools.tools import SpecInt, SpecInt2, Irr
from luts.luts import LUT, MLUT, Idx, merge, read_mlut_hdf, plot_polar, read_mlut
from smartg.tools.smartg_view import smartg_view, input_view
import numpy as np
from warnings import warn
from warnings import filterwarnings


option_save_pdf = True


##########################################################################################
# Some functions (DON'T NEED TO BE MODIFIED)
##########################################################################################
def compute_err (exact, approx, name_err):
    '''
    Args:
    - exact    : vector with the reference values
    - approx   : vector with the simulation values
    - name_err : name of the error we want to compute
                 RMSE -> root mean square of the absolute error
                 RRMSE -> root mean square of the relative error
                 MAE -> Mean Absolute Error
                 MAPE -> Mean Absolute Pourcentage Error
                 L1 -> Norm L1
                 L2 -> Norm L2
                 Linf -> Norm Linf
    Output:
    - A scalar which is the error of one of the listed error name
    '''
    res = 0.
    error = exact[:] - approx[:]
    relative_err = (exact[:]-approx[:])/exact[:]
    if name_err == 'RMSE':
        for i in range (len(error)):
            res = res + error[i]*error[i]
        res = res*(1./len(error))
        res = res**(1./2.)
    elif name_err == 'RRMSE':
        for i in range (len(exact)):
            res = res + relative_err[i]*relative_err[i]
        res = res*(1./len(exact))
        res = (res**(1./2.))*100 # res in pourcentage
    elif name_err == 'MAE':
        for i in range (len(error)):
            res = res + abs(error[i])
        res = res*(1./len(error))
    elif name_err == 'MAPE':
        for i in range (len(error)):
            res = res + abs(relative_err[i])
        res = res*(1./len(error))*100 # res in pourcentage
    elif name_err == 'L1': 
        for i in range (len(error)):
            res = res + abs(error[i])
    elif name_err == 'L2':
        for i in range (len(error)):
            res = res + error[i]*error[i]
        res = res**(1./2.)
    elif name_err == 'Linf':
        res = max(abs(error))
    else:
        raise NameError('Unknow error name for computing error!')
    return res


def compare_view3(mlut, mref, field='up (TOA)', vmax=0.5, vmin=0, emax=1e-1):
    '''
    Function (created by Didier) which print a more global comparaison
    between reference and simulation
    
    Args:
    - mlut  : LUT table from the simulation
    - mref  : LUT table from the reference
    - field : the field we want to compare (ex: field='up (TOA)')
    - vmax  : max value of graphs
    - vmin  : min value of graphs
    - emax  : gap between vmax, vmin

    Output:
    A list of matplotib figures (A figure for each wl)
    '''
    stokes = ['I','Q','U']
    sign = [1,1,-1,1] # sign convention for first half space
    lam=mlut['I_up (TOA)'].axes[0]
    N = mlut['N' + '_' + field]
    fig_l=[]
    for k in range(4):
        phi_pp = 0.0
        fig,ax = subplots(3,4, sharey=False,sharex=True)
        fig.set_size_inches(15, 6)
        #fig.set_dpi=300 # must be comment for pdf save
    
        for i in range(4):

            if i!=3 :
                S = mlut[stokes[i] + '_' + field].sub()[k,:,:]
                E = mlut[stokes[i] + '_stdev_' + field].sub()[k,:,:]
                th = S.axes[1]                  
                Sref = mref[stokes[i] + '_' + field].sub()[k,:,:]
                S.desc = r'$' + S.desc[0:3]  + S.desc[4:] + '-%.0fnm$'%lam[k]
               
            else:
                I=mlut[stokes[0] + '_' + field].sub()[k,:,:]
                Q=mlut[stokes[1] + '_' + field].sub()[k,:,:]
                U=mlut[stokes[2] + '_' + field].sub()[k,:,:]
                EI=mlut[stokes[0] + '_stdev_' + field].sub()[k,:,:]
                EQ=mlut[stokes[1] + '_stdev_' + field].sub()[k,:,:]
                EU=mlut[stokes[2] + '_stdev_' + field].sub()[k,:,:]
                S= (((Q*Q+U*U).apply(np.sqrt))/I) * 100
                S.desc= r'$DoLP' + I.desc[1:3] + I.desc[4:] + '-%.0fnm$'%lam[k]
                Iref=mref[stokes[0] + '_' + field].sub()[k,:,:]
                Qref=mref[stokes[1] + '_' + field].sub()[k,:,:]
                Uref=mref[stokes[2] + '_' + field].sub()[k,:,:]
                Sref= (((Qref*Qref+Uref*Uref).apply(np.sqrt))/Iref) * 100
                #E = ((EI/I) + (EQ/Q.apply(abs)) + (EU/U.apply(abs))) * S
                E = (1./N.apply(np.sqrt).sub()[k,:,:]) * 3 * S
                            

            if i==0 : 
                vmi=0
                vma=vmax
            else: vmi=-vmax
            if i==3 :
                vmi=0.
                vma=100.
                ema=emax*1e3
            else:
                ema=emax
                

            for phi0,sym1,sym2,labref in [(phi_pp,'r','.','ref'),(120.,'g','.','')]:

                
                # both points at their own abscissas
                refp = Sref[Idx(phi0,round=True),:] # reference for >0 view angle
                refm = Sref[Idx(180.-phi0,round=True),:] #      reference for <0 view angle
                sp   = sign[i]*S[Idx(180.-phi0),:]       #     simulation for >0 view angle
                sm   = S[Idx(phi0),:]
                ep   = E[Idx(180.-phi0),:]
                em   = E[Idx(phi0),:]

                ax[0,i].plot(th, refp,'k'+sym2)
                ax[0,i].plot(-th,refm,'k'+sym2, label=labref)
                ax[0,i].errorbar(th, sp,fmt=sym1+'-')
                ax[0,i].errorbar(-th,sm,fmt=sym1+'-', \
                            label=r'smartg $\Phi=%.0f-%.0f$'%(phi0,180.-phi0))
                ax[0,i].set_ylim([vmi, vma])
                ax[0,i].set_xlim([-89., 89.])

                ax[1,i].errorbar(th,sp-refp,yerr=ep,\
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor='k')
                ax[1,i].errorbar(-th,sm-refm,yerr=em,fmt=sym1+sym2,ecolor='k')                
                ax[1,i].set_ylim([-ema,ema])
                ax[1,i].set_xlim([-89., 89.])  

                ax[2,i].errorbar(th,(sp-refp)/refp*100,\
                                 fmt=sym1+sym2,label=r'$\Phi=%.0f-%.0f$'%(phi0,180.-phi0),ecolor='k')
                ax[2,i].errorbar(-th,(sm-refm)/refm*100,fmt=sym1+sym2,ecolor='k')

                #if i!=3 : ax[2,i].set_ylim([-ema*500,ema*500])
                #else : ax[2,i].set_ylim([-ema,ema])
                ax[2,i].set_ylim([-0.5,0.5])
                    
                ax[2,i].set_xlim([-89., 89.])    
                ax[1,i].plot([-89,89],[0.,0.],'k--')
                ax[2,i].plot([-89,89],[0.,0.],'k--')
                #ax[0,i].set_title(stokes[i] + '_'+field)
                ax[0,i].set_title(S.desc)

                if i==2: 
                    ax[0,i].legend(loc='best',fontsize = 7,labelspacing=0.0)
                if i==0:
                    ax[1,i].text(-50.,ema*0.75,r'$N_{\Phi}$:%i, $N_{\theta}$:%i'%\
                             (S.axes[0].shape[0],S.axes[1].shape[0]))
                    ax[1,i].set_ylabel(r'$\Delta$')
                    ax[2,i].set_ylabel(r'$\Delta (\%)$')

        fig_l.append(fig)
    return fig_l


class Compare_error(object):
    ind_plot = 0
    def __init__(self, simlut, reflut, rows, cols):
        '''
        Initialisation of the class Compare_error

        This class is used in order to compare the error (see the function 
        compute_error) from  wl (wave lenght) = 350, 450, 550 and 560 of a
        given test case.

        Args:
        - simlut : simulation LUT table
        - reflut : reference LUT table
        - rows   : the number of rows of the figure size (graphs ploting)
        - cols   : the number of columns of the figure size
        '''
        Compare_error.ind_plot = 0
        self.simlut = simlut
        self.reflut = reflut
        self.fig = plt.figure (figsize = (4*cols, 3*rows))
        self.rows = rows
        self.cols = cols
            
    
    def compare(self, field_Stoke, nerr, seuil):
        '''
        This function will serve to plot the error

        Args:
        - field_Stoke : the fiels + stokes we want to use
                        (exemple: field ='I_up (TOA)')
        - nerr        : The error name (see the function compute_error)
        - seuil       : Error treshold/limit of the error
        '''

        filterwarnings ('once', '.*Warning.*',)
        Compare_error.ind_plot += 1
        wl=[350.0525, 450.0666, 550.084, 650.099]
        wl_paper=[350,450,550,650]
        ax = plt.subplot(self.rows, self.cols, self.ind_plot)
        red_err_vec = np.zeros(len(wl), dtype=np.float32)
        green_err_vec = np.zeros(len(wl), dtype=np.float32)
        analysis = seuil

        for k in range(len(wl)):
            Sref = self.reflut[field_Stoke].sub()[k,:,:]
            Ssim = self.simlut[field_Stoke].sub()[k,:,:]
            for indice2, value2 in enumerate([0., 120.]):
                refP = Sref[Idx(value2, round=True),:]       # reference for  > 0 view angle (p for positive)
                if field_Stoke[:1] != "U":                   # simulation for > 0 view angle (p for positive)
                    simP = Ssim[Idx(180-value2),:]
                else:
                    simP = -1.*Ssim[Idx(180-value2),:]
                refN = Sref[Idx(180-value2,round=True),:]    # reference for  < 0 view angle (n for negative)
                simN = Ssim[Idx(value2),:]                   # simulation for < 0 view angle (n for negative)

                Tref = np.append(refN, refP)                 # regroup negative and positive ref values
                Tsim = np.append(simN, simP)                 # regroup negative and positive sim values

                if value2 == 0.:
                    red_err_vec[k] = compute_err(Tref, Tsim, nerr)
                    if (red_err_vec[k] > analysis):
                        warn ('Warning! ' + field_Stoke + ' computing ' + nerr + ': Wrong' \
                              ' values for wl=%d' % wl_paper[k] + ' and Phi=%.0f-%.0f' %(0,180))
                if value2 == 120.:
                    green_err_vec[k] = compute_err(Tref, Tsim, nerr)
                    if (green_err_vec[k] > analysis):
                        warn ('Warning! ' + field_Stoke + ' computing ' + nerr + ': Wrong' \
                              ' values for wl=%d' % wl_paper[k] + ' and Phi=%.0f-%.0f' %(120,60))
        ax.plot(wl_paper, red_err_vec,'r^', linestyle='-', label=r'smartg $\Phi=%.0f-%.0f$'%(0.,180.))
        ax.plot(wl_paper, green_err_vec,'g.', linestyle='-', label=r'smartg $\Phi=%.0f-%.0f$'%(120.,60.))
        ax.set_xticks(np.asarray(wl_paper))
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(10) 
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.legend(loc='best',fontsize = 7, labelspacing=0.0)
        ax.set_title(field_Stoke)
        if nerr == 'MAPE' or nerr == 'RRMSE':
            ax.set_ylabel('$\Delta (\%)$ ('+ nerr + ')')
        else:
            ax.set_ylabel('$\Delta$ ('+ nerr + ')')

    def save(self, pdf_name):
        '''
        Arg:
        - pdf_name : The name of the pdf (ex: pdf_name = 'test.pdf')
        '''
        plt.savefig(pdf_name)

    def cfig(self):
        plt.close(self.fig)
##########################################################################################


##########################################################################################
# Test cases used to validate the SMART-G code (Accuracy) MODIFICATIONS HERE
##########################################################################################
def test_val_ray_surf():
    '''
    Validation test case with Rayleigh atm + surface
    '''
    wl=[350.0525, 450.0666, 550.084, 650.099]
    wl_paper=[350,450,550,650]
    list_compare = ['MAE', 'RMSE', 'MAPE', 'RRMSE', 'L1', 'L2', 'Linf']
    stokes_TOA = ['I_up (TOA)', 'Q_up (TOA)', 'U_up (TOA)']
    stokes_Oplus = ['I_up (0+)', 'Q_up (0+)', 'U_up (0+)']

    atm=AtmAFGL('afglms',lat=0., O3=0., NO2=False)
    surf=RoughSurface(SUR=1,NH2O=1.34,WIND=7.)

    ml30=read_mlut_hdf('auxdata/validation/ml30_AOS_I')
    ml60=read_mlut_hdf('auxdata/validation/ml60_AOS_I')

    NBPHOTONS=1e6
    S=Smartg(double=True, debug_photon=False)
    ##########
    phi = np.array(ml30['I_up (0+)'].axes[1],dtype=np.float32)
    th = np.array(ml60['I_up (0+)'].axes[2],dtype=np.float32)
    le={}
    le.update(phi=phi*np.pi/180)
    le.update(th=(th*np.pi/180))

    m30=S.run(XGRID=128, XBLOCK=64, THVDEG=30., wl=wl, NBPHOTONS=NBPHOTONS, DEPO=0., le=le, NBLOOP=NBPHOTONS/1e2,
               atm=atm, surf=surf, OUTPUT_LAYERS=3, stdev=True)

    m60=S.run(THVDEG=60., wl=wl, NBPHOTONS=NBPHOTONS, DEPO=0., le=le, NBLOOP=NBPHOTONS/1e2,
               atm=atm, surf=surf, OUTPUT_LAYERS=3, stdev=True)

    if option_save_pdf:
        h_l = compare_view3(m30, ml30, field='up (TOA)',vmax=0.4,emax=5e-4)
        with PdfPages('AOS1_ts30_TOA.pdf') as pdf:
            for k in range(len(wl)):
                pdf.savefig(h_l[k])
                plt.close(h_l[k])

        h_l = compare_view3(m60, ml60, field='up (TOA)',vmax=0.4,emax=5e-4)
        with PdfPages('AOS1_ts60_TOA.pdf') as pdf:
            for k in range(len(wl)):
                pdf.savefig(h_l[k])
                plt.close(h_l[k])

        h_l = compare_view3(m30, ml30, field='up (0+)',vmax=0.4,emax=5e-4)
        with PdfPages('AOS1_ts30_surface(O+).pdf') as pdf:
            for k in range(len(wl)):
                pdf.savefig(h_l[k])
                plt.close(h_l[k])

        h_l = compare_view3(m60, ml60, field='up (0+)',vmax=0.4,emax=5e-4)
        with PdfPages('AOS1_ts60_surface(O+).pdf') as pdf:
            for k in range(len(wl)):
                pdf.savefig(h_l[k])
                plt.close(h_l[k])

    C = Compare_error(m30, ml30, 4, 3)
    for k in stokes_TOA:
        C.compare(k, 'MAE', 5e-3)
    for k in stokes_TOA:
        C.compare(k, 'L1', 3e-2)
    for k in stokes_TOA:
        C.compare(k, 'L2', 8e-3)
    for k in stokes_TOA:
        C.compare(k, 'Linf', 5e-3)
    if option_save_pdf:
        C.save('err_analysis_ts30_ray_TOA.pdf')
    C.cfig()

    C = Compare_error(m30, ml30, 4, 3)
    for k in stokes_Oplus:
        C.compare(k, 'MAE', 5e-3)
    for k in stokes_Oplus:
        C.compare(k, 'L1', 3e-2)
    for k in stokes_Oplus:
        C.compare(k, 'L2', 8e-3)
    for k in stokes_Oplus:
        C.compare(k, 'Linf', 5e-3)
    if option_save_pdf:
        C.save('err_analysis_ts30_ray_surface(O+).pdf')
    C.cfig()


# def test_val_oce_surf():
#     '''
#     Validation test case with ocean + surface
#     '''
#     wl=[350.0525, 450.0666, 550.084, 650.099]
#     wl_paper=[350,450,550,650]
#     stokes_Oplus = ['I_up (0+)', 'Q_up (0+)', 'U_up (0+)']

#     water=IOP_1(chl=1.) #IOP_AOS_WATER()
#     surf=RoughSurface(SUR=3,NH2O=1.34,WIND=7.)

#     ml30=read_mlut('auxdata/validation/ml30_AOS_II.nc')
#     ml60=read_mlut('auxdata/validation/ml60_AOS_II.nc')

#     S1=Smartg(double=True)
#     NBPHOTONS=1e6
#     NBLOOP=NBPHOTONS/100
#     DEPO = 0.
#     phi = np.array(ml30['I_up (0+)'].axes[1]*np.pi/180,dtype=np.float32)
#     th = np.array(ml60['I_up (0+)'].axes[2]*np.pi/180,dtype=np.float32)
#     le={}
#     le.update(phi=phi)
#     le.update(th=th)
#     m30le=S1.run(THVDEG=30., wl=wl, NBPHOTONS=NBPHOTONS, DEPO=DEPO, le=le,
#             water=water,surf=surf, OUTPUT_LAYERS=3, stdev=True)
#     m60le=S1.run(THVDEG=60., wl=wl, NBPHOTONS=NBPHOTONS, DEPO=DEPO, le=le,
#             water=water,surf=surf, OUTPUT_LAYERS=3, stdev=True)
    
#     if option_save_pdf:
#         h_l=compare_view3(m30le, ml30,field='up (0+)',vmax=0.1,emax=1e-3)
#         with PdfPages('AOSII_ts30_surface(O+).pdf') as pdf:
#             for k in xrange(len(wl)):
#                 pdf.savefig(h_l[k])
#                 plt.close(h_l[k])

#         h_l=compare_view3(m60le, ml60,field='up (0+)',vmax=0.1,emax=1e-3)
#         with PdfPages('AOSII_ts60_surface(O+).pdf') as pdf:
#             for k in xrange(len(wl)):
#                 pdf.savefig(h_l[k])
#                 plt.close(h_l[k])

#     C = Compare_error(m30le, ml30, 4, 3)
#     for k in stokes_Oplus:
#         C.compare(k, 'MAE', 5e-3)
#     for k in stokes_Oplus:
#         C.compare(k, 'L1', 1.5e-1)
#     for k in stokes_Oplus:
#         C.compare(k, 'L2', 3e-2)
#     for k in stokes_Oplus:
#         C.compare(k, 'Linf', 1.8e-2)
#     if option_save_pdf:      
#         C.save('err_analysis_ts30_oce_surface(O+).pdf')
#     C.cfig()

##########################################################################################


if __name__ == '__main__':

    test_val_ray_surf()
    #test_val_oce_surf()
