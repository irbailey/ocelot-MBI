"""
@author: Igor Zagorodnov @ Martin Dohlus
Created on 27.03.2015
Revision on 01.06.2017: coordinate transform to the velocity direction
2019: Added LSC: S. Tomin and I. Zagorodnov
"""

import time
import numpy as np
from ocelot.common.globals import *
from ocelot.cpbd.elements import *
from ocelot.cpbd.coord_transform import *
from scipy import interpolate
from scipy.optimize import fmin
import scipy.special
import multiprocessing
from scipy.special import exp1, k1
from ocelot.cpbd.physics_proc import PhysProc
from ocelot.common.math_op import conj_sym
from ocelot.cpbd.beam import s_to_cur, global_slice_analysis
from ocelot.common import conf
from ocelot.cpbd.optics import get_map, lattice_transfer_map_z
from ocelot.cpbd.magnetic_lattice import MagneticLattice
import logging
import copy
from decimal import Decimal

try:
    from scipy.special import factorial
except:
    from scipy.misc import factorial    # legacy support

logger = logging.getLogger(__name__)

try:
    pyfftw_flag = True
    from pyfftw.interfaces.numpy_fft import fftn
    from pyfftw.interfaces.numpy_fft import ifftn
    import pyfftw
except:
    pyfftw_flag = False
    logger.debug("mbi.py: module PYFFTW is not installed. Install it to speed up calculation")
    from numpy.fft import ifftn
    from numpy.fft import fftn

try:
    import numexpr as ne
    ne_flag = True
except:
    logger.debug("mbi.py: module NUMEXPR is not installed. Install it to speed up calculation")
    ne_flag = False

'''
Derived parameters relevant for calculating MBI gain. Mostly based on Sci. Rep. 11.7895
https://www.nature.com/articles/s41598-021-87041-0
'''
# initial fractional uncorrelated energy spread rms
def sigd0(sigdE, E0):
    return sigdE/E0
# Half-Amplitude of the LH-induced energy modulation, in [MeV]
def DeltaE_lh(sigdE_lh):
    return 2*sigdE_lh

# Half length of the compressor
def halfL(Lb, DL):
    halfL = DL+Lb
# halfL2 = DL2+Lb2;
# maximum dispersion function in the chicane, in [m]
def eta_max(theta, Lb, DL):
    return theta*(halfL(Lb, DL))
# R56 of DS1-chicane in [m]
def R56(theta, Lb, DL):
    return -2*(theta**2)*(((2/3)*Lb)+DL)
# bunch length
def bunch_length(Q, I0):
    return speed_of_light*Q/(5.5*I0)

# Lorentz factor for mean energy along D0
def gamma(energy, linac={}):
    linac.update({'gradient': linac['energy_final'] - linac['energy_initial'] / linac['length']})
    linac.update({'gamma_final': linac['energy_final'] / m_e_MeV})
    linac.update({'gamma_mean': (linac['gamma_final'] - linac['energy_initial'] / m_e_MeV) / 2})
    return linac

# Lorentz factor for mean energy along linac
def gamma_along_linac(distance, linac={}):
    linac.update({'gradient': float(linac['energy_final'] - linac['energy_initial']) / linac['length']})
    linac.update({'gamma_final': linac['energy_final'] / m_e_MeV})
    linac.update({'gamma_mean': (linac['gamma_final'] - linac['energy_initial'] / m_e_MeV) / 2})
    linac.update({'gamma': (linac['energy_initial'] + (distance * linac['gradient'])) / m_e_MeV})

# uncompressed wave number in [1/m]
def k_wn(lamb):
    return 2*np.pi/lamb
# initial bunch factor form shot noise
def b0(lamb, I0):
    return np.sqrt(q_e*speed_of_light/(I0*lamb))
def h(theta, compression_factor, Lb, DL):
    if R56(theta, Lb, DL) < 0:
        return abs((1 - 1/compression_factor)/R56(theta, Lb, DL))
    else:
        return 0

def sigd(sigd1, sigd2):
    return np.sqrt(sigd1**2 + sigd2**2)
def sigd_normalised(sig_energies, energy):
    sige_total = []
    for i in sig_energies:
        sige_total.append(i ** 2)
    return np.sqrt((sum(sige_total))) / energy

# correlated fractional energy RMS spread at BC entrance
def sigd_cor(theta, compression_factor, Lb, DL, I0, Q):
    h1 = h(theta, compression_factor, Lb, DL)
    return abs(h1*Q*speed_of_light/(I0*np.sqrt(2*np.pi)))
# horizontal H-function at the exit of first dipole chicane, in [m]
def H1(theta, beta):
    return ((theta**2)*beta)
# horizontal H-function at the exit of second+third dipole of chicane, in [m]
def H2(theta, beta, Lb, Dl):
    return ((theta**2)*beta)+(2*(eta_max(theta, Lb, Dl)**2)/beta)
# constant factor in Bane's approximation
def D_bane(Q, bunchlength, enx):
    return (ro_e**2)*(Q/q_e)/(8*bunchlength*enx)

def compressed_current(I0, compression_factor):
    return I0 * compression_factor

def geometric_emittance(enx, energy):
    return enx/(energy/m_e_GeV)

def eta_p(theta):
    return theta
def eta(theta, DL, Lb):
    return theta*(DL+Lb);
def gamma_twiss(alpha, beta):
    return (1+(alpha**2))/beta;

A_csr = -0.94 + 1.63j

class IBS(PhysProc):
    """
    Longitudinal Space Charge
    smooth_param - 0.1 smoothing parameter, resolution = np.std(p_array.tau())*smooth_param
    """
    def __init__(self, step=1):
        PhysProc.__init__(self, step)
        self.smooth_param = 0.1
        self.step_profile = False
        self.napply = 0
        self.num_slice = 1
        self.lattice = None
        self.first = True
        #self.lattice = None
        self.zpos = 0
        self.slice = None
        self.dist = []
        self.slice_params = []
        self.optics_map = []

    def get_slice_params(self, p_array, nparts_in_slice=5000, smooth_param=0.05, filter_base=2, filter_iter=2):
        slice_params = global_slice_analysis(p_array)
        sli_cen = len(slice_params.s)/2
        sli_tot = len(slice_params.s)
        sli_min = int(sli_cen - (sli_tot/2))
        sli_max = int(sli_cen + (sli_tot/2))
        sli = {}
        for k, v in slice_params.__dict__.items():
            if k in ['beta_x', 'alpha_x', 'beta_y', 'alpha_y', 'I', 'sig_x', 'sig_y']:
                sli.update({k: np.mean(v[sli_min: sli_max])})
            elif k == 'se':
                sli.update({'sdelta': np.mean(v[sli_min: sli_max] / slice_params.me[sli_min: sli_max])})
            elif k == 'me':
                sli.update({'me': np.mean(v[sli_min: sli_max] / 1e9)})
                sli.update({'gamma': np.mean(v[sli_min: sli_max] / 1e6 / m_e_MeV)})
            elif k in ['ex', 'ey']:
                sli.update({k: np.mean(v[sli_min: sli_max])})
        sli.update({'s': p_array.s})
        sli.update({'q': np.sum(p_array.q_array)})
        sli.update({'bunch_length': 2.355 * np.std(p_array.tau())})
        return sli

    def apply(self, p_array, dz):
        """
        wakes in V/pC

        :param p_array:
        :param dz:
        :return:
        """
        if dz < 1e-10:
            logger.debug(" LSC applied, dz < 1e-10, dz = " + str(dz))
            return
        logger.debug(" LSC applied, dz =" + str(dz))
        p_array_c = copy.deepcopy(p_array)
        mean_b = np.mean(p_array_c.tau())
        sigma_tau = np.std(p_array_c.tau())
        slice_min = mean_b - sigma_tau / 2.5
        slice_max = mean_b + sigma_tau / 2.5
        self.slice_params.append(self.get_slice_params(p_array_c))
        self.z0 = self.slice_params[-1]['s'] - self.slice_params[0]['s']
        self.ltm, self.elem = lattice_transfer_map_z(self.lattice, self.slice_params[0]['me'], self.z0)
        self.optics_map.append(self.ltm)
        self.dist.append(self.z0)
        print('\n')
        if len(self.slice_params) > 1:
            self.distance = (self.slice_params[-1]['s'] - self.slice_params[-2]['s'])
            self.sigd = self.sigd_ibs(self.slice_params, self.optics_map, self.distance, self.elem)
            self.sigdvals = np.random.normal(0, self.sigd, len(p_array.rparticles[5]))
            p_array.rparticles[5] += self.sigdvals
            print(f"sdelta {self.slice_params[-1]['sdelta']} sigd {self.sigd}")

    def qmax(self, slice_params, distance):
        '''
        Eq. 22: Coulomb log function in terms of min/max scattering angle

        :param sliceparams: beam slice parameters
        :param distance: distance travelled

        :return: [log]
        '''
        numer = distance * (slice_params[-1]['q'] / q_e) * (ro_e**2)
        denom = 2 * (slice_params[-1]['gamma'] * slice_params[-1]['ex'])**1.5 * slice_params[-1]['bunch_length'] * np.sqrt(slice_params[-1]['beta_x'])
        return np.sqrt(numer / denom)

    def coulomb_log(self, slice_params, distance):
        return np.log(self.qmax(slice_params, distance) * np.sqrt(slice_params[-1]['ex'] * slice_params[-1]['ey']) / (2 * np.sqrt(2) * ro_e))

    def gradient(self, slice_params, distance):
        return (slice_params[-1]['gamma'] - slice_params[-2]['gamma']) / (m_e_MeV * distance)

    def ibs_k(self, slice_params, distance):
        '''
        Eq. 28: factor for accelerating gradient

        :param slice_params: beam slice parameters
        :param distance: distance travelled

        :return: k
        '''
        grad = self.gradient(slice_params, distance)
        numer = ro_e * (slice_params[-1]['q'] / q_e) * m_e_MeV
        denom = 4 * grad * (slice_params[-1]['ex']**1.5) * (slice_params[-1]['beta_x']**0.5) * slice_params[-1]['bunch_length']
        return numer / denom

    def sigd_ibs(self, slice_params, optics_map, distance, elem):
        '''
        Eq. 29: increase in sigma_delta

        :param slice_params: beam slice parameters
        :param distance: distance travelled

        :return: sigd
        '''
        if elem.__class__ == Cavity:
            # db = D_bane(slice_params[-1]['q'], slice_params[-1]['bunch_length'], slice_params[-1]['ex'])
            # banefactor = m_e_eV * db / (self.gradient(slice_params, distance) * np.sqrt(slice_params[-1]['ex'] * slice_params[-1]['beta_x']))
            # gammadifffactor = (1 / np.sqrt(slice_params[-2]['gamma']) - 1 / np.sqrt(slice_params[-1]['gamma']))
            # # clog = self.coulomb_log(slice_params, distance)
            # clog = self.coulomb_log(slice_params, elem.l)
            # loggamfcub = np.log(slice_params[-1]['gamma'] ** 3)
            # gamfsqrt = np.sqrt(slice_params[-1]['gamma'])
            # loggamicub = np.log(slice_params[-2]['gamma'] ** 3)
            # gamisqrt = np.sqrt(slice_params[-2]['gamma'])
            # return np.sqrt(abs(banefactor * (gammadifffactor * (4 * clog - 6) + (loggamfcub / gamfsqrt - loggamicub / gamisqrt))))
            db = D_bane(slice_params[-1]['q'], slice_params[-1]['bunch_length'], slice_params[-1]['ex'])
            grad = self.gradient(slice_params, distance)
            gam1 = slice_params[-1]['gamma']
            gam0 = slice_params[-2]['gamma']
            fac1 = (2 * m_e_MeV * db / (3 * (gam1**2) * grad * np.sqrt(slice_params[-1]['ex'] * slice_params[-1]['beta_x'])))
            fac21 = gam1**(1.5) - gam0**(1.5)
            fac22 = 2 * self.coulomb_log(slice_params, elem.l) * (fac21)
            fac23 = (gam0**(1.5) * np.log(gam0**(0.75))) - (gam1**(1.5) * np.log(gam1**(0.75)))
            return np.sqrt(fac1 * (fac21 + fac22 + fac23))
            # sqrt((2 * me * D / (3 * gammaf1 ^ 2 * G1 * sqrt(enx * betax1))) * (
            #             gammaf1 ^ (3 / 2) - gamma0 ^ (3 / 2) + 2 * ln1 * (
            #                 gammaf1 ^ (3 / 2) - gamma0 ^ (3 / 2)) + gamma0 ^ (3 / 2) * log(
            #         gamma0 ^ (3 / 4)) - gammaf1 ^ (3 / 2) * log(gammaf1 ^ (3 / 4))));

        elif (self.dispersion_invariant_x(slice_params, optics_map) > 1e-6):
            afacn = (ro_e ** 2 * slice_params[-1]['q'] / q_e)
            afacd = 4 * (slice_params[-1]['gamma'] ** 2) * slice_params[-1]['ex'] * np.sqrt(
                slice_params[-1]['sig_x'] * slice_params[-1]['sig_y']) * slice_params[-1]['bunch_length']
            afac = afacn / afacd
            bfac = np.sqrt(distance * afac) * slice_params[-1]['ex'] / 2 / ro_e
            hfac = slice_params[-1]['gamma'] * self.dispersion_invariant_x(slice_params, optics_map) / slice_params[-1]['ex']
            sigdibs = (scipy.optimize.fmin(self.solBC, x0=slice_params[-1]['sdelta'], args=(slice_params[-2]['sdelta'], hfac, afac, bfac, distance)))
            return 0*np.sqrt(abs((sigdibs ** 2) - slice_params[-1]['sdelta'] ** 2))
        else:
            db = D_bane(slice_params[-1]['q'], slice_params[-1]['bunch_length'], slice_params[-1]['ex'])
            numer = 2 * db * distance * self.coulomb_log(slice_params, distance)
            denom = (slice_params[-1]['gamma'])**1.5 * np.sqrt(slice_params[-1]['ex'] * slice_params[-1]['beta_x'])
            return np.sqrt((numer/denom))


    def dispersion_invariant_x(self, slice_params, optics_map):
        '''
        Eq. 8: H-functions

        :param sigmamatrix: SDDSobject containing beam sigma matrices
        :param beammatrix: SDDSobject containing transport matrices
        :param index: index in lattice

        :return: Hx
        '''
        r16sq = optics_map[-1][0, 5] ** 2
        betax = slice_params[-1]['beta_x']
        alphax = slice_params[-1]['alpha_x']
        numerator = r16sq + ((betax * optics_map[-1][1, 5]) + (alphax * optics_map[-1][1, 5])) ** 2
        return numerator / betax

    def dispersion_invariant_y(self, slice_params, optics_map):
        '''
        Eq. 8: H-functions

        :param sigmamatrix: SDDSobject containing beam sigma matrices
        :param beammatrix: SDDSobject containing transport matrices
        :param index: index in lattice

        :return: Hy
        '''
        r36sq = optics_map[-1][2, 5] ** 2
        betay = slice_params[-1]['beta_x']
        alphay = slice_params[-1]['alpha_x']
        numerator = r36sq + ((betay * optics_map[-1][4, 5]) + (alphay * optics_map[-1][3, 5])) ** 2
        return numerator / betay

    def solBC(self, y, sigd, hBC, aBC, bBC, distance):
        logfactor1 = 3 * np.log(np.sqrt(hBC * (sigd ** 2) + 1) / bBC)
        logfactor2 = 3 * np.log(np.sqrt(hBC * (y ** 2) + 1) / bBC)
        expfac1 = -scipy.special.expi(logfactor1)
        expfac2 = scipy.special.expi(logfactor2)
        return abs((2 * (bBC**3) / hBC) * (expfac1 + expfac2) - (aBC * distance))