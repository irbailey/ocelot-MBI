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
    return np.sqrt(2*q_e*speed_of_light/(I0*lamb))
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
def D_bane(Q, I0, enx):
    return (ro_e**2)*(Q/q_e)/(8*bunch_length(Q, I0)*enx)

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

class MBI(PhysProc):
    """
    Longitudinal Space Charge
    smooth_param - 0.1 smoothing parameter, resolution = np.std(p_array.tau())*smooth_param
    """
    def __init__(self, step=1, lamb_range=[1e-6, 50e-6, 10]):
        PhysProc.__init__(self, step)
        self.smooth_param = 0.1
        self.step_profile = False
        self.napply = 0
        self.lsc = False
        self.csr = False
        self.ibs = False
        self.num_slice = 1
        self.first = True
        self.lamb_range = lamb_range
        #self.lattice = None
        self.zpos = 0
        self.dist = []
        self.bf = [[] for i in range(len(self.lamb_range))]
        self.pf = [[] for i in range(len(self.lamb_range))]
        self.slice_params = []
        self.optics_map = []

    def set_lamb_range(self, lamb_range):
        self.lamb_range = lamb_range
        self.bf = [[]] * (len(self.lamb_range))
        self.pf = [[]] * (len(self.lamb_range))

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
                sli.update({k: np.mean(v[sli_min: sli_max]) / np.mean(slice_params.me[sli_min: sli_max])})
        sli.update({'s': p_array.s})
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
        print(self.ltm)
        # print(self.optics_map[-1][4,5])
        # print(self.z0)
        for i, l in enumerate(self.lamb_range):
            self.ld_0s = self.ld0s(l, self.slice_params, self.optics_map)
            self.b0fac = b0(l, self.slice_params[0]['I']) * self.ld_0s
            if i == 1:
                print(f'b0 {self.b0fac} i {self.slice_params[-1]["I"]} ld0s {self.ld_0s}')
            self.bf[i].append(self.b0fac)
            if len(self.slice_params) > 1:
                self.k0tot = []
                self.k1tot = []
                self.k2tot = []
                for j, b in enumerate(self.bf[i]):
                    #if i == 0:
                        #print(f'j {j} b {b} lenslipar {len(self.slice_params)}')
                    # if j < len(self.slice_params):
                    self.fac = 0.5 if j == 0 else 1
                    self.distance = self.slice_params[-1]['s'] if j == 0 else (self.slice_params[-1]['s'] - self.slice_params[-2]['s'])
                    # self.k0r = self.fac * self.kernel_K0(l, self.slice_params, self.optics_map, j)
                    # self.k0tot.append(((self.k0r.real * b) + (self.k0r.imag * b)))
                    self.k1r = self.distance * self.fac * self.kernel_K1(l, self.slice_params, self.optics_map, j, self.elem)
                    self.k1tot.append(abs((self.k1r.real * b) + (self.k1r.imag * b)))
                    # self.k2r = self.fac * self.kernel_K2(l, self.slice_params, self.optics_map, j)
                    # self.k2tot.append(abs((self.k2r.real * b) + (self.k2r.imag * b)))
                #print(f'{dz} {l} {np.sum(self.k1tot)} {self.slice_params[-1]["I"]}')
                # print('here')
                # print(self.bf[i][-1])
                # self.bf[i][-1] += np.nansum(self.k0tot)
                # print(self.bf[i][-1])
                if i == 11:
                    # print(i)
                    # print(self.k1tot)
                    print(np.nansum(self.k1tot))
                self.bf[i][-1] += np.nansum(self.k1tot)
                # self.bf[i][-1] += np.nansum(self.k2tot)
                # self.bf[i] = self.bf[i].tolist()
        # print(np.transpose(self.bf)[-1])

    def ld0s(self, lamb, slice_params, optics_map):
        '''
        Eq. 46: Landau damping between first lattice point and subsequent points

        :param lamb: initial modulation wavelength
        :param sigmamatrix: SDDSobject containing beam sigma matrices
        :param beammatrix: SDDSobject containing transport matrices
        :param index: index to which to calculate

        :return: Landau damping factor LD(0->s)
        '''
        compfac = slice_params[-1]['I'] / slice_params[0]['I']
        compfac1 = 1# if compfac < 1 else compfac
        kfac = -(k_wn(lamb * compfac) ** 2) / 2
        ex = slice_params[0]['ex']
        ey = slice_params[0]['ey']
        betax = slice_params[0]['beta_x']
        betay = slice_params[0]['beta_y']
        alphax = slice_params[0]['alpha_x']
        alphay = slice_params[0]['alpha_y']
        sigd = slice_params[0]['sdelta']
        r51s = optics_map[-1][4,0]
        r52s = optics_map[-1][4,1]
        r53s = optics_map[-1][4,2]
        r54s = optics_map[-1][4,3]
        r56s = optics_map[-1][4,5]
        r51r52 = (r51s - ((alphax / betax) * r52s)) ** 2
        r52 = r52s ** 2
        r53r54 = (r53s - ((alphay / betay) * r54s)) ** 2
        r54 = r54s ** 2
        r56 = r56s ** 2
        exbeta = ex * betax
        eybeta = ey * betay
        exobeta = ex / betax
        eyobeta = ey / betay
        exponent = (exbeta * r51r52) + (exobeta * r52) + (eybeta * r53r54) + (eyobeta * r54) + ((sigd ** 2) * r56)
        return np.exp(kfac * exponent)

    def ldtaus(self, lamb, slice_params, optics_map, i1):
        '''
        Eq. 50: Landau damping between two lattice elements

        :param lamb: initial modulation wavelength
        :param sigmamatrix: SDDSobject containing beam sigma matrices
        :param beammatrix: SDDSobject containing transport matrices
        :param index1: index from which to calculate
        :param index2: index to which to calculate

        :return: Landau damping factor LD(tau,s)
        '''
        kfac = -(k_wn(lamb) ** 2) / 2
        ex = slice_params[0]['ex']
        ey = slice_params[0]['ey']
        betax = slice_params[0]['beta_x']
        betay = slice_params[0]['beta_y']
        alphax = slice_params[0]['alpha_x']
        alphay = slice_params[0]['alpha_y']
        sigd = slice_params[i1]['sdelta']
        r51s = optics_map[-1][4, 0]
        r52s = optics_map[-1][4, 1]
        r53s = optics_map[-1][4, 2]
        r54s = optics_map[-1][4, 3]
        r56s = optics_map[-1][4, 5]
        r51tau = optics_map[i1][4, 0]
        r52tau = optics_map[i1][4, 1]
        r53tau = optics_map[i1][4, 2]
        r54tau = optics_map[i1][4, 3]
        r56tau = optics_map[i1][4, 5]
        r51taus = r51s - r51tau
        r52taus = r52s - r52tau
        r53taus = r53s - r53tau
        r54taus = r54s - r54tau
        r56taus = r56s - r56tau
        r51r52 = (r51taus - ((alphax / betax) * r52taus)) ** 2
        r52 = r52taus ** 2
        r53r54 = (r53taus - ((alphay / betay) * r54taus)) ** 2
        r54 = r54taus ** 2
        r56 = r56taus ** 2
        exbeta = ex * betax
        eybeta = ey * betay
        exobeta = ex / betax
        eyobeta = ey / betay
        exponent = (exbeta * r51r52) + (exobeta * r52) + (eybeta * r53r54) + (eyobeta * r54) + ((sigd ** 2) * r56)
        result = np.exp(kfac * exponent)
        return result

    def r56taus(self, optics_map, i1):
        '''
        Eq. 49: R56 transport parameter between beamline elements

        :param beammatrix: SDDSobject containing transport matrices
        :param index1: index from which to calculate
        :param index2: index to which to calculate

        :return: R56(tau->s)
        '''
        r51s = optics_map[i1][4, 0]
        r52s = optics_map[i1][4, 1]
        r53s = optics_map[i1][4, 2]
        r54s = optics_map[i1][4, 3]
        r55s = optics_map[i1][4, 4]
        r56s = optics_map[i1][4, 5]
        r51tau = optics_map[-1][4, 0]
        r52tau = optics_map[-1][4, 1]
        r53tau = optics_map[-1][4, 2]
        r54tau = optics_map[-1][4, 3]
        r55tau = optics_map[-1][4, 4]
        r56tau = optics_map[-1][4, 5]
        return (r56s * r55tau) - (r56tau * r55s) + (r51tau * r52s) - (r51s * r52tau) + (r53tau * r54s) - (r53s * r54tau)
        # return [r56s, r56tau, r51tau, r52s, r51s, r52tau, r53tau, r54s, r53s, r54tau]

    def kernel_K0(self, lamb, slice_params, optics_map, i1):
        '''
        Eq. A17b: Kernel K1

        :param lamb: initial modulation wavelength
        :param sigmamatrix: SDDSobject containing beam sigma matrices
        :param beammatrix: SDDSobject containing transport matrices
        :param cenmatrix: SDDSobject containing beam centroids
        :param index1: index from which to calculate
        :param index2: index to which to calculate
        :param linac: can't remember what this was used for...
        :param lscon: include LSC impedance

        :return: K1
        '''
        currentfac = slice_params[i1]['I'] / ((slice_params[i1]['gamma']) * I_Alfven)
        compfac = (slice_params[i1]['I'] / slice_params[0]['I'])
        lamb_compressed = lamb / compfac
        impedancefac = self.lscimpedance(lamb_compressed, slice_params, i1) if self.lsc else 0
        if self.csr and (elem.__class__ in [RBend, SBend, Bend]):
            impedancefac += self.csrimpedance(lamb_compressed, elem)
        ldfac = self.ldtaus(lamb, slice_params, optics_map, i1)
        return currentfac * impedancefac * ldfac

    def kernel_K1(self, lamb, slice_params, optics_map, i1, elem):
        '''
        Eq. A17b: Kernel K1

        :param lamb: initial modulation wavelength
        :param sigmamatrix: SDDSobject containing beam sigma matrices
        :param beammatrix: SDDSobject containing transport matrices
        :param cenmatrix: SDDSobject containing beam centroids
        :param index1: index from which to calculate
        :param index2: index to which to calculate
        :param linac: can't remember what this was used for...
        :param lscon: include LSC impedance

        :return: K1
        '''
        currentfac = slice_params[i1]['I'] / ((slice_params[i1]['gamma']) * I_Alfven)
        compfac = (slice_params[i1]['I'] / slice_params[0]['I'])
        lamb_compressed = lamb / compfac
        kfac = k_wn(lamb_compressed)
        impedancefac = self.lscimpedance(lamb_compressed, slice_params, i1) if self.lsc else 0
        if self.csr and (elem.__class__ in [RBend, SBend, Bend]):
            impedancefac += self.csrimpedance(lamb_compressed, elem)
        ldfac = self.ldtaus(lamb, slice_params, optics_map, i1)
        r56fac = self.r56taus(optics_map, i1)
        return currentfac * kfac * r56fac * impedancefac * ldfac


    def kernel_K2(self, lamb, slice_params, optics_map, i1):
        '''
        Eq. A17c: Kernel K2

        :param lamb: initial modulation wavelength
        :param sigmamatrix: SDDSobject containing beam sigma matrices
        :param beammatrix: SDDSobject containing transport matrices
        :param cenmatrix: SDDSobject containing beam centroids
        :param index1: index from which to calculate
        :param index2: index to which to calculate
        :param linac: can't remember what this was used for...
        :param lscon: include LSC impedance

        :return: K1
        '''
        currentfac = slice_params[i1]['I'] / ((slice_params[i1]['gamma']) * I_Alfven)
        compfac = (slice_params[i1]['I'] / slice_params[0]['I'])
        lamb_compressed = lamb / compfac
        kfac = k_wn(lamb_compressed) ** 2
        impedancefac = self.lscimpedance(lamb_compressed, slice_params, i1) if self.lsc else 0
        if self.csr and (elem.__class__ in [RBend, SBend, Bend]):
            impedancefac += self.csrimpedance(lamb_compressed, elem)
        ldfac = self.ldtaus(lamb, slice_params, optics_map, i1)
        r56fac = self.r56taus(optics_map, i1) ** 2
        return currentfac * kfac * r56fac * impedancefac * ldfac

    def lscimpedance(self, lamb, slice_params, i1):
        '''
        Eq. 52: LSC impedance (although I use a function from PRAB. 23, 014403 (Eq. 26)

        :param lamb: initial modulation wavelength
        :param sigmamatrix: SDDSobject containing beam sigma matrices
        :param cenmatrix: SDDSobject containing beam centroids
        :param index: index in lattice

        :return: LSC impedance
        '''
        kz = k_wn(lamb)
        rb = 0.8375 * (slice_params[i1]['sig_x'] + slice_params[i1]['sig_y'])
        gamma = slice_params[i1]['gamma']
        xib = kz * rb / gamma
        besselfac = scipy.special.kv(1, xib)
        # initfac = (1j * constant.Z0) / (np.pi * gamma * rb)
        initfac = (4j) / (gamma * rb)
        lscfac = (1 - (xib * besselfac)) / xib
        # return 1j * (Z0 / (np.pi * kz * (rb ** 2))) * (1 - (xib * scipy.special.kv(1, xib)))
        return initfac * lscfac
        # return 1j * (Z0 / (np.pi * gamma * rb)) * (1 - (xib * scipy.special.kv(1, xib) * scipy.special.iv(0, xib))) / xib

    def csrimpedance(self, lamb, elem):
        '''
        Eq. 51: CSR impedance

        :param lamb: initial modulation wavelength
        :param sigmamatrix: SDDSobject containing beam sigma matrices
        :param index: index in lattice

        :return: CSR impedance
        '''
        if elem.angle < 1e-10:
            return 0
        else:
            kz = k_wn(lamb)
            bendradius = elem.l / elem.angle
            return -1j * A_csr * (kz ** (1 / 3)) / (bendradius ** (2 / 3))