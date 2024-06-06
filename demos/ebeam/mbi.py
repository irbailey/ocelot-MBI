import sys, os
for i,j in enumerate(sys.path):
    if 'ocelot' in j:
        sys.path.pop(i)
sys.path.append('../../')
import ocelot
import yaml
import importlib
import numpy as np
import os
import h5py
from copy import deepcopy
import time
import scipy.special
import subprocess
import random
from inspect import getsourcefile
import ocelot.gui.accelerator
import matplotlib.pyplot as plt
import ocelot.cpbd.mbi
import copy

if os.name == 'nt':
    strsplit = '\\'
else:
    strsplit = '/'

def run_ocelot():
    if '~' in settings['lattice_dir']:
        fulllatpath = os.path.expanduser(settings['lattice_dir']) + '/' + settings['ebeam']['lattice'].replace('.py', '')
    else:
        fulllatpath = settings['ebeam']['lattice_path'].replace('.py', '')
    latname = settings['ebeam']['lattice']
    latfile = import_lattice_from_path(fulllatpath, latname)
    lattice = ocelot.cpbd.magnetic_lattice.MagneticLattice(latfile.cell)
    p_array_input = deepcopy(ocelot.cpbd.io.load_particle_array(settings['input']))
    navi = ocelot.cpbd.optics.Navigator(lattice)
    charge = settings['ebeam']['charge']
    datalen = p_array_input.size()
    p_array_input.q_array = np.ones(datalen) * charge / datalen
    p_array_input_1 = deepcopy(p_array_input)
    navi = ocelot.cpbd.optics.Navigator(lattice)
    lattice, navi, mbi = set_collective_effects(lattice, navi)
    p_array_in = deepcopy(p_array_input)
    tws_track, p_array_output = ocelot.cpbd.track.track(lattice, p_array_input, navi)
    results = {'track': [tws_track, p_array_output, lattice]}
    # return results
    outfiles = save_ocelot_files(lattice, p_array_in, settings['input_name'], tws_track, p_array_output)
    s, curr = ocelot.cpbd.beam.get_current(p_array_output)
    maxloc = max(range(len(curr)), key=curr.__getitem__)
    currend = float(np.mean(curr[int(maxloc) - 20: int(maxloc) + 20]))
    settings['ebeam'].update({'current_end': float(round(currend, 3))})
    settings['ebeam'].update({'energy_end': float(tws_track[-1].E)})
    tw0=ocelot.cpbd.beam.Twiss()
    tw0.beta_x = settings['ebeam']['betax_init']
    tw0.beta_y = settings['ebeam']['betay_init']
    tw0.alpha_x = settings['ebeam']['alphax_init']
    tw0.alpha_y = settings['ebeam']['alphay_init']
    tw0.E = settings['ebeam']['energy_init']
    tw=ocelot.cpbd.optics.twiss(lattice, tws0=tw0)
    with open(os.path.abspath(getsourcefile(lambda:0)).replace('mbi.py', 'settings.yaml'), "w") as outfile:
        yaml.dump(settings, outfile, default_flow_style=False)
    np.savetxt('mbi.dat', mbi.bf)

def import_lattice_from_path(path, file):
    '''
    Use a string to import an OCELOT lattice

    :param path: /path/to/lattice
    :param file: name of lattice file
    :return: lattice loaded as Python module
    '''
    spec = importlib.util.spec_from_file_location(file, path+'.py')
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return foo

def save_ocelot_files(lattice, inputfile, infilename, tws, p_array, scan=False):
    '''
    Write OCELOT lattice file, H5 file with Twiss values and final beam
    distribution to a local directory (based on initial timestamp)
    '''
    latticeio = ocelot.cpbd.latticeIO.LatticeIO()
    if scan:
        endelem = lattice[0].sequence[-1].id
    else:
        endelem = lattice.sequence[-1].id
    timestr = settings['time_stamp']
    line = settings['line']
    outdirshort = line + '_' + timestr
    if '~' in settings['out_dir']:
        outdir = os.path.expanduser(settings['out_dir'])
    else:
        outdir = settings['out_dir']
    if not os.path.isdir(outdir):
        try:
            os.makedirs(outdir)
            os.chdir(outdir)
        except:
            print("ERROR!!!! Could not create directory " + outdir)
    else:
        os.chdir(outdir)
    inputbeamfile = infilename
    if os.name == 'nt':
        inputpath = outdir + settings['input_name']
    else:
        inputpath = outdir + settings['input_name']
    ocelot.cpbd.io.save_particle_array(inputpath, inputfile)
    beamfilename = endelem + '.npz'
    latticeio.save_lattice(lattice, tws0=tws, file_name='lattice.py')
    beamfilename = endelem + '.npz'
    beamfile = ocelot.cpbd.io.save_particle_array(beamfilename, p_array)
    twissfilename = inputpath.replace(settings['input_name'], 'twiss.h5')
    twissfile = write_to_h5(tws, filename=twissfilename)
    return [outdirshort, inputbeamfile, beamfilename, twissfilename]

def save_outfile(lattice, p_array, settings):
    '''
    Write OCELOT lattice file, H5 file with Twiss values and final beam
    distribution to a local directory (based on initial timestamp)
    '''
    latticeio = ocelot.cpbd.latticeIO.LatticeIO()
    endelem = lattice.sequence[-1].id
    timestr = settings['time_stamp']
    line = settings['line']
    if '~' in settings['out_dir']:
        outdir = os.path.expanduser(settings['out_dir'])
    else:
        outdir = settings['out_dir']
    if not os.path.isdir(outdir):
        try:
            os.makedirs(outdir)
            os.chdir(outdir)
        except:
            print("ERROR!!!! Could not create directory " + outdir)
    else:
        os.chdir(outdir)
    beamfilename = endelem + '.npz'
    beamfile = ocelot.cpbd.io.save_particle_array(beamfilename, p_array)
    return beamfilename

def write_to_h5(d, filename='settings'):
    print(filename)
    file = h5py.File(filename, 'w')
    file.create_group('twiss_values')
    betax = [p.beta_x for p in d]
    betay = [p.beta_y for p in d]
    alphax = [p.alpha_x for p in d]
    alphay = [p.alpha_y for p in d]
    energy = [p.E for p in d]
    s = [p.s for p in d]
    idn = [p.id for p in d]
    idnp = [x.encode('utf-8') for x in idn]
    namelist = ['element', 's', 'beta_x', 'beta_y', 'alpha_x', 'alpha_y', 'energy']
    formats = [('S10'), (float), (float), (float), (float), (float), (float)]
    ds_dt = np.dtype({'names': namelist, 'formats': formats})
    rec_arr = np.rec.fromarrays([idnp, s, betax, betay, alphax, alphay, energy], dtype=ds_dt)
    file['twiss_values'].create_dataset('beta_x', data=betax)
    file['twiss_values'].create_dataset('beta_y', data=betay)
    file['twiss_values'].create_dataset('alpha_x', data=alphax)
    file['twiss_values'].create_dataset('alpha_y', data=alphay)
    file['twiss_values'].create_dataset('energy', data=energy)
    file['twiss_values'].create_dataset('s', data=s)
    file['twiss_values'].create_dataset('element', data=idnp)
    file['twiss_values'].create_dataset('all_twiss', data=rec_arr)
    file.close()

def set_collective_effects(lattice, navi, all=False):
    '''
    Enable collective effects (based on \'settings\' dictionary
    :param lattice: OCELOT lattice
    :param navi: OCELOT Navigator
    :return: Lattice w/ collective effects applied
    :return: Navigator w/ collective effects
    '''
    navi = navi
    lattice = lattice
    setsc = settings['ebeam']['space_charge']
    setlsc = settings['ebeam']['lsc']
    setcsr = settings['ebeam']['csr']
    setwake = settings['ebeam']['wakes']
    setlh = settings['ebeam']['lh']
    mbi1 = ocelot.cpbd.mbi.MBI(lamb_range=np.linspace(0.1e-6,50e-6,30))
    mbi1.step = settings['ebeam']['unit_step']
    mbi1.navi = copy.deepcopy(navi)
    mbi1.lattice = copy.deepcopy(lattice)
    mbi1.lsc = True
    navi.add_physics_proc(mbi1, lattice.sequence[0], lattice.sequence[-1])
    if setsc:
        sc1, sc2, scst, scend = set_space_charge()
        if all:
            navi.add_physics_proc(sc2, lattice.sequence[0], lattice.sequence[-1])
        else:
            navi.add_physics_proc(sc2, lattice.sequence[settings['space_charge']['start']],
                                       lattice.sequence[settings['space_charge']['end']])
    if setcsr:
        if all:
            csr = ocelot.cpbd.csr.CSR()
            csr.n_bin = settings['csr']['n_bin']
            csr.m_bin = settings['csr']['m_bin']
            csr.sigma_min = settings['csr']['sigma_min']
            csr.step = settings['ebeam']['unit_step']
            navi.add_physics_proc(csr, lattice.sequence[0], lattice.sequence[-1])
        else:
            csr, csrst, csrend = set_csr(lattice)
            if (len(csrst) == len(csrend)) and (len(csrst) > 0):
                for st, en in zip(csrst, csrend):
                    navi.add_physics_proc(csr, lattice.sequence[st], lattice.sequence[en])
    if setwake:
        latticeio = ocelot.cpbd.latticeIO.LatticeIO()
        names = np.array([lat.id for lat in lattice.sequence])
        cavall = latticeio._find_objects(lattice, [ocelot.cpbd.elements.Cavity])
        cavs = []
        for cav in cavall:
            if 'ACC' in cav.id:
                cavs.append(cav)
        for cav in cavs:
            if settings['wake'][cav.id + '_wake'] != 'NO':
                zwake, st = set_wake(cav, names)
                navi.add_physics_proc(zwake, lattice.sequence[st-1], lattice.sequence[st+1])
    if setlsc:
        lsc = set_lsc()
        if all:
            navi.add_physics_proc(lsc, lattice.sequence[0], lattice.sequence[-1])
        else:
            navi.add_physics_proc(lsc, lattice.sequence[settings['space_charge']['start']],
                                       lattice.sequence[settings['space_charge']['end']])
    if setlh:
        allids = [e.id for e in lattice.sequence]
        if settings['laser_heater']['laser'] in allids:
            lhlaserind = allids.index(settings['laser_heater']['laser'])
            lhundind = allids.index(settings['laser_heater']['und'])
            lhund = lattice.sequence[lhundind]
            lhlaser = lattice.sequence[lhlaserind]
            lhphysproc = ocelot.cpbd.physics_proc.LaserModulator()
            lhphysproc.Ku = lhund.Ky
            lhund.Kx = 0
            lhphysproc.Lu = lhund.lperiod * lhund.nperiods
            lhphysproc.lperiod = lhund.lperiod
            twis = ocelot.cpbd.beam.Twiss()
            twis.beta_x = settings['ebeam']['betax_init']
            twis.beta_y = settings['ebeam']['betay_init']
            twis.alpha_x = settings['ebeam']['alphax_init']
            twis.alpha_y = settings['ebeam']['alphay_init']
            twis.E = settings['ebeam']['energy_init']
            twisslh = ocelot.cpbd.optics.twiss(lattice, tws0=twis)
            emitx = 1e-6 / (twis.E * 1e3 / 0.511)
            emity = 1e-6 / (twis.E * 1e3 / 0.511)
            xx = np.sqrt(twisslh[lhundind].beta_x * emitx)
            yy = np.sqrt(twisslh[lhundind].beta_y * emity)
            lhx = np.mean([xx, yy]) * 1.2
            lhkfac = (lhund.Ky ** 2) / (4 + (2 * (lhund.Ky ** 2)))
            jjfac = scipy.special.jv(0, lhkfac) - scipy.special.jv(1, lhkfac)
            beamsizefac = np.sqrt(lhx**2 / (2*(xx**2 + lhx**2)))
            powfac = np.sqrt((lhlaser.c0) / (8.7e9))
            lhfac = (lhund.Ky * jjfac * lhund.nperiods * (0.1 * lhund.lperiod)) / (lhx * twis.E * 1e3 / 0.511)
            lhphysproc.dE = beamsizefac * powfac * lhfac
            navi.add_physics_proc(lhphysproc, lattice.sequence[lhundind], lattice.sequence[lhundind + 1])
    navi.unit_step = settings['ebeam']['unit_step']
    return [lattice, navi, mbi1]

def set_space_charge():
    '''
    Enable space charge
    :return: SpaceCharge (OCELOT class)
    :return: SpaceCharge (OCELOT class)
    :return: List of elements where space charge calculation is started
    :return: List of elements where space charge calculation is finished
    '''
    sc1 = ocelot.cpbd.sc.SpaceCharge()
    sc2 = ocelot.cpbd.sc.SpaceCharge()
    sc1.nmesh_xyz = settings['space_charge']['mesh']
    sc2.nmesh_xyz = settings['space_charge']['mesh']
    sc1.step = 1
    sc2.step = settings['space_charge']['step']
    start = settings['space_charge']['start']
    end = settings['space_charge']['end']
    return [sc1, sc2, start, end]
    
def set_lsc():
    '''
    Enable space charge
    :return: LSC (OCELOT class)
    :return: LSC (OCELOT class)
    :return: List of elements where space charge calculation is started
    :return: List of elements where space charge calculation is finished
    '''
    sc1 = ocelot.cpbd.sc.LSC()
    return sc1

def set_csr(lattice):
    '''
    Enable CSR
    :param lattice: OCELOT lattice
    :return: CSR (OCELOT class)
    :return: List of elements where CSR calculation is started
    :return: List of elements where CSR calculation is finished
    '''
    csr = ocelot.cpbd.csr.CSR()
    names = np.array([lat.id for lat in lattice.sequence])
    start_elems = []
    end_elems = []
    csr.n_bin = settings['csr']['n_bin']
    csr.m_bin = settings['csr']['m_bin']
    csr.sigma_min = settings['csr']['sigma_min']
    csr.unit_step = settings['ebeam']['unit_step']
    elems = ['BC1', 'SCL']
    for el in elems:
        st = settings['csr'][el+'_start']
        en = settings['csr'][el+'_end']
        if st in names:
            start_elems.append(np.where(names == st)[0][0])
        if en in names:
            end_elems.append(np.where(names == en)[0][0])
    return [csr, start_elems, end_elems]

def set_wake(cav, names):
    '''
    Enable geometric wakes in a cavity
    :param cav: OCELOT cavity element
    :param names: Names of all elements in lattice
    :return: Wakefield
    :return: Start of where wake is applied
    '''
    wakename = settings['wake'][cav.id + '_wake']
    if '~' in settings['lattice_dir']:
        loc = os.path.expanduser(settings['lattice_dir']) + '/' + wakename
    else:
        loc = settings['lattice_dir'] + wakename
    zwaketab = ocelot.cpbd.wake3D.WakeTable(loc)
    zwake = ocelot.cpbd.wake3D.Wake()
    wakelen = len(zwaketab.TH[:][0][0][4])
    zwake.w_sampling = wakelen
    zwake.wake_table = zwaketab
    zwake.step = settings['ebeam']['unit_step']
    zwake.factor = settings['wake'][cav.id + '_ncell']
    st = np.where(names == cav.id)[0][0]
    return [zwake, st]
    
if __name__ == "__main__":
    with open(sys.argv[1], "r") as stream:
        try:
            settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit()
    run_ocelot()
