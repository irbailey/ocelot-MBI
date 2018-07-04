"""Test of the demo file demos/ebeam/dba_tracking.py"""

import os
import sys
from copy import copy
import time

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_RES_DIR = FILE_DIR + '/ref_results/'

from demos_tests.params import *
from dba_tracking_conf import *


def test_lattice_transfer_map(lattice, parametr=None, update_ref_values=False):
    """R maxtrix calculation test"""

    r_matrix = lattice_transfer_map(lattice, 0.0)
    
    if update_ref_values:
        return numpy2json(r_matrix)
    
    r_matrix_ref = json2numpy(json_read(REF_RES_DIR + sys._getframe().f_code.co_name + '.json'))
    
    result = check_matrix(r_matrix, r_matrix_ref, TOL, assert_info=' r_matrix - ')
    assert check_result(result)


def test_twiss(lattice, parametr=None, update_ref_values=False):
    """Twiss parameters calculation function test"""

    tw0 = Twiss()
    tw0.x = 0.1
    tw0.y = 0.2
    
    tws = twiss(lattice, tw0, nPoints=1000)

    tws = obj2dict(tws)
    
    if update_ref_values:
        return tws
    
    tws_ref = json_read(REF_RES_DIR + sys._getframe().f_code.co_name + '.json')

    result = check_dict(tws, tws_ref, TOL, 'absotute', assert_info=' tws - ')
    assert check_result(result)


@pytest.mark.parametrize('parametr', [0, 1, 2])
def test_tracking_step(lattice, parametr, update_ref_values=False):
    """Tracking step function test
    :parametr=0 - tracking with transverse shift
    :parametr=1 - tracking with negative energy shift
    :parametr=2 - tracking with positive energy shift
    """

    p = []
    p.append(Particle(x=0.1, y=0.2))
    p.append(Particle(p=-0.001))
    p.append(Particle(p=0.001))

    navi = Navigator(lattice)
    dz = 0.01

    P = []
    for iii in range(int(lattice.totalLen/dz)):
        tracking_step(lattice, p[parametr], dz=dz, navi=navi)
        P.append(copy(p[parametr]))
        
    P = obj2dict(P)

    if update_ref_values:
        return P

    P_ref = json_read(REF_RES_DIR + sys._getframe().f_code.co_name + str(parametr) +'.json')

    result = check_dict(P, P_ref, TOL, 'absotute', assert_info=' P - ')
    assert check_result(result)
    

def setup_module(module):

    f = open(pytest.TEST_RESULTS_FILE, 'a')
    f.write('### DBA_TRACKING START ###\n\n')
    f.close()


def teardown_module(module):

    f = open(pytest.TEST_RESULTS_FILE, 'a')
    f.write('### DBA_TRACKING END ###\n\n\n')
    f.close()


def setup_function(function):
    
    f = open(pytest.TEST_RESULTS_FILE, 'a')
    f.write(function.__name__)
    f.close()

    pytest.t_start = time.time()


def teardown_function(function):
    f = open(pytest.TEST_RESULTS_FILE, 'a')
    f.write(' execution time is ' + '{:.3f}'.format(time.time() - pytest.t_start) + ' sec\n\n')
    f.close()
    

@pytest.mark.update
def test_update_ref_values(lattice, cmdopt):
    
    update_functions = []
    update_functions.append('test_lattice_transfer_map')
    update_functions.append('test_twiss')
    update_functions.append('test_tracking_step')
    
    update_function_parameters = {}
    update_function_parameters['test_tracking_step'] = [0, 1, 2]
    
    parametr = update_function_parameters[cmdopt] if cmdopt in update_function_parameters.keys() else ['']

    if cmdopt in update_functions:
        for p in parametr:
            result = eval(cmdopt)(lattice, p, True)
        
            if os.path.isfile(REF_RES_DIR + cmdopt + str(p) + '.json'):
                os.rename(REF_RES_DIR + cmdopt + str(p) + '.json', REF_RES_DIR + cmdopt + str(p) + '.old')
            
            json_save(result, REF_RES_DIR + cmdopt + str(p) + '.json')