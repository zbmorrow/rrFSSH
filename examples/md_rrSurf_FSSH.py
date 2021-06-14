import numpy as np
import numpy.matlib
import os, sys
sys.path.append('../src/')
import Tasmanian as tsg    # must have Tasmanian in PYTHONPATH variable
import time
from rrmd_core import *


if len(sys.argv) > 2:
    Nrun = int(sys.argv[1])
    Tfinal = float(sys.argv[2])
elif len(sys.argv) > 1:
    print('md_rrSurf_mixed.py [Nrun=1] [Tfinal=5000.0]')
    Nrun = int(sys.argv[1])
    Tfinal = 5000.0
else:
    print('md_rrSurf_mixed.py [Nrun=1] [Temp=298.15] [Tfinal=2500.0]')
    Nrun = 1
    Tfinal = 5000.0

print('  Run index: ',Nrun)
print('     Tfinal: ',Tfinal,'fs')

chkDir = 'FSSH_singlet_trans_example'
if not os.path.exists(chkDir):
    os.mkdir(chkDir)

chkDir += '/%d' % Nrun
if not os.path.exists(chkDir):
    os.mkdir(chkDir)

# read atomic masses
state_folder = 'matfiles_iptotal_5_iptotal_8'
M = read_atom(state_folder)
M = np.array([M]).T
N_atom = len(M)
M = np.reshape(np.matlib.repmat(M,1,3),(3*N_atom,1))

ev2ieu = 9.648e-3  # convert 1 eV to internal energy units [IEU = amu * angstrom^2 fs^(-2)]

# set input record
inpRec = dict();
inpRec['restart'] = False
inpRec['method'] = 'NVE'
inpRec['N_per'] = 3
inpRec['M'] = np.reshape(M,(M.size,))   # units = amu
inpRec['Temp'] = 298.15                 # units = K
inpRec['dt'] = 0.25                     # units = fs
inpRec['Tfinal'] = Tfinal               # units = fs
inpRec['maxStep'] = int(inpRec['Tfinal'] / inpRec['dt'] + 1)
# need bounds since domains must be on same scale within the MD simulator
inpRec['bounds'] = [[-180,180], [-180,180], [-180,180], [1.1, 2.5], [90, 270]]  # units = GIC
inpRec['Gamma'] = 0.001                 # units = fs^(-1)
inpRec['currSurf'] = 1

# make sparse grids for PES
sg_poly_energy = tsg.SparseGrid()
sg_poly_energy.makeGlobalGrid(2,1,8,'iptotal','clenshaw-curtis')
sg_poly_energy.setDomainTransform(np.asarray([[0, 1], [0,1]]))
N_poly = len(sg_poly_energy.getPoints())

sg_trig_energy = tsg.SparseGrid()
sg_trig_energy.makeFourierGrid(3,N_poly,5,'iptotal',[1, 2, 2])
sg_trig_energy.setDomainTransform(np.asarray([[0, 1], [0,1], [0, 1]]))
N_trig = len(sg_trig_energy.getPoints())

dataE = ev2ieu * np.loadtxt('%s/energy.dat' % state_folder)
dataE = np.reshape(dataE,(N_trig,N_poly))
sg_trig_energy.loadNeededPoints(dataE)

inpRec['UQ_0'] = lambda Q : eval_mixed(Q, sg_trig=sg_trig_energy, sg_poly=sg_poly_energy, ndof=1)
inpRec['dUdQ_0'] = lambda Q : eval_mixed_grad(Q, sg_trig=sg_trig_energy, sg_poly=sg_poly_energy, ndof=1)

# make sparse grids for PES (first singlet ES)
sg_poly_energy_ES = tsg.SparseGrid()
sg_poly_energy_ES.makeGlobalGrid(2,1,8,'iptotal','clenshaw-curtis')
sg_poly_energy_ES.setDomainTransform(np.asarray([[0,1], [0,1]]))

sg_trig_energy_ES = tsg.SparseGrid()
sg_trig_energy_ES.makeFourierGrid(3,N_poly,5,'iptotal',[1, 2, 2])
sg_trig_energy_ES.setDomainTransform(np.asarray([[0,1], [0,1], [0,1]]))

excitation_energy_ES1 = ev2ieu * np.loadtxt('%s/excitation_S0_S1.dat' % state_folder)
excitation_energy_ES1 = np.reshape(excitation_energy_ES1,(N_trig,N_poly))
sg_trig_energy_ES.loadNeededPoints(dataE + excitation_energy_ES1)

inpRec['UQ_1'] = lambda Q : eval_mixed(Q, sg_trig=sg_trig_energy_ES, sg_poly=sg_poly_energy_ES, ndof=1)
inpRec['dUdQ_1'] = lambda Q : eval_mixed_grad(Q, sg_trig=sg_trig_energy_ES, sg_poly=sg_poly_energy_ES, ndof=1)

# make sparse grids for Cartesian geometry
sg_poly_geom = tsg.SparseGrid()
sg_poly_geom.makeGlobalGrid(2,1,8,'iptotal','clenshaw-curtis')
sg_poly_geom.setDomainTransform(np.asarray([[0, 1], [0,1]]))

sg_trig_geom = tsg.SparseGrid()
sg_trig_geom.makeFourierGrid(3,N_poly*N_atom*3,5,'iptotal',[1, 2, 2])
sg_trig_geom.setDomainTransform(np.asarray([[0, 1], [0,1], [0, 1]]))

dataG = np.loadtxt('%s/geomcart.dat' % state_folder,delimiter=',')
sg_trig_geom.loadNeededPoints(dataG)

inpRec['XQ'] = lambda Q : eval_mixed(Q, sg_trig=sg_trig_geom, sg_poly=sg_poly_geom, ndof=3*N_atom)
inpRec['dXdQ'] = lambda Q : eval_mixed_grad(Q, sg_trig=sg_trig_geom, sg_poly=sg_poly_geom, ndof=3*N_atom)

# get initial position, velocity, and momentum
burn_in_dir = 'rrmd_langevin_trig_singlet_trans_298K'
Q0 = np.loadtxt(os.path.join(burn_in_dir, 'Qlist.txt'), delimiter=' ')[20000+40*Nrun,:]
Q0 = to_canonical(Q0, inpRec['bounds'])
P0 = np.loadtxt(os.path.join(burn_in_dir, 'Plist.txt'), delimiter=' ')[20000+40*Nrun,:]

sobol = np.loadtxt('%s/sobol.dat' % state_folder, delimiter=',')
NACfull = np.loadtxt('%s/NACfull.dat' % state_folder)
estimate_NAC = lambda Q: estimate_NAC_uni(pbcfun(Q, inpRec['N_per']), sobol, NACfull, inpRec['dXdQ'], radius=0.05)

# do the integration and save the output
t = time.time()
chkFreq = 500

GQ = lambda Q : mass_metric(Q, inpRec['dXdQ'], inpRec['M'])
mdHdQfun_NVE = lambda Q,P : mdhdqfun_uni(Q, P, inpRec['dUdQ_%d' % inpRec['currSurf']], GQ)
dHdPfun_NVE = lambda Q,P: dhdpfun_uni(Q, P, GQ)
C0 = np.asarray([0.0, 1.0])
Qlist = [from_canonical(Q0, inpRec['bounds'])]
Plist = [P0]
NAClist = [estimate_NAC(Q0)]
KElist = [ke_uni(Q0,P0,GQ)]
UQlist = [inpRec['UQ_%d' % inpRec['currSurf']](Q0)]
tlist = [0]
surflist = [inpRec['currSurf']]

def saveResults():
    np.savetxt(os.path.join(chkDir, 'Qlist.txt'), np.asarray(Qlist), '%.15f')
    np.savetxt(os.path.join(chkDir, 'Plist.txt'), np.asarray(Plist), '%.15f')
    np.savetxt(os.path.join(chkDir, 'NAClist.txt'), np.asarray(NAClist), '%.15f')
    np.savetxt(os.path.join(chkDir, 'KElist.txt'), np.asarray(KElist), '%.15f')
    np.savetxt(os.path.join(chkDir, 'UQlist.txt'), np.asarray(UQlist), '%.15f')
    np.savetxt(os.path.join(chkDir, 'tlist.txt'), np.asarray(tlist), '%.5f')
    np.savetxt(os.path.join(chkDir, 'surflist.txt'), np.asarray(surflist), '%d')

for i in range(1,inpRec['maxStep']):
    (Qt, Pt, _) = propagator_stormer_verlet_uni(Q0,P0,inpRec,mdHdQfun_NVE,dHdPfun_NVE)

    Qlist.append(from_canonical(Qt, inpRec['bounds']))
    Plist.append(Pt)
    NAClist.append(estimate_NAC(Qt))
    KElist.append(ke_uni(Qt,Pt,GQ))
    UQlist.append(inpRec['UQ_%d' % inpRec['currSurf']](Qt))
    tlist.append(tlist[-1]+inpRec['dt'])
    surflist.append(inpRec['currSurf'])

    Qlist_internal = [Q0, Qt]

    # for TDSH, use smaller dt than classical dynamics time step
    UQinternal = np.asarray([[inpRec['UQ_0'](Q0), inpRec['UQ_1'](Q0)],
                             [inpRec['UQ_0'](Qt), inpRec['UQ_1'](Qt)]])
    Ct = crank_nicolson(C0, Qlist_internal, UQinternal, NAClist[-2:], [inpRec['dt'], 0.01])

    (Pt, switched) = switching_test(Ct, Qt, Pt, inpRec, NAClist[-1], dHdPfun_NVE)

    # if switched, update UQ surface
    if switched:
        print('Switching states at i=%d!' % i)
        inpRec['currSurf'] = 1 if inpRec['currSurf'] == 0 else 0
        mdHdQfun_NVE = lambda Q,P : mdhdqfun_uni(Q, P, inpRec['dUdQ_%d' % inpRec['currSurf']], GQ)

    # checkpoint the output
    if i % chkFreq == 0 or switched:
        saveResults()
        # if switched:
        #    break
    if Qt[3] > 1.0:
        print('Methyl group dissociated. Saving results...')
        saveResults()
        break

    Q0 = Qt
    P0 = Pt
    C0 = Ct

saveResults()
np.savetxt(os.path.join(chkDir, 'runtime.txt'), [time.time()-t], '%.4f')
