import numpy as np
from aux import *
import sys
import argparse
from mpi4py import MPI
from tqdm import tqdm

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


trunc_points = ["K1", "K2"]
material = MoS2

p = argparse.ArgumentParser(description="Generate two-band model inputs.")
p.add_argument("--res", type=int, help="k-point grid density", default=15)
p.add_argument("--cres", type=int, help="coarse k-point grid density", default=15)
p.add_argument("--tb", type=int, help="two band model? 0 is No, 1 is Yes", default=1)
p.add_argument("--Rk", type=float, help="truncation radius in 2pi/a", default=0.2)
p.add_argument("--tmax", type=float, help="total simulation time", default=100)
p.add_argument(
    "--trunc_points",
    type=str,
    nargs="+",
    help="high-symmetry points to truncate around",
    default=trunc_points,
)
p.add_argument(
    "--Rx", type=float, help="exciton truncation radius in 2pi/a", default=0.1
)
p.add_argument(
    "--exciton_trunc_points",
    type=str,
    nargs="+",
    help="high-symmetry points to truncate exciton around",
    default=["G"],
)
p.add_argument(
    "--nexbnd",
    type=int,
    help="number of exciton bands to include in calculation",
    default=2,
)
p.add_argument("--init", type=str, help="initial state: linabs, depol", default="depol")
p.add_argument("--temp", type=float, help="temperature in K", default=300)
p.add_argument("--num_trajs", type=int, help="number of trajectories", default=100)
p.add_argument(
    "--batch_size", type=int, help="number of trajectories per batch", default=1
)
p.add_argument("--ind", type=int, help="index for multiple runs", default=-1)
p.add_argument("--phoccs", type=int, help="calculate the phonon occupancies, 0 = No, 1 = Yes", default=0)
p.add_argument("--Ropt", type=float, help="truncation radius of the optical phonon, 0 = no truncation", default=0)
p.add_argument("--Rac", type=float, help="truncation radius of the acoustic phonon, 0 = no truncation", default=0)
args = p.parse_args()

res = args.res
Rk = args.Rk
trunc_points = args.trunc_points
Rx = args.Rx
exciton_trunc_points = args.exciton_trunc_points
nexbnd = args.nexbnd
init = args.init
temp_K = args.temp
temp_therm = temp_K / 293
num_trajs = args.num_trajs
batch_size = args.batch_size
run_ind = args.ind
tmax = args.tmax
phoccs = args.phoccs
Ropt = args.Ropt
Rac = args.Ropt

cres = args.cres
if args.tb == 0:
    tb = False
else:
    tb = True

if tb:
    ex_band_ind = [0,1]
    nexbnd = len(ex_band_ind)
    hamilt_dir = './inits/'+str(res)+'_'+str(Rk)+'_'+"_".join(trunc_points) + str("_tb")
else:
    # Ab initio code not implemented here.
    exit()

eph_dir = hamilt_dir + "/eph/"
overlaps_dir = hamilt_dir + "/overlaps/"
e_bands_dir = hamilt_dir + "/e_bands/"
ph_bands_dir = hamilt_dir + "/ph_bands/"
kgrid_dir = hamilt_dir + "/kgrid/"
exciton_dir = hamilt_dir + "/excitons/"
exph_dir = hamilt_dir + "/exph/"


name_noh5 = (
    str(res)
    + "_"
    + str(Rk)
    + "_"
    + str(Rx)
    + "_"
    + "_".join(trunc_points)
    + str("_tb")
    + "_"
    + str(init)
    + "_"
    + str(temp_K)
)
name = name_noh5 + ".h5"
if run_ind == -1:
    from qclab import Data
    from glob import glob

    data_files = glob(name_noh5 + "_*.h5")
    print("Found existing files:", data_files)
    data = Data()
    for file in tqdm(data_files):
        data_tmp = Data().load(file)
        data.add_data(data_tmp)
    data.save(name)
    print("Saved to ", name)
    exit()
else:
    name = name_noh5 + "_" + str(run_ind) + ".h5"
    if os.path.exists(name):
        print("Found existing data file")
        exit()

kpt_trunc_cart = np.load(kgrid_dir + "kpt_trunc_cart.npy")
kpt_trunc_cryst = np.load(kgrid_dir + "kpt_trunc_cryst.npy")
kpt_trunc_fbz = np.load(kgrid_dir + "kpt_trunc_fbz.npy")
kpt_trunc_ind = np.load(kgrid_dir + "kpt_trunc_ind.npy")
qpt_trunc_cart = np.load(kgrid_dir + "qpt_trunc_cart.npy")
qpt_trunc_cryst = np.load(kgrid_dir + "qpt_trunc_cryst.npy")
qpt_trunc_fbz = np.load(kgrid_dir + "qpt_trunc_fbz.npy")
qpt_trunc_ind = np.load(kgrid_dir + "qpt_trunc_ind.npy")
# position of each q point in the untruncated k grid.
qpt_trunc_pos = np.load(kgrid_dir + "qpt_trunc_pos.npy")
umk_point_mat = np.load(kgrid_dir + "umk_point_mat.npy")

qpt_trunc_opt_mask = np.ones_like(qpt_trunc_pos)
if Ropt != 0:
    qpt_trunc_opt_mask[np.linalg.norm(qpt_trunc_fbz,axis=1) >= Ropt] = 0

qpt_trunc_ac_mask = np.ones_like(qpt_trunc_pos)
if Rac != 0:
    qpt_trunc_ac_mask[np.linalg.norm(qpt_trunc_fbz,axis=1) >= Rac] = 0

nkpts = len(kpt_trunc_cart)
nqpts = len(qpt_trunc_cart)

umk_ind_mat = np.load(kgrid_dir + "umk_ind_mat.npy")

n2 = kpt_trunc_ind[:, 1].astype(int)
n1 = kpt_trunc_ind[:, 0].astype(int)


exciton_kpt_cart = np.copy(qpt_trunc_cart)
exciton_kpt_crystal = np.copy(qpt_trunc_cryst)
exciton_kpt_fbz = np.copy(qpt_trunc_fbz)
exciton_kpt_ind = np.copy(qpt_trunc_ind)
exciton_kpt_pos = np.copy(qpt_trunc_pos)

exciton_trunc_points_cart = np.array(
    [high_symmetry_points[key] for key in exciton_trunc_points]
)
if Rx == 0:
    exciton_trunc_rad = 10000
else:
    exciton_trunc_rad = Rx  # units of 2\pi/a

exciton_kpt_trunc_pos = (
    np.round(
        np.min(
            np.linalg.norm(
                k_Umklapp2(
                    (
                        exciton_trunc_points_cart[:, np.newaxis, :]
                        - exciton_kpt_cart[np.newaxis, :, :]
                    ).reshape(
                        (len(exciton_trunc_points_cart) * len(exciton_kpt_cart), 2)
                    )
                ),
                axis=1,
            ).reshape((len(exciton_trunc_points_cart), len(exciton_kpt_cart))),
            axis=0,
        ),
        3,
    )
    < exciton_trunc_rad
)

exciton_kpt_trunc_cart = exciton_kpt_cart[exciton_kpt_trunc_pos]
exciton_kpt_trunc_crystal = exciton_kpt_crystal[exciton_kpt_trunc_pos]
exciton_kpt_trunc_fbz = exciton_kpt_fbz[exciton_kpt_trunc_pos]
exciton_kpt_trunc_pos = np.arange(len(qpt_trunc_ind), dtype=int)[
    exciton_kpt_trunc_pos
]  # index of the truncated exciton grid in the qpt grid
exciton_kpt_trunc_ind = exciton_kpt_ind[exciton_kpt_trunc_pos]
exciton_kpt_trunc_pos = exciton_kpt_pos[exciton_kpt_trunc_pos]


kappa_n2 = exciton_kpt_trunc_ind[:, 1].astype(int)
kappa_n1 = exciton_kpt_trunc_ind[:, 0].astype(int)

ind_fbz = np.roll(
    np.arange(-(res - 1) / 2, 1 + (res - 1) / 2, 1).astype(int),
    int(1 + ((res - 1) / 2)),
)

kappa_n1_fbz = ind_fbz[n1]
kappa_n2_fbz = ind_fbz[n2]
kappa_umk_n1_fbz = np.zeros(
    (len(exciton_kpt_trunc_ind), len(exciton_kpt_trunc_ind)), dtype=int
)
kappa_umk_n2_fbz = np.zeros(
    (len(exciton_kpt_trunc_ind), len(exciton_kpt_trunc_ind)), dtype=int
)
for i in range(len(exciton_kpt_trunc_ind)):
    kappa_umk_n1_fbz[i] = ind_fbz[kappa_n1[i] - kappa_n1]
    kappa_umk_n2_fbz[i] = ind_fbz[kappa_n2[i] - kappa_n2]
# now map the fbz back to the MP
kappa_umk_n1_mp = np.arange(len(ind_fbz), dtype=int)[kappa_umk_n1_fbz]
kappa_umk_n2_mp = np.arange(len(ind_fbz), dtype=int)[kappa_umk_n2_fbz]
fbz_to_exciton_kpt_trunc = np.zeros((res**2), dtype=int) - 1
fbz_to_exciton_kpt_trunc[exciton_kpt_trunc_pos] = np.arange(
    len(exciton_kpt_trunc_pos), dtype=int
)
fbz_to_qpt_trunc = np.zeros((res**2), dtype=int) - 1
fbz_to_qpt_trunc[qpt_trunc_pos] = np.arange(len(qpt_trunc_pos), dtype=int)
kappa_umk_ind_mat = kappa_umk_n1_mp * res + kappa_umk_n2_mp
# index in the exciton_kptc_trunc grid
# kappa_umk_ind_mat = fbz_to_exciton_kpt_trunc[kappa_umk_ind_mat]
kappa_umk_ind_mat = fbz_to_qpt_trunc[kappa_umk_ind_mat]
umk_ind_mat_qpt = fbz_to_qpt_trunc[umk_ind_mat]  # index in the qpt_trunc grid
assert np.all(umk_ind_mat != -1)
assert np.all(umk_ind_mat_qpt != -1)
exciton_kpt_trunc_pos = fbz_to_qpt_trunc[exciton_kpt_trunc_pos]
assert np.all(exciton_kpt_trunc_pos != -1)
num_quantum_states = len(exciton_kpt_trunc_pos) * nexbnd
if rank == 0:
    print("Number of exciton states: ", num_quantum_states)


w_mev = np.load(ph_bands_dir + "ph_sym_mev.npy", mmap_mode="r")
nmds = w_mev.shape[1]

phonon_pos = np.sort(
    np.unique(kappa_umk_ind_mat.flatten())
)  # index in qpt_trunc of all phonons needed
phonon_pos = phonon_pos[phonon_pos != -1]
# print(phonon_pos)
num_classical_coordinates = len(phonon_pos) * nmds
if rank == 0:
    print("Number of phonons coordinates: ", num_classical_coordinates)

h_q_mev = np.zeros((num_quantum_states, num_quantum_states), dtype=complex)
P_x_init = np.zeros((num_quantum_states), dtype=complex)

ind = 0
for q_pos in exciton_kpt_trunc_pos:
    _, _, evals_ev = load_exciton_data(q_pos, exciton_dir)
    ran = np.arange(ind * nexbnd, (ind + 1) * nexbnd)
    # h_q_mev[ran, ran] = evals_ev[:nexbnd] * 1000  # convert to meV
    h_q_mev[ran, ran] = evals_ev[ex_band_ind] * 1000  # convert to meV
    if np.linalg.norm(exciton_kpt_trunc_fbz[ind]) < 1e-9:
        P_x_ex = np.load(exciton_dir + "P_x_ex.npy")
        P_x_init[ran] = P_x_ex[ex_band_ind]
    ind += 1
harmonic_frequency_mev = np.zeros((num_classical_coordinates), dtype=float)
# dh_qc_dzc_mev = np.zeros((num_classical_coordinates, num_quantum_states, num_quantum_states), dtype=complex)
# dh_qc_dz_mev = np.zeros((num_classical_coordinates, num_quantum_states, num_quantum_states), dtype=complex)
z = np.ones((num_classical_coordinates))
h_qc_mat = np.zeros((num_quantum_states, num_quantum_states), dtype=complex)
p_q_ind_mat = np.zeros((num_quantum_states, num_quantum_states), dtype=int)
m_q_ind_mat = np.zeros((num_quantum_states, num_quantum_states), dtype=int)
g_q_mat = np.zeros((nmds, num_quantum_states, num_quantum_states), dtype=complex)
# print(np.unique(kappa_umk_ind_mat.flatten()))
# print(phonon_pos)
# exit()
dh_qc_dzc_coord_ind = np.array([],dtype=int)
dh_qc_dzc_left_ind = np.array([],dtype=int)
dh_qc_dzc_right_ind = np.array([],dtype=int)
dh_qc_dzc_mels_mev = np.array([])
if rank == 0:
    print('Constructing gradients')
for kq_ind, kq_pos in tqdm(enumerate(exciton_kpt_trunc_pos)):
    #print(kq_pos, kq_ind)
     for k_ind, k_pos in enumerate(exciton_kpt_trunc_pos):
        q_pos = kappa_umk_ind_mat[kq_ind, k_ind]
        if q_pos == -1:
            continue
        q_ind = np.where(phonon_pos == q_pos)[0][0]
        m_q_pos = kappa_umk_ind_mat[k_ind, kq_ind]
        m_q_ind = np.where(phonon_pos == m_q_pos)[0][0]
        g_q = load_exph(kq_pos, k_pos, exph_dir)
        g_mq = load_exph(k_pos, kq_pos, exph_dir)
        # print(np.shape(qpt_trunc_opt_mask), np.shape(q_pos))
        if qpt_trunc_opt_mask[q_pos] == 0: # include the optical phonon if mask == 1 
            # zero out the optical contribution
            g_q[1] *= 0
            g_mq[1] *= 0
        if qpt_trunc_ac_mask[q_pos] == 0: # include the acoustic phonon if mask == 1
            # zero out the acoustic contribution
            g_q[0] *= 0
            g_mq[0] *= 0
        for band_i in range(nexbnd):
            for band_j in range(nexbnd):
                p_q_ind_mat[kq_ind*nexbnd + band_i, k_ind*nexbnd + band_j] = q_ind#*nmds# + mode_ind
                m_q_ind_mat[kq_ind*nexbnd + band_i, k_ind*nexbnd + band_j] = m_q_ind#*nmds + mode_ind
                for mode_ind in range(nmds):
                    ex_band_i = ex_band_ind[band_i]
                    ex_band_j = ex_band_ind[band_j]
                    g_q_mat[mode_ind, kq_ind*nexbnd + band_i, k_ind*nexbnd + band_j] = g_q[mode_ind, ex_band_i, ex_band_j]
                    # h_qc_mat[kq_ind*nexbnd + band_i, k_ind*nexbnd + band_j] = g_q[mode_ind, ex_band_i, ex_band_j]*(z[q_ind*nmds + mode_ind] + np.conj(z)[m_q_ind*nmds + mode_ind])
                    dh_qc_dzc_mels_mev = np.append(dh_qc_dzc_mels_mev, g_mq[mode_ind, ex_band_i, ex_band_j])
                    dh_qc_dzc_coord_ind = np.append(dh_qc_dzc_coord_ind, q_ind*nmds + mode_ind)
                    dh_qc_dzc_left_ind = np.append(dh_qc_dzc_left_ind, k_ind*nexbnd + band_i)
                    dh_qc_dzc_right_ind = np.append(dh_qc_dzc_right_ind, kq_ind*nexbnd + band_j)
                    # dh_qc_dzc_mev[q_ind*nmds + mode_ind, k_ind*nexbnd + band_i, kq_ind*nexbnd + band_j] = g_mq[mode_ind, ex_band_i, ex_band_j]
                    harmonic_frequency_mev[q_ind*nmds + mode_ind] = w_mev[q_pos, mode_ind] 
if rank == 0:
    print("finished.")
evals_idx = np.argsort(np.diag(h_q_mev))
g_q_mat = g_q_mat[:, evals_idx, :][:, :, evals_idx] * meV_to_293K
p_q_ind_mat = p_q_ind_mat[evals_idx][:, evals_idx]
m_q_ind_mat = m_q_ind_mat[evals_idx][:, evals_idx]
h_q_mev = h_q_mev[evals_idx][:, evals_idx]
ext_evals_idx = np.argsort(evals_idx)
dh_qc_dzc_left_ind = ext_evals_idx[dh_qc_dzc_left_ind]
dh_qc_dzc_right_ind = ext_evals_idx[dh_qc_dzc_right_ind]
energy_offset_mev = 100  # meV offset
h_q_mev = (
    h_q_mev
    - np.identity(num_quantum_states) * np.min(np.diag(h_q_mev))
    + np.identity(num_quantum_states) * (energy_offset_mev)
)
harmonic_frequency_therm = harmonic_frequency_mev * meV_to_293K
harmonic_frequency_therm[harmonic_frequency_therm < 1e-10] = 1e-10  # avoid div by zero
zero_pos = harmonic_frequency_therm < 1e-10
h_q_therm = h_q_mev * meV_to_293K
dh_qc_dzc_mels_therm = dh_qc_dzc_mels_mev * meV_to_293K
inds_1 = (dh_qc_dzc_coord_ind, dh_qc_dzc_left_ind, dh_qc_dzc_right_ind)
mels_1 = dh_qc_dzc_mels_therm

def dh_qc_dzc(model, parameters, **kwargs):
    z = kwargs["z"]
    batch_size = len(z)
    num_classical_coordinates = model.constants.num_classical_coordinates
    num_quantum_states = model.constants.num_quantum_states
    shape = (
        batch_size,
        num_classical_coordinates,
        num_quantum_states,
        num_quantum_states,
    )
    batch_id = (
        np.array([np.ones(len(mels_1)) * i for i in range(batch_size)])
        .astype(int)
        .flatten()
    )
    coord_id = np.tile(inds_1[0], batch_size)
    state1_id = np.tile(inds_1[1], batch_size)
    state2_id = np.tile(inds_1[2], batch_size)
    inds = (batch_id, coord_id, state1_id, state2_id)
    mels = np.tile(mels_1, batch_size)
    return inds, mels, shape


def h_q(model, parameters, **kwargs):
    batch_size = kwargs["batch_size"]
    return np.broadcast_to(
        h_q_therm, (batch_size, num_quantum_states, num_quantum_states)
    )


def h_qc(model, parameters, **kwargs):
    z = kwargs["z"]
    h_qc = np.zeros((len(z), num_quantum_states, num_quantum_states), dtype=complex)
    for batch in range(len(z)):
        for mode in range(nmds):
            h_qc[batch] += g_q_mat[mode] * (z[batch][p_q_ind_mat*nmds + mode] + np.conj(z[batch][m_q_ind_mat*nmds + mode]))
    return h_qc


import qclab.functions as functions
def h_c_harmonic(model, parameters, **kwargs):
    """
    Harmonic oscillator classical Hamiltonian function.

    :math:`H_c = \\frac{1}{2}\sum_{n} \\left( \\frac{p_n^2}{m_n} + m_n \omega_n^2 q_n^2 \\right)`

    Keyword Args
    ------------
    z : ndarray
        Complex classical coordinate.

    Required Constants
    ------------------
    ``harmonic_frequency`` : ndarray
        Harmonic frequency of each classical coordinate.
    """
    del parameters
    z = kwargs["z"]
    w = model.constants.harmonic_frequency[np.newaxis, :]
    m = model.constants.classical_coordinate_mass[np.newaxis, :]
    h = model.constants.classical_coordinate_weight[np.newaxis, :]
    q = functions.z_to_q(z, m, h)
    p = functions.z_to_p(z, m, h)
    h_c = np.sum(0.5 * (((p**2) / m) + m * (w**2) * (q**2)), axis=-1)
    return h_c

import qclab
print(qclab.__version__)
from qclab import Model, Simulation, Data, tasks
import qclab.ingredients as ingredients
from qclab.algorithms import MeanField, FewestSwitchesSurfaceHopping
from qclab.dynamics import (
    serial_driver,
    parallel_driver_multiprocessing,
    parallel_driver_mpi,
)


class TMDModel(Model):
    def __init__(self, constants=None):
        if constants is None:
            constants = {}
        self.default_constants = {"kBT": temp_therm}
        super().__init__(self.default_constants, constants)
        self.update_dh_qc_dzc = False
        self.update_h_q = False

    def _init_model(self, parameters, **kwargs):
        self.constants.num_quantum_states = num_quantum_states
        self.constants.num_classical_coordinates = num_classical_coordinates
        self.constants.harmonic_frequency = harmonic_frequency_therm
        self.constants.classical_coordinate_mass = np.ones(
            num_classical_coordinates, dtype=float
        )
        self.constants.classical_coordinate_weight = harmonic_frequency_therm
        self.constants.init_position = np.ones(num_classical_coordinates, dtype=float)
        self.constants.init_momentum = np.ones(num_classical_coordinates, dtype=float)
        return

    ingredients = [
        ("h_q", h_q),
        ("h_qc", h_qc),
        ("h_c", ingredients.h_c_harmonic),
        ("dh_c_dzc", ingredients.dh_c_dzc_harmonic),
        ("dh_qc_dzc", dh_qc_dzc),
        ("init_classical", ingredients.init_classical_wigner_harmonic),
        ("hop", ingredients.hop_harmonic),
        ("_init_model", _init_model),
    ]


def calc_obs(sim, state, parameters, **kwargs):
    wf_db = state["wf_db"]
    if sim.t_ind == 0:
        state["wf_db_0"] = np.copy(wf_db)
    wf_db_0 = state["wf_db_0"]
    state["output_dict"]["resp"] = np.sum(np.conj(wf_db_0) * wf_db, axis=-1)
    state["output_dict"]['pops_db'] = np.real(np.einsum('tii->ti', state["dm_db"]))
    return state, parameters

def calc_phonon_occs(sim, state, parameters, **kwargs):
    z = state["z"]
    if sim.t_ind == 0:
        state["z_0"] = np.copy(z)
    state, parameters = tasks.update_quantum_classical_force(sim, state, parameters,
            wf_db_name="act_surf_wf",
            wf_changed=True,
            z_name = "z",
            quantum_classical_force_name = "f_matrix",
        )
    f = -state["f_matrix"]/sim.model.constants.harmonic_frequency[np.newaxis]
    z_offset = z - f
    z0_offset = state["z_0"] - f
    state["output_dict"]["ph_occ_shift"] = np.abs(z_offset)**2 - np.abs(z0_offset)**2
    state["output_dict"]["ph_occ"] = np.abs(z)**2 - np.abs(state["z_0"])**2
    return state, parameters
    

sim = Simulation(
    {
        "num_trajs": num_trajs,
        "batch_size": batch_size,
        "tmax": tmax,
        "dt_update": 0.001,
        "dt_collect": 0.1,
        "debug":True
    }
)
sim.model = TMDModel({"kBT": temp_therm})
myFSSH = FewestSwitchesSurfaceHopping({"gauge_fixing":"phase_overlap"})
myFSSH.collect_recipe.pop(5) # remove collect of dm_db
myFSSH.collect_recipe.append(calc_obs)
if phoccs == 1:
    myFSSH.collect_recipe.append(calc_phonon_occs)
sim.algorithm = myFSSH
sim.initial_state["wf_db"] = np.zeros((num_quantum_states), dtype=complex)
if init == "linabs": 
    wf_db = np.conj(P_x_init / np.sqrt(np.sum(np.abs(P_x_init)**2)))
    sim.initial_state["wf_db"] = np.copy(wf_db)
elif init == "depol":
    sim.initial_state["wf_db"][1] = 1
  
name_noh5 = (
    str(res)
    + "_"
    + str(Rk)
    + "_"
    + str(Rx)
    + "_"
    + "_".join(trunc_points)
    + str("_tb")
    + "_"
    + str(init)
    + "_"
    + str(temp_K)
)
name = name_noh5 + ".h5"
if run_ind == -1:
    from glob import glob
    data_files = glob(name_noh5 + "_*.h5")
    print("Found existing files:", data_files)
    data = Data()
    for file in tqdm(data_files):
        data_tmp = Data().load(file)
        data.add_data(data_tmp)
    data.save(name)
    print("Saved to ", name)
else:
    name = name_noh5 + "_" + str(run_ind) + ".h5"
    if os.path.exists(name):
        print("Found existing data file")
        exit()
    if rank == 0:
        print("Output to ", name)
    data = parallel_driver_mpi(
        sim,
        seeds=np.arange(run_ind * num_trajs, (run_ind + 1) * num_trajs, 1).astype(int),
    )
    data.log += "\nQC Lab version: " + str(qclab.__version__) +"\n"
  
if rank == 0:
    print(data.data_dict["seed"])
    print(data.log)
    data.save(name)
