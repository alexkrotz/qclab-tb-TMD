import numpy as np
import itertools
import os


hartree_to_eV = 27.2114 # eV/Ha
ryd_to_eV = 0.5*hartree_to_eV
ryd_to_meV = 1000*ryd_to_eV
invcm_to_eV = 1/8065.54429
hartree_to_meV = 1000*hartree_to_eV
meV_to_293K = 1/(25.2488)

def clip_to_contraction(S, tol=1e-12):
    U, s, Vh = np.linalg.svd(S, full_matrices=False)
    s_clipped = np.minimum(s, 1.0)
    out = (U * s_clipped) @ Vh  # broadcasting scales U's columns
    return out
def polar_from_rect(S):
    # Unit, stable polar: S (S^H S)^-1/2
    H = S.conj().T @ S
    w, V = np.linalg.eigh(H)                   # H is Hermitian PSD
    w_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(w, 1e-14)))
    H_inv_sqrt = V @ w_inv_sqrt @ V.conj().T
    return S @ H_inv_sqrt

def get_ind(point,point_list):
    dist = np.sum(np.abs(point_list - point[np.newaxis]),axis=-1)
    if np.min(dist) < 1e-5:
        return np.where(dist < 1e-5)[0][0], True
    else:
        # print('Gridpoint not found', dist)
        return np.where(dist < 1e-5)[0], False

MoS2 = {
    "t":1.10,										# eV
    "lambda_c":-3.0e-3,								# eV
    "lambda_v":148e-3,								# eV
    "E_g":1.66,  # eV
    "ge1K1K1ac_mev": 0.109437 * np.sqrt(1/2) * 1000, # meV #4.33/np.sqrt(2),
    "ge0K1K1op_mev": 0.0956642 * np.sqrt(2/3) * 1000, #3.79 * np.sqrt(2/3),
    "gh1K1K1ac_mev": -0.0607982 * np.sqrt(1/2) * 1000, #-2.41 / np.sqrt(2),
    "gh0K1K1op_mev": -0.0758716 * np.sqrt(2/3) * 1000, #-3.00 * np.sqrt(2/3),
    "lat":3.193 / 0.5292, # in Bohr
    "chi_2D":6.60 / 0.5292, # in Bohr
    "w_opt_mev": 1.90/meV_to_293K,#47.99, # in meV
    "w_ac_mev": 3.39/meV_to_293K,#85.48, # mev * |q| (in 2pi/a) * this is really the sound velocity times some constants. 
}


def read_input_file(filename):
    variables = {}
    with open(filename) as f:
        code = f.read()
    exec(code, {}, variables)  # run file contents in an empty dict
    return variables

reciprocal_lattice = np.array([[1.000000000000000e0, 5.773502691895850e-1],
                                [0.000000000000000e0, 1.154700538379170e0]]) # in units of 2\pi/a
reciprocal_lattice_crystal = np.array([[1, 0],[0, 1]])
high_symmetry_points = {"G":np.array([0,0]),
                        "M1":0.5*reciprocal_lattice[0],
                        "M2":-0.5*reciprocal_lattice[0],
                        "K1":(1/3)*reciprocal_lattice[0] + (1/3)*reciprocal_lattice[1],
                        "Q1":(1/6)*reciprocal_lattice[0] + (1/6)*reciprocal_lattice[1],
                        "K2":(-1/3)*reciprocal_lattice[0] + (-1/3)*reciprocal_lattice[1],
                        "Q2":(-1/6)*reciprocal_lattice[0] + (-1/6)*reciprocal_lattice[1],
                       }
high_symmetry_points_crystal = {"G":np.array([0,0]),
                        "M1":0.5*reciprocal_lattice_crystal[0],
                        "M2":-0.5*reciprocal_lattice_crystal[0],
                        "K1":(1/3)*reciprocal_lattice_crystal[0] + (1/3)*reciprocal_lattice_crystal[1],
                        "Q1":(1/6)*reciprocal_lattice_crystal[0] + (1/6)*reciprocal_lattice_crystal[1],
                        "K2":(-1/3)*reciprocal_lattice_crystal[0] + (-1/3)*reciprocal_lattice_crystal[1],
                        "Q2":(-1/6)*reciprocal_lattice_crystal[0] + (-1/6)*reciprocal_lattice_crystal[1],
                       }


centers = np.sum(
        np.array(tuple(
            itertools.product(np.outer([-2, -1, 0, 1, 2],reciprocal_lattice[0][:2]), np.outer([2, -1, 0, 1, 2], reciprocal_lattice[1][:2])))),
        axis=1)
centers_crystal = np.sum(
        np.array(tuple(
            itertools.product(np.outer([-2, -1, 0, 1, 2],reciprocal_lattice_crystal[0][:2]), np.outer([-2, -1, 0, 1, 2], reciprocal_lattice_crystal[1][:2])))),
        axis=1)

def k_Umklapp2(k):
    k_array = np.zeros((len(k), len(centers), 2))
    center_array = np.zeros((len(k), len(centers), 2))
    center_array[:] = centers
    for n in range(len(centers)):
        k_array[:, n, :] = k - center_array[:, n, :]
    x_array = np.abs(k_array[:, :, 0])
    y_array = np.abs(k_array[:, :, 1])
    norm_array = np.round(np.sqrt(x_array ** 2 + y_array ** 2), 7)
    mask_array = np.argmin(norm_array, axis=1)
    return k - centers[mask_array]

def write_folder(folder_name):
    if not(os.path.exists(folder_name)):
        os.mkdir(folder_name)
    # else:
    #     print(folder_name, " already exists, skipping.")
    return
    
def screened_int(k_dist, chi_2D):
    return 2 * np.pi / (k_dist * (1 + 2 * np.pi * chi_2D * k_dist))


def screened_int_near_zero(d_k, chi_2D):
    # perform a 4D integration over two cubes separated by 0, with sidelength d_k
    res_fine = 4
    d_k_fine = d_k / res_fine
    k_combis = np.array(
        list(itertools.product(range(res_fine), range(res_fine), range(res_fine), range(res_fine)))) * d_k_fine
    out =  np.sum(
        screened_int(np.sqrt((k_combis[:, 0] - k_combis[:, 2]) ** 2 + (k_combis[:, 1] - k_combis[:, 3]) ** 2) + 1e-4,
                     chi_2D)) * d_k_fine ** 4
    return out
    
def unscreened_int(k_dist):
    if k_dist < 1e-10:
        return 0
    return 2*np.pi/(k_dist)

def keldysh(k_dist, chi_2D, d_k):
    zer_pos = k_dist < 1e-10
    out = np.zeros(len(k_dist))
    out[zer_pos] = screened_int_near_zero(d_k, chi_2D)
    out[~zer_pos] = screened_int(k_dist[~zer_pos], chi_2D)
    return out

def load_exciton_data(q_ind, exciton_dir):
    evecs = np.load(exciton_dir + '/evecs_'+str(q_ind)+'.npy')
    evals = np.load(exciton_dir + '/evals_'+str(q_ind)+'.npy')
    full_ids = np.load(exciton_dir + '/full_ids_'+str(q_ind)+'.npy')
    return evecs, full_ids, evals

def load_exciton_spin_data(q_ind, exciton_dir):
    spins = np.load(exciton_dir + '/spins_'+str(q_ind)+'.npy')
    return spins

def does_q_exist(q_ind, exciton_dir):
    if os.path.exists(exciton_dir + '/evals_' + str(q_ind)+'.npy') and \
    os.path.exists(exciton_dir + '/evecs_' + str(q_ind) + '.npy'):
        #print('found', q_ind)
        return True
    else:
        return False

def does_exph_exist(q1_ind, q2_ind, exph_dir):
    if os.path.exists(exph_dir + '/g_exph_' + str(q1_ind) + '_' + str(q2_ind) + '.npy'):
        return True
    else:
        return False
    
def my_block(N, size, r):
    i0 = (N * r) // size
    i1 = (N * (r + 1)) // size
    return i0, i1

def load_exph(q1_ind, q2_ind, exph_dir):
    if not os.path.exists(exph_dir + '/g_exph_' + str(q1_ind) + '_' + str(q2_ind) + '.npy'):
        print('Error: exph file does not exist:', exph_dir + '/g_exph_' + str(q1_ind) + '_' + str(q2_ind) + '.npy')
        exit()
    g_exph = np.load(exph_dir + '/g_exph_' + str(q1_ind) + '_' + str(q2_ind) + '.npy')
    return g_exph
