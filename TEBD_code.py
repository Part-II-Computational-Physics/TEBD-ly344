import numpy as np
from numpy import linalg as LA
import scipy as sci
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, eigs
from typing import Optional
from collections.abc import Iterable
from ncon import ncon
import scipy.sparse.linalg as spsl
from scipy.stats import ortho_group # Requires version 0.18 of scipy
import matplotlib.pyplot as plt
from tenpy.networks.mps import MPS
from tenpy.algorithms.tebd import TEBDEngine
from tenpy.models.tf_ising import TFIChain
from tenpy.networks.site import SpinSite
from tenpy.algorithms import tebd
from tfi_exact import finite_gs_energy
import scipy.sparse as sparse
import scipy.sparse.linalg.eigen.arpack as arp
import warnings
import scipy.integrate



# Copyright 2019-2023 TeNPy Developers, GNU GPLv3

def gs_energy_ising_finite(L, g, chi, N_steps, trotter_order = 2):
    
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None)
    M = TFIChain(model_params)
    
    local_state = ["up", "down"]
    product_state = [local_state[np.random.randint(2)] for i in range(0, M.lat.N_sites, 1)]
    
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    tebd_params = {
        'order': trotter_order,
        'delta_tau_list': [0.1, 0.01, 0.001, 1.e-4, 1.e-5],
        'N_steps': N_steps,
        'max_error_E': 1.e-6,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10
        },
    }
    eng = tebd.TEBDEngine(psi, M, tebd_params)
    eng.run_GS()  # the main work...

    # expectation values
    E = np.sum(M.bond_energies(psi))  # M.bond_energies() works only a for NearestNeighborModel
    # alternative: directly measure E2 = np.sum(psi.expectation_value(M.H_bond[1:]))
    
    trunc_err = eng.trunc_err.eps
    
    return E, psi, M, trunc_err

def evolve_mps_imag_time(L, g, chi, N_steps, psi, dt, trotter_order = 2):
    
    model_params = dict(L=L, J=1., g=g, bc_MPS='finite', conserve=None)
    M = TFIChain(model_params)
    
    tebd_params = {
        'order': trotter_order,
        'delta_tau_list': [dt],
        'N_steps': N_steps,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10
        },
    }
    eng = tebd.TEBDEngine(psi, M, tebd_params)
    eng.run_imag()  # the main work...

    # expectation values
    E = np.sum(M.bond_energies(psi))  # M.bond_energies() works only a for NearestNeighborModel
    # alternative: directly measure E2 = np.sum(psi.expectation_value(M.H_bond[1:]))
    
    trunc_err = eng.trunc_err.eps
    
    return E, psi, M, trunc_err

def evolve_mps_real_time(L, g, chi, N_steps, psi, dt, trotter_order = 2):
    model_params = dict(L=L, J=1., g=1, bc_MPS='finite', conserve=None)
    ham = TFIChain(model_params)
    
    tebd_params = {
        'order': trotter_order,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10
        },
    }
    eng = tebd.TEBDEngine(psi, ham, tebd_params)

    eng.run_evolution(N_steps, dt)
    
    return psi

def find_energy(L, psi):
    model_params = dict(L=L, J=1., g=1, bc_MPS='finite', conserve=None)
    M = TFIChain(model_params)
    E = np.sum(M.bond_energies(psi))
    return E

def sigma_x(psi):
    result = np.sum(psi.expectation_value("Sigmax"))
    return result

def sigma_z(psi):
    result = np.sum(psi.expectation_value("Sigmaz"))
    return result

def finite_gs_energy(L, J, g):
    """For comparison: obtain ground state energy from exact diagonalization.
    Exponentially expensive in L, only works for small enough `L` <~ 20.
    """
    if L >= 20:
        warnings.warn("Large L: Exact diagonalization might take a long time!")
    # get single site operaors
    sx = sparse.csr_matrix(np.array([[0., 1.], [1., 0.]]))
    sz = sparse.csr_matrix(np.array([[1., 0.], [0., -1.]]))
    id = sparse.csr_matrix(np.eye(2))
    sx_list = []  # sx_list[i] = kron([id, id, ..., id, sx, id, .... id])
    sz_list = []
    for i_site in range(L):
        x_ops = [id] * L
        z_ops = [id] * L
        x_ops[i_site] = sx
        z_ops[i_site] = sz
        X = x_ops[0]
        Z = z_ops[0]
        for j in range(1, L):
            X = sparse.kron(X, x_ops[j], 'csr')
            Z = sparse.kron(Z, z_ops[j], 'csr')
        sx_list.append(X)
        sz_list.append(Z)
    H_xx = sparse.csr_matrix((2**L, 2**L))
    H_z = sparse.csr_matrix((2**L, 2**L))
    for i in range(L - 1):
        H_xx = H_xx + sx_list[i] * sx_list[(i + 1) % L]
    for i in range(L):
        H_z = H_z + sz_list[i]
    H = -J * H_xx - g * H_z
    E, V = arp.eigsh(H, k=1, which='SA', return_eigenvectors=True, ncv=20)
    return E[0]