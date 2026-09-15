"""IPRT tools common to all the phases.

The delta_m metric of IPRT and the grouping of its inputs work on plain
I, Q, U and V arrays, whatever phase and result file the signals come
from. The phases and their modules are described in smartg.iprt.

Key Functions
-------------
group_iquv
    Gather several IQUV result matrices into a single array.
compute_deltam
    Compute the IPRT delta_m metric between two IQUV signal sets.
"""

import numpy as np


def group_iquv(
    i_list: list[np.ndarray],
    q_list: list[np.ndarray],
    u_list: list[np.ndarray],
    v_list: list[np.ndarray],
) -> np.ndarray:
    """Gather several IQUV result matrices into a single array.

    Each matrix is flattened and the matrices are concatenated in the
    given order, so that several viewing configurations can be
    compared with a single delta_m. The matrices may come from any
    IPRT phase.

    Parameters
    ----------
    i_list : list of ndarray
        The I matrices to gather.
    q_list : list of ndarray
        The Q matrices, in the same order.
    u_list : list of ndarray
        The U matrices, in the same order.
    v_list : list of ndarray
        The V matrices, in the same order.

    Returns
    -------
    ndarray
        The gathered signals, of shape (4, nvalues).
    """
    n_values = int(0)
    i_tot = i_list[0].flatten()
    q_tot = q_list[0].flatten()
    u_tot = u_list[0].flatten()
    v_tot = v_list[0].flatten()

    for i in range(len(i_list)):
        n_values += round(i_list[i].shape[0] * i_list[i].shape[1])
        if i > 0:
            i_tot = np.concatenate((i_tot, i_list[i].flatten()))
            q_tot = np.concatenate((q_tot, q_list[i].flatten()))
            u_tot = np.concatenate((u_tot, u_list[i].flatten()))
            v_tot = np.concatenate((v_tot, v_list[i].flatten()))

    iquv_tot = np.zeros((4, n_values), dtype=np.float32)
    iquv_tot[0, :] = i_tot
    iquv_tot[1, :] = q_tot
    iquv_tot[2, :] = u_tot
    iquv_tot[3, :] = v_tot

    return iquv_tot


def compute_deltam(
    obs: np.ndarray | list[np.ndarray],
    mod: np.ndarray | list[np.ndarray],
    print_res: bool = True,
) -> np.ndarray:
    """Compute the IPRT delta_m metric from two IQUV signal sets.

    delta_m is the root mean square of the difference between the
    model and the reference, relative to the root mean square of the
    reference, in percent. It is reported per Stokes parameter, and
    is set to 0 when the reference is uniformly zero. The signals may
    come from any IPRT phase.

    Parameters
    ----------
    obs : ndarray or list of ndarray
        Observed, or reference model, I, Q, U and V signals, either
        as an array of shape (4, nvalues) or as four arrays.
    mod : ndarray or list of ndarray
        Modelled I, Q, U and V signals, in the same layout.
    print_res : bool
        Print each delta_m next to its Stokes parameter.

    Returns
    -------
    ndarray
        The delta_m of I, Q, U and V, in percent.
    """
    if isinstance(obs, np.ndarray):
        obs_tmp = obs.copy()
        obs = []
        for i in range(0, 4):
            obs.append(obs_tmp[i, :])

    if isinstance(mod, np.ndarray):
        mod_tmp = mod.copy()
        mod = []
        for i in range(0, 4):
            mod.append(mod_tmp[i, :])

    stk = ['I', 'Q', 'U', 'V']
    delta_m = np.zeros(4, dtype=np.float32)
    for i in range(len(stk)):
        with np.errstate(divide='raise', invalid='raise'):
            try:
                delta_m[i] = (
                    100
                    * np.sqrt(np.sum((obs[i] - mod[i]) ** 2))
                    / np.sqrt(np.sum(obs[i] ** 2))
                )
            except FloatingPointError:
                delta_m[i] = 0.
        if print_res:
            print(stk[i], f"{delta_m[i]:.3f}")
    return delta_m
