"""
Functions and classes for interacting with IBMQ backends
"""

import numpy as np

from qiskit import IBMQ, Aer, QuantumRegister
from qiskit.utils import QuantumInstance
from qiskit.aqua.utils.backend_utils import (
    is_ibmq_provider,
    is_simulator_backend,
)
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel
from qiskit.result import Result

NB_SHOTS_DEFAULT = 256
OPTIMIZATION_LEVEL_DEFAULT = 1
FULL_LIST_DEVICES = [
    'ibmq_rochester',
    'ibmq_paris',
    'ibmq_toronto',
    'ibmq_manhattan',
    'ibmq_qasm_simulator'
] # '', ibmq_poughkeepsie
# There may be more free devices
FREE_LIST_DEVICES = [
    'ibmq_16_melbourne',
    'ibmq_vigo',
    'ibmq_armonk',
    'ibmq_essex',
    'ibmq_burlington',
    'ibmq_london',
    'ibmq_rome',
    'qasm_simulator',
]


class ProcessedResult(Result):
    """
    The qiskit results class can be slow. This derived class pre-processes a
    results object to make calls to `.get_counts()` much faster
    """
    def __init__(self, raw_results, expand=False):
        """
        Parameters
        ----------
        raw_results : qiskit.result.Result
            Results object to process
        expand : boolean, default False
            If set to True, counts histogram expanded to have 2**(n_qubits)
            entries, i.e. including all zeros excluded by qiskit
        """

        # call base class constructor
        _date = None
        if hasattr(raw_results, 'date'):
            _date = raw_results.date
        _status = None
        if hasattr(raw_results, 'status'):
            _status = raw_results.status
        _header = None
        if hasattr(raw_results, 'header'):
            _header = raw_results.header
        super().__init__(
            backend_name=raw_results.backend_name,
            backend_version=raw_results.backend_version,
            qobj_id=raw_results.qobj_id,
            job_id=raw_results.job_id,
            success=raw_results.success,
            results=raw_results.results,
            date=_date,
            status=_status,
            header=_header,
            **raw_results._metadata,
        )

        # make results access maps for speed
        self.results_access_map = {
            res.header.name: idx for idx, res in enumerate(raw_results.results)
        }

        # get and cache counts dictionary
        self.processed_counts = []
        self.int_keys = []
        for idx, res in enumerate(raw_results.results):
            _counts_dict = raw_results.get_counts(idx)

            if expand:

                n_qubits = res.header.n_qubits
                self.int_keys.append(np.arange(2**n_qubits))
                tmp = {}
                for val in range(2**n_qubits):
                    meas_str = format(
                        int(str(bin(val))[2:], 2), '0{}b'.format(n_qubits)
                    )
                    tmp[meas_str] = _counts_dict.get(meas_str, 0)
                self.processed_counts.append(tmp)

            else:

                int_keys = [int(binval, 2) for binval in _counts_dict.keys()]

                # sort based on int_keys and save
                _order = np.argsort(int_keys)
                self.int_keys.append(np.array(int_keys)[_order])
                self.processed_counts.append(
                    dict(zip(
                        np.array(list(_counts_dict.keys()))[_order],
                        np.array(list(_counts_dict.values()))[_order]
                    ))
                )

    def get_counts(self, access_key):
        """ """
        if not isinstance(access_key, (int, str)):
            # fall back on parent class method
            return super().get_counts(access_key)

        if isinstance(access_key, str):
            access_key = self.results_access_map[access_key]

        return self.processed_counts[access_key]

    def combine_counts(self, names, save_name):
        """
        Parameters
        ----------
        names : list[int] OR list[str]
            List of experiments indexes (as either ints or strings) to combine
        save_name : str
            Name to use to save combined counts
        """
        if len(names) == 0:
            # skip if not passed any experiments
            return

        if save_name in self.results_access_map:
            # skip if already have a result with this name
            return

        if not all(isinstance(_nm, (int, str)) for _nm in names):
            raise TypeError(
                'Can only combine counts using str and int experiment'
                + ' references.'
            )

        n_qubits = None
        summed_counts = None
        for exp_name in names:
            if isinstance(exp_name, int):
                _idx = exp_name
            elif isinstance(exp_name, str):
                _idx = self.results_access_map[exp_name]

            tmp_n_qubits = self.results[_idx].header.n_qubits
            if n_qubits is None:
                n_qubits = tmp_n_qubits
            elif n_qubits != tmp_n_qubits:
                raise ValueError(
                    'Trying to combine incompatible experiments, with'
                    + ' different numbers of qubits'
                )
            if summed_counts is None:
                summed_counts = np.zeros(2**n_qubits, dtype=int)

            summed_counts[self.int_keys[_idx]] += np.array(
                list(self.processed_counts[_idx].values())
            )

        # compress by removing zeros
        _nonzero_idx = np.nonzero(summed_counts)
        int_keys = np.arange(2**n_qubits)[_nonzero_idx]
        meas_strs = [
            format(int(str(bin(val))[2:], 2), '0{}b'.format(n_qubits))
            for val in int_keys
        ]

        self.results_access_map[save_name] = len(self.results_access_map)
        self.int_keys.append(int_keys)
        self.processed_counts.append(
            dict(zip(meas_strs, summed_counts[_nonzero_idx]))
        )


def make_quantum_instance(
    backend_name,
    hub='partner-samsung',
    group='internal',
    project='imperial',
    measurement_error_mitigation=1,
    nb_shots=8192,
    cals_matrix_refresh_period=30,
    backend_options=None,
    initial_layout=None,
    simulate_ibmq=False,
    noise_model=None,
    seed_simulator=None,
):
    """
    Wraps the tasks of generating a QuantumInstance obj, getting the noise
    profile of a real device if we are doing a noisy simulation (see
    `simulate_ibmq` and `noise_model` args)

    Parameters
    ----------
    backend_name : str
    hub : str, optional
    group : str, optional
    project : str, optional
    measurement_error_mitigation : int (0 or 1)
    nb_shots : int
    cals_matrix_refresh_period : int
    backend_options : dict or None
    initial_layout : (optional) array
    simulate_ibmq : boolean, default False
        If True will return a quantum_instance that is a qasm_simulator
        simulating the `backend_name` backend
    noise_model : (optional) qiskit NoiseModel
        If simulate_ibmq=True by default it will take the current noise profile
        of the target backend, that can be overwritten by passing a NoiseModel
        obj here.
    seed_simulator : (optional) int
        Passed to QuantumInstance if a simulator

    Returns
    -------
    quantum_instance : qiskit QuantumInstance obj
    """
    IBMQ.load_account()

    premium_provider = IBMQ.get_provider(
        hub=hub,
        group=group,
        project=project,
    )
    free_provider = IBMQ.get_provider(group='open')

    premium_devices = [bknd.name() for bknd in premium_provider.backends()]
    free_devices = [bknd.name() for bknd in free_provider.backends()
                    if bknd.name() not in premium_devices]

    if backend_name in premium_devices:
        backend = premium_provider.get_backend(backend_name)
    elif backend_name in free_devices:
        backend = free_provider.get_backend(backend_name)
    else:
        backend = Aer.get_backend(backend_name)

    measurement_error_mitigation_cls = None
    if measurement_error_mitigation and 'statevector' not in backend.name():
        measurement_error_mitigation_cls = CompleteMeasFitter

    coupling_map = None
    if simulate_ibmq and not is_simulator_backend(backend):
        coupling_map = backend.configuration().coupling_map
        if noise_model is None:
            noise_model = NoiseModel.from_backend(backend.properties())
        backend = Aer.get_backend("qasm_simulator")

    return QuantumInstance(
        backend,
        shots=nb_shots,
        optimization_level=0,
        initial_layout=initial_layout,
        noise_model=noise_model,
        coupling_map=coupling_map,
        skip_qobj_validation=(not is_ibmq_provider(backend)),
        measurement_error_mitigation_cls=measurement_error_mitigation_cls,
        cals_matrix_refresh_period=cals_matrix_refresh_period,
        backend_options=backend_options,
        seed_simulator=seed_simulator,
    )


# ------------------------------------------------------
# Back end management related utilities
# ------------------------------------------------------


class BackendManager():
    """ Custom backend manager to deal with different users
    self.LIST_DEVICES : list of devices accessible to the user
    self.simulator: 'qasm_simulator' backend
    self.current_backend: 'current' backend by default the simulator but which 
        can be set updated by using get_backend method with inplace=True
    
    """
    def __init__(self):
        provider_free = IBMQ.load_account()
        try:
            self.LIST_OF_DEVICES = FULL_LIST_DEVICES
            # provider_imperial = IBMQ.get_provider(hub='ibmq', group='samsung', project='imperial')
            provider_imperial = IBMQ.get_provider(hub='partner-samsung', group='internal', project='imperial')
            self.provider_list = {'free':provider_free, 'imperial':provider_imperial}
        except:
            self.LIST_OF_DEVICES = FREE_LIST_DEVICES
            self.provider_list = {'free':provider_free}
        self.simulator = Aer.get_backend('qasm_simulator')
        self.current_backend = self.simulator
       

    # backend related utilities
    def print_backends(self):
        """List all providers by deafult or print your current provider"""
        #provider_list = {'Imperial':provider_free, 'Free':provider_imperial}
        for pro_k, pro_v in self.provider_list.items():
            print(pro_k)
            print('\n'.join(str(pro_v.backends()).split('IBMQBackend')))
            print('\n') 
        try:
            print('current backend:')
            print(self.current_backend.status())
        except:
            pass

    # backend related utilities
    def get_backend(self, name, inplace=False):
        """ Gets back end preferencing the IMPERIAL provider
        Can pass in a named string or number corresponding to get_current_status output
        Comment: The name may be confusing as a method with the same name exists in qiskit
        """
        # check if simulator is chose
        if name == len(self.LIST_OF_DEVICES) or name == 'qasm_simulator':
            temp = self.simulator
        else: # otherwise look for number/name
            if type(name) == int: name = self.LIST_OF_DEVICES[name-1]
            try: #  tries imperial first
                temp = self.provider_list['imperial'].get_backend(name)
            except:
                temp = self.provider_list['free'].get_backend(name)
                
        # if inplace update the current backend
        if inplace:
            self.current_backend = temp
        return temp

    def get_current_status(self):
        """ Prints the status of each backend """
        for ct, device in enumerate(self.LIST_OF_DEVICES): # for each device
            ba = self.get_backend(device)
            print(ct+1, ':   ', ba.status()) # print status

    def gen_instance_from_current(self, nb_shots = NB_SHOTS_DEFAULT, 
                     optim_lvl = OPTIMIZATION_LEVEL_DEFAULT,
                     noise_model = None, 
                     initial_layout=None,
                     seed_transpiler=None,
                     measurement_error_mitigation_cls=None,
                     **kwargs):
        """ Generate an instance from the current backend
        Not sure this is needed here: 
            + maybe building an instance should be decided in the main_script
            + maybe it should be done in the cost function
            + maybe there is no need for an instance and everything can be 
              dealt with transpile, compile
            
            ++ I've given the option to just spesify the gate order as intiial layout here
                however this depends on NB_qubits, so it more natural in the cost function
                but it has to be spesified for the instance here??? Don't know the best
                way forward. 
        """
        if type(initial_layout) == list:
            nb_qubits = len(initial_layout)
            logical_qubits = QuantumRegister(nb_qubits, 'logicals')  
            initial_layout = {logical_qubits[ii]:initial_layout[ii] for ii in range(nb_qubits)}
        instance = QuantumInstance(self.current_backend, shots=nb_shots,
                            optimization_level=optim_lvl, noise_model= noise_model,
                            initial_layout=initial_layout,
                            seed_transpiler=seed_transpiler,
                            skip_qobj_validation=(not is_ibmq_provider(self.current_backend)),
                            measurement_error_mitigation_cls=measurement_error_mitigation_cls,
                            **kwargs)
        print('Generated a new quantum instance')
        return instance

    def gen_noise_model_from_backend(self, name_backend='ibmq_essex', 
                                     readout_error=True, gate_error=True):
        """ Given a backend name (or int) return the noise model associated"""
        backend = self.get_backend(name_backend)
        properties = backend.properties()
        noise_model = noise.device.basic_device_noise_model(properties, 
                            readout_error=readout_error, gate_error=gate_error)
        return noise_model
    
