"""
Functions and classes for interacting with IBMQ backends
"""

from qiskit import IBMQ, Aer, QuantumRegister
from qiskit.utils import QuantumInstance
from qiskit.aqua.utils.backend_utils import (
    is_ibmq_provider,
    is_simulator_backend,
)
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter
from qiskit.providers.aer import noise
from qiskit.providers.aer.noise import NoiseModel

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
    
