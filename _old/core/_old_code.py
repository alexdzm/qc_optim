

class Batch():
    """ New class that accepts a list of ansatz ciruits OR list of cost functinos 
        OR list of gate maps, and will package them together to to run more 
        efficiently in a queue. Also has methods to update a list of Basiean 
        optimizers with new results 
        TODO: allow class to accept pre-transpile circs as inputs """
    def __init__(self, cost_list = None, instance = None):
        self._last_param_list = None
        self._last_results_obj = None
        self.instance = instance
        self.cost_list = cost_list
    
    def __call__(self, param_list):
        """ Efficient call to IBMQ devices that packages everything as a single job \n
            * param_list = 2d array (1 param per cost) \n
            * OR \n
            * param_list = 3d array (list of params per cost)
            * Each dim need not be equal
            TODO: add support for spreading across multiple jobs if > 900 circs
            TODO: better check for inputs
            TODO: add check for cost.instances and input instance to ensure transposition is for right backend"""
        assert len(param_list) == len(self.cost_list), " Incompatable length of inputs"
        if not hasattr(self.cost_list, '__iter__'):
            raise LookupError("cost functions not created yet")
        if not hasattr(param_list[0][0], '__iter__'):
            param_list = [[list(par)] for par in param_list]
        self._last_param_list = param_list
        circs_to_ex = self._batch_package(param_list)
        results_obj = self.instance.execute(circs_to_ex, had_transpiled=True)
        costs = self._batch_evaluate_results(results_obj)
        return costs


    def _batch_create(self, gate_map, ansatz, cost_function, 
                      nb_params, nb_qubits,
                      be_manager, nb_shots, optim_lvl, seed):
        raise NotImplementedError
        def _gen_inst_list(self, nb_shots, optim_lvl):
            """ Generates a list of instances with different layouts from which the 
                circuits are transpiled. These instances are NEVER used to execute"""
            gate_map = self.gate_map
            inst_list = []
            for gm in gate_map:
                inst = self._backend_manager.gen_instance_from_current(nb_shots=nb_shots,
                                                                       optim_lvl=optim_lvl,
                                                                       initial_layout=gm,
                                                                       seed_transpiler=self.seed)
                inst_list.append(inst)
            return inst_list
        self._instance_list = self._gen_inst_list(nb_shots=nb_shots,
                                                  optim_lvl=optim_lvl)
        self.seed = seed
        self._backend_manager = be_manager
        self.ansatz = np.atleast_1d(ansatz).tolist()
        self.cost_function = np.atleast_1d(cost_function).tolist()
        self.gate_map = np.atleast_2d(gate_map).tolist()
        """ Returns a list of cost classes for each of inputs"""
        if len(self.cost_function) > 1 and len(self.ansatz) > 1 and len(self.gate_map) > 1:
            raise NotImplementedError()
        cost_list = []    
        if len(self.cost_function) > 1:
            # iter over cost functions
            instance = self._instance_list[0]
            ansatz = self.ansatz[0]
            for cf in self.cost_function:
                cost_list.append(cf(ansatz = ansatz,
                                    N = nb_qubits,
                                    instance = instance,
                                    nb_params = nb_params))
        elif len(self.ansatz) > 1:
            # iter over ansatz
            cost_function = self.cost_function[0]
            instance = self._instance_list[0]
            for ans in self.ansatz:
                cost_list.append(cost_function(ansatz = ans,
                                               N = nb_qubits, 
                                               instance = instance, 
                                               nb_params = nb_params))
        elif len(self.gate_map) > 1:
            # iter over gate map
            cost_function = self.cost_function[0]
            ansatz = self.ansatz[0]
            for inst in self._instance_list:
                cost_list.append(cost_function(ansatz = ansatz,
                                               N = nb_qubits, 
                                               instance = inst, 
                                               nb_params = nb_params))
        return cost_list
    
    
    def _batch_package(self, param_list = None):
        """ Takes several parameter per cost function and returns a list of circuits 
            that can be executed in a single job. \n
            * Assumed to be 3d array by this state
            * Assumes input is [[c1_p1, c1_p2...], ...[cn_p1, cn_p2...]] where ci_pj is a LIST of parameters \n
            * Dimentions don't have to agree \n
            TODO: add ability to name circs"""
        if  not hasattr(param_list, '__iter__'):
            param_list = self._last_param_list
        cost_list = self.cost_list
        circs_to_ex = []
        for c, p_list in zip(cost_list, param_list):
            for p in p_list:
                circs_to_ex += bind_params(c.meas_circuits, p, c._qk_vars)
        assert len(circs_to_ex) < 900, "Total number of measurement circs do not fit in single job."
        return circs_to_ex
    
    
    def _batch_evaluate_results(self, results_obj):
        """ Uses each cost in cost_list to evaluate the cost function from the 
            given results object.\n
            * Currently only works IFF the results_job was created just before in __call__
            TODO: allow passing circuit names to evaluate
            TODO: allow cross evaluation of results between cost functions
            """
        evaluation_list = []
        cost_list = self.cost_list
        param_list = self._last_param_list
        ct = 0
        for cost_obj, param_ls in zip(cost_list, param_list):
            evals = []
            for param in param_ls:
                relevant_counts = []
                for ii in range(len(cost_obj.meas_circuits)):
                    relevant_counts += [results_obj.get_counts(ii + ct)]
                ct += ii + 1
                evals.append(cost_obj._meas_func(relevant_counts))
            evaluation_list.append(evals)
        return evaluation_list
    
    
    def shot_noise(self, param_list, nb_shots):
        """ Package shot noise to get, accepts single param or list containing 
            single param per cost function, accets independed nb_shots for each
            cost"""
        if isinstance(param_list[0], (int, float)):
            param_list = [param_list] * len(self.cost_list)
        if type(nb_shots) == int:
            nb_shots = [nb_shots] * len(param_list)
        param_list_new = [[par]*sh for par, sh in zip(param_list, nb_shots)]
        return self.__call__(param_list_new)
    
        
    def check_cost_functions(self):
        """ Looks to see if each individual cost function passes all it's tests
            TODO: remove and put in batch_create function."""
        test = True
        for c in self.cost_list:
            test &= c.check_depth()
            if self.instance.backend.name() != 'qasm_simulator':
                test &= c.check_layout()
        return test


    def print(self, arg):
        """ Use keyword to inspect elements of cost list \n
            arg = cost, main, gate"""
        if arg == 'cost':
            print(*self.cost_list, sep = '\n')
        elif arg == 'main':
            for cs in self.cost_list:
                print(cs.main_circuit)
        elif arg == 'gate':
            print(*self.gate_map, sep = '\n')
            
            
    # the following functions should probably implimented in extended classes just put here for now 
    def get_new_param_points(self, bo_list):
        """ Returns new param points (for each bopt in the list) to try"""
        [bopt._update_model(bopt.normalization_type) for bopt in bo_list] # to verif
        x_new = [bopt._compute_next_evaluations() for bopt in bo_list]
        x_new = np.squeeze(x_new)
        return x_new
    
    
    def update_bo_inplace(self, bo_list, x_new, y_new, share_results=False):
        """ Updates each bo in list with each x_new and y_new \n
            TODO: impliment sharing of results between BO's"""
        if share_results:
            raise NotImplementedError()
        for ii in range(len(bo_list)):
            bo_list[ii].X = np.vstack((bo_list[ii].X, x_new[ii]))
            bo_list[ii].Y = np.vstack((bo_list[ii].Y, y_new[ii]))
    
    
            
        