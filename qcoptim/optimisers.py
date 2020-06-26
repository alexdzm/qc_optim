
# list of * contents
__all__ = [
    'Method',
    'ParallelRunner',
    'MethodBO',
    'MethodSPSA',
    'check_cost_objs_consistency',
    'SingleBO',
    'SingleSPSA'
]

import sys
import pdb
import copy
from abc import ABC, abstractmethod

import GPyOpt
import numpy as np

from . import utilities as ut
from . import cost

pi = np.pi

class Method(ABC):
    """
    Interface for an optimisation method. The optimiser must have two main methods:
    one that returns a list of parameters to test and the second to update the 
    internal state. The other methods deal with creating and updating optim hyper params
    """
    
    def __init__(self, args = None):
        self._nb_request = 1 # nb of points requested at each step
        self._iter = 0 # keep track of iterations
        self._type = None # type
        self._iter_max = None
        self._best_x = None
        self._sub_class_init(args = args)
        
        
    def __call__(self, args):
        """ Allow's quick init of internal optim hyperparams etc..."""
        self._sub_class_init(args = args)
    
    @abstractmethod
    def _sub_class_init(self, args):
        """ Any method spesific initial points to be passed here"""
        raise NotImplementedError
        
    @abstractmethod
    def next_evaluation_params(self):
        """ Returns a list of next parameters to be evaluated"""
        raise NotImplementedError
        
    @abstractmethod
    def update(self,x,y):
        """ 
        Process a new set of paramater evaluation points
        
        Parameters:
        ------------
        x: List of new parameter points evaluate
        
        y: Cost functgino evaluation(s) corresponding to x
        """
        raise NotImplementedError
    
    @property
    def iter(self):
        """ Returns number of time this method has been itterated"""
        return self._iter

    @property
    def best_x(self):
        """ Returns the best guess for current optimum"""
        return self._best_x


    def _run_with_cost(self, nb_iter, cost):
        """ 
        Run the full optimization as long a cost is provided 
        Parameters
        ----------
        nb_iter : int
            Number of steps
        cost: Cost
            a callable taking 1d-2d arrays of X and returning 2d arrays of Y
        Parameters
        ---------- 
        It may be moved somewhere else
        """
        for n in range(nb_iter):
            x_new = self.next_evaluation_params()
            y_new = cost(x_new)
            self.update(x_new, y_new)

class MethodBO(Method):
    """
    Creates a warapper around GPyOpt.methods.BayesianOptimization 
    TODO: .update() implement by-hand updating of dynamic weights/update model 
    """
    @property
    def best_x(self):
        """ Update and return the best guess for current optimum"""
        self._best_x = self.optimiser.X[np.argmin(self.optimiser.model.predict(self.optimiser.X, with_noise=False)[0])]
        return self._best_x
    
    def _sub_class_init(self, args):
        """
        BO spesific init, optimiser is constructor if created without args, else
        optimiser is a initialized optimiser
        
        Parameters
        ------------------
        args: Input dict for a GPyOpt.methods.BayesianOptimization object. If args=None
            optimizer is the class constructor
        """
        self._type = 'BO'
        args_cp = copy.deepcopy(args)
        if args_cp['X'] is None:
            self.evaluated_init = False
            self._nb_init = args_cp['initial_design_numdata']
            args_cp['initial_design_numdata'] = 0
        else:
            self.evaluated_init = True
        self.optimiser = GPyOpt.methods.BayesianOptimization(**args_cp)
        # workaround
        self.optimiser.num_acquisitions = 1
        self._setup_dynamic_weights(args_cp)
        self._args = args_cp
        
    def next_evaluation_params(self):
        """
        Returns the next evaluation points requested by this optimiser
        """
        if self.evaluated_init:
            x_new = self.optimiser._compute_next_evaluations()
        else:
            size = self._nb_init
            x_new = self._get_random_points_in_domain(size = size)
        return x_new
    
    def update(self, x_new, y_new):
        """
        Updates the interal state of the optimiser with the data provided
        
        TODO: Fix input input so update can accept vectors. 
        TODO: User better method to update internal model
        
        Parametres:
        -------------
        x_new: Parameter points that were requested/provided
        
        y_new: Cost functino evalutations for those parameter points
        """
        #raise Warning('Need to fix dims: can currently only update one at a time')
        if not self.evaluated_init:
            self.optimiser.X = np.array([x_new])
            self.optimiser.Y = np.array([[y_new]])
        else:
            self.optimiser.X = np.vstack((self.optimiser.X, x_new))
            self.optimiser.Y = np.vstack((self.optimiser.Y, y_new))
        # update
        self.optimiser._update_model(self.optimiser.normalization_type)                
        if(self._acq_weights_update): 
            self._update_weights(self.optimiser)
        
        # It could also be + len(x_new)
        self._iter += 1
        self.optimiser.num_acquisitions += 1

        # self.optimiser._update_model(self.optimiser.normalization_type)
        # self._best_x = self.optimiser.X[np.argmin(self.optimiser.model.predict(self.optimiser.X, with_noise=False)[0])]
        self.evaluated_init = True
        
    def _get_random_points_in_domain(self, size=1):
        """ 
        Generate a requested number of random points distributed uniformly
        over the domain of the BO parameters. (moved from ParallelRunner)
        
        Parameters:
        ----------
        size: Number of random points requested
        """
        for idx,dirn in enumerate(self._args['domain']):
            assert int(dirn['name'])==idx, 'BO domain dims not being returned in correct order.'
            assert dirn['type']=='continuous', 'BO domain is not continuous, this is not supported.'
    
            dirn_min = dirn['domain'][0]
            dirn_max = dirn['domain'][1]
            if idx==0:
                rand_points = np.random.uniform(dirn_min, dirn_max, size=(size,1))
            else:
                _next =  np.random.uniform(dirn_min, dirn_max, size=(size,1))
                rand_points = np.hstack((rand_points,_next))
    
        return rand_points
    
    def _setup_dynamic_weights(self, bo_args):
        """ building the rules to update the weights of the LCB acquisition 
        function. 
        Only effectively used if several conditions are fulfilled.
        Parameters:
        -------------
        bo_args: dict
            args to sub_init BO
            needs 'acquisition_weight_lindec' to be True and 
            'nb_iter' needs to be filled to be activated
        """
        if(bo_args.get('acquisition_weight_lindec') == True):
            if(bo_args.get('nb_iter') is not None):
                self._acq_weights_update = True
                nb_iter_total = bo_args['nb_iter']
                w_init = bo_args['acquisition_weight']
                def update_weights(bo):
                    new_w = max(0.000001, w_init  * (1 - bo.num_acquisitions / nb_iter_total))
                    bo.acquisition.exploration_weight = new_w
                self._update_weights = update_weights
            else:
                raise Warning('to allow for weights decay, you need to pass a nb_iter')
                self._acq_weights_update = False
                self._update_weights = False
        else:
            self._acq_weights_update = False
            self._update_weights = False
        
class MethodSPSA(Method):
    """ Implementation of the Simultaneous Perturbation Stochastic Algo,
    Implemented to perform minimization (can be extended for maximization)
    """
    def _sub_class_init(self, args):
        """ 
        Parameters
        ----------
            cost_obj : Cost object
            verbose : bool, optional
                Set level of output of the object
            args : dict with keys/values
                'x_init': None or np.array
                'domain': None or list of N tuples
                'a': float
                'b':float 
                's':float
                't':float
                'A':float
                'minimize':True (optional it will behave as True by default)
            typical_args = {'a':1, 'b':0.628, 's':0.602, 't':0.101,'A':0,'domain':[(0,1)]}
        Comments
        ----------
        Implementation follows [Spall98] (with alpha->s and gamma->t)
        + additional restricted domain
        """
        self._type = 'SPSA'
        self._nb_request = 2
        self._updated = True
        domain = args['domain']
        x_init = args['x_init']
        if domain is None:
            self._x_min, self._x_max = -np.inf, np.inf
            assert x_init is not None, "If domain is None, x_init should be specified"
            x_init = np.atleast_1d(np.squeeze(x_init))
            assert np.ndim(x_init) == 1, "x_init should be a single set of paarmeters"
        else:
            self._x_min, self._x_max = np.array(domain)[:,0], np.array(domain)[:,1]
            if x_init is None:
                x_init = np.array([np.random.uniform(*d) for d in domain])
        self.domain = domain
        self.x_init = x_init
        
        # KIRAN: Is this correct? 
        self.nb_params = len(x_init)
        
        self._best_x = x_init
        self._args = args
        self._x = [x_init] # track x
        self._x_mp = [] # track x -/+ perturbations
        self._y_mp = []  # track y for x -/+ perturbations
        
        #Scheduleof the perturbations and step sizes
        a, A, s, b, t = [args[k] for k in ['a', 'A','s','b','t']]
        self._alpha_schedule = lambda k: a / np.power(k+1+A, s) # schedule size perturbation
        self._beta_schedule = lambda k: b / np.power(k+1, t)  # schedule step
        self._minimize = args.get('minimize', True)
        if self._minimize:
            self._factor_minimize = 1
        else:
            self._factor_minimize = -1

    def next_evaluation_params(self):
        """ Needs evaluation of 2 points: x_m (x minus some perturbation) and 
        x_p (x plus some perturbation)"""
        b_k = self._beta_schedule(self._iter) # size of the perturbation
        eps = np.sign(np.random.uniform(0, 1, self.nb_params) - 0.5) # direction of the perturbation
        x_last = self._x[-1]
        x_p = np.clip(x_last + b_k * eps, self._x_min, self._x_max)
        x_m = np.clip(x_last - b_k * eps, self._x_min, self._x_max)
        return np.array([x_m, x_p])

    def update(self, x_new, y_new):
        """ Process a new set of X, Y to update fully or partially the state 
        of the optimizer
        Parameters
        ----------
        X: 1d 2d array 
           after being expanded to a 2d array it should have either one or 2 
           elements in the first dimension 
        Y: 1d 2d array 
           after being expanded to a 2d array it should have either one or 2 
           elements in the first dimension 
        Comment
        ----------
        A complete update is performed if x_new and y_new contains both 2 elements
            then the finite difference gradient can be computed
        A partial update is performed when each element of x_new, y_new 
            are updated separately. The full update will happen when it receives 
            the second element
        """
        x_new, y_new = np.atleast_2d(x_new), np.atleast_2d(y_new)
        assert (len(x_new) == len(y_new)) & (len(x_new) in [1, 2])
        if len(x_new) == 1:
            if self._updated:
                self._partial_x = x_new
                self._partial_y = y_new
                self._updated = False
            else:
                x_new = np.vstack((self._partial_x, x_new))
                y_new = np.vstack((self._partial_y, y_new))
                self._partial_x = None
                self._partial_y = None
                self.update(x_new, y_new)
        elif len(x_new) == 2:
            x_m, x_p = x_new
            y_m, y_p = y_new
            g_k = np.squeeze((y_m - y_p))/(x_p - x_m) #finite diff gradient approx 
            a_k = self._alpha_schedule(self._iter) # step size
            self._best_x = np.clip(self._x[-1] + self._factor_minimize * a_k * g_k, self._x_min, self._x_max)
            self._x_mp.append(x_new)
            self._y_mp.append(y_new)
            self._x.append(self._best_x)
            self._updated = True
            self._iter += 1
        else:
            raise IndexError("update X vector must be <= len 2")

class ParallelRunner():
    """ 
    Class that wraps a set of quantum optimisation tasks. It separates 
    out the cost function evaluation requests from the updating of the 
    internal state of the optimisers to allow aggregation of quantum 
    jobs. It also supports different information sharing approaches 
    between the set of optimisers (see 'method' arg under __init__)

    TODO
    ----
    _cross_evaluation : allow vectorized verion for fast evaluation?
    add extra checks to inputs etc...
    Fix padding circuits
    Fix updating bug in 'shared' method ()
    Systematic generation of x_new points??? 
    """

    def __init__(self, 
                 cost_objs,
                 optimizer, # to replace default BO, extend to list? 
                 optimizer_args = None, # also allow list of input args
                 method = 'shared',
                 share_init = True): 
        """ 
        Parameters
        ----------
        cost_objs : list of Cost objects
            Cost functions being max/minimised by the internal optimsers
        optimizer : **class/list of classes under some interface?**
            Class(es) of individual internal optimiser objects
        optimizer_args : { dict, list of dicts }
            The initialisation args to pass to the internal optimisation 
            objects, either a single set to be passed to all or a list to
            be distributed over the optimisers
        method : {'independent','shared','random','left','right'}
            This controls the evaluation sharing of the internal optimiser 
            objects, cases:
                'independent' : The optimiser do not share data, each only 
                    recieves its own evaluations.
                'shared' :  Each optimiser obj gains access to evaluations 
                    of all the others. 
                'random1' : The optimsers do not get the evaluations others 
                    have requested, but in addition to their own they get an 
                    equivalent number of randomly chosen parameter points 
                'random2' : The optimisers do not get the evaluations others 
                    have requested, but in addition to their own they get an 
                    equivalent number of randomly chosen parameter points. 
                    These points are not chosen fully at random, but instead 
                    if x1 and x2 are opt[1] and opt[2]'s chosen evaluations 
                    respectively then opt[1] get an additional point y2 that 
                    is |x2-x1| away from x1 but in a random direction, 
                    similar for opt[2], etc. (Only really relevant to BO.)
                'left', 'right' : Implement information sharing but in a 
                    directional way, so that (using 'left' as an example) 
                    opt[1] gets its evaluation as well as opt[0]; opt[2] gets 
                    its point as well as opt[1] and opt[0], etc. To ensure all 
                    BO's get an equal number of evaluations this is padded 
                    with random points. These points are not chosen fully at 
                    random, they are chosen in the same way as 'random2' 
                    described above. (Only really relevant to BO.)
        share_init : boolean, optional
            Do the optimiser objects share initialisation data, or does each
            generate their own set?
        init_jobs : int, default 1
            (BO) The number of qiskit jobs to use to generate initial data. 
            (Most real device backends accept up to 900 circuits in one job.)
        """
        # make (almost certainly) unique id
        self._prefix = ut.safe_string.gen(5) 

        # check the method arg is recognised
        if not method in ['independent','shared','left','right']:
            print('method '+f'{method}'+' not recognised, please choose: '
                +'"independent", "shared", "left" or "right".',file=sys.stderr)
            raise ValueError
        elif method in ['random1','random2']:
            raise NotImplementedError

        # store inputs
        self.cost_objs = cost_objs

        self.method = method
        self._share_init = share_init
        
        # make internal assets
        self.optim_list = self._gen_optim_list(optimizer, optimizer_args)
        self._sharing_matrix = self._gen_sharing_matrix()
        self.circs_to_exec = None
        self._parallel_x = {}
        self._parallel_id = {}
        self._last_results_obj = None
        self._last_x_new = None
        
        # Assumes all inputs are either all init or all not (add extra check)
        if hasattr(self.optim_list[0], 'evaluated_init'):
            self._evaluated_init = self.optim_list[0].evaluated_init
        else:
            self._evaluated_init = False
        
        
        # unused currently
        # self.optimizer = optimizer
        self.optimizer_args = optimizer_args

    
    def _gen_optim_list(self, optimizer, optimizer_args):
        """ 
        Generate the list of internal optimser objects, takes list of optimizer_args, 
        or list of args and list of optimizers
        
        Parameters:
        ---------
        optimizer: An instance of the Method class, can be a list of different methods
        
        optimizer_args: Either None, a single args dict,or list of args dicts to initalize the optimizer
            if None, it assumes optimizer has already been initalized
        """
        
        optim_list = list(np.atleast_1d(optimizer))
        optim_args_list = list(np.atleast_1d(optimizer_args))
        if len(optim_list) == 1:
            optim_list = optim_list * len(self.cost_objs)
        if len(optim_args_list) == 1:
            optim_args_list = optim_args_list*len(self.cost_objs)
        
        if type(optimizer_args) == ut.NoneType:
            return optim_list
        else:
            return [opt(arg) for opt, arg in zip(optim_list, optim_args_list)]
             

    def _gen_sharing_matrix(self):
        """ 
        Generate the sharing tuples based on sharing mode
        """
        nb_optim = len(self.optim_list)
        if self.method == 'shared':
            return [(ii, jj, jj) for ii in range(nb_optim) for jj in range(nb_optim)]
        elif self.method == 'independent':
            #return [(ii, ii, ii) for ii in range(nb_optim)]
            return [(ii, ii, jj) for ii, opt in enumerate(self.optim_list) for jj in range(opt._nb_request)]
        elif self.method == 'left':
            tuples = []
            for consumer_idx in range(nb_optim):
                for generator_idx in range(nb_optim):
                    if consumer_idx >= generator_idx:
                        # higher indexed optims consume the evaluations generated by
                        # lower indexed optims
                        tuples.append((consumer_idx,generator_idx,generator_idx))
                    else:
                        # lower indexed optims generate extra 'padding' evaluations so
                        # that they recieve the same number of new data points
                        tuples.append((consumer_idx,consumer_idx,generator_idx))
            # sanity check
            assert len(tuples)==nb_optim*nb_optim
            return tuples
        elif self.method == 'right':
            tuples = []
            for consumer_idx in range(nb_optim):
                for generator_idx in range(nb_optim):
                    if consumer_idx <= generator_idx:
                        # higher indexed optims consume the evaluations generated by
                        # lower indexed optims
                        tuples.append((consumer_idx,generator_idx,generator_idx))
                    else:
                        # lower indexed optims generate extra 'padding' evaluations so
                        # that they recieve the same number of new data points
                        tuples.append((consumer_idx,consumer_idx,generator_idx))
            # sanity check
            assert len(tuples)==nb_optim*nb_optim
            return tuples


    def _gen_padding_params(self, x_new):
        """
        Different sharing modes e.g. 'left' and 'right' require padding
        of the evaluations requested by the optimisers with other random
        points, generate those circuits here
        """

        def _find_min_dist(a,b):
            """
            distance is euclidean distance, but since the values are angles we want to
            minimize the (element-wise) differences over optionally shifting one of the
            points by ±2\pi
            """
            a = np.array(a)
            b = np.array(b)
            disp_vector = np.minimum((a-b)**2,((a+2*np.pi)-b)**2)
            disp_vector = np.minimum(disp_vector,((a-2*np.pi)-b)**2)
            return np.sqrt(np.sum(disp_vector))

        x_new_mat = [[None for ii in range(len(self.cost_objs))] for jj in range(len(self.cost_objs))]
        for consumer_idx,requester_idx,pt_idx in self._sharing_matrix:
            # print(consumer_idx, requester_idx, pt_idx)
            # case where we need to generate a new evaluation
            if (consumer_idx==requester_idx) and (requester_idx==pt_idx):
                x_new_mat[requester_idx][pt_idx] = x_new[requester_idx][0]
            if (consumer_idx==requester_idx) and not (requester_idx==pt_idx):
                # get the points that the two optimsers indexed by
                # (`consumer_idx`==`requester_idx`) and `pt_idx` chose for their evals
                
                # Replaced by Kiran
                # generator_pt = self._parallel_x[requester_idx,requester_idx]
                # pt = self._parallel_x[pt_idx,pt_idx]
                generator_pt = x_new[requester_idx][0]
                pt = x_new[pt_idx][0]
                # separation between the points
                dist = _find_min_dist(generator_pt,pt)
                
                # generate random vector in N-d space then scale it to have length we want, 
                # using 'Hypersphere Point Picking' Gaussian approach
                random_displacement = np.random.normal(size=self.cost_objs[requester_idx].ansatz.nb_params)
                random_displacement = random_displacement * dist/np.sqrt(np.sum(random_displacement**2))
                # element-wise modulo 2\pi
                new_pt = np.mod(generator_pt+random_displacement,2*np.pi)
                x_new_mat[requester_idx][pt_idx] = new_pt
                
                # make new circuit now done in gen_circ_from_params
                # this_id = ut.gen_random_str(8)
                # named_circs = ut.prefix_to_names(self.cost_objs[requester_idx].meas_circuits, 
                #     this_id)
                # circs_to_exec += cost.bind_params(named_circs, new_pt, 
                #     self.cost_objs[requester_idx].ansatz.params)
                # self._parallel_id[requester_idx,pt_idx] = this_id
                # self._parallel_x[requester_idx,pt_idx] = new_pt
        return x_new_mat


    def _cross_evaluation(self, 
                          cst_eval_idx, 
                          optim_requester_idx, 
                          point_idx=None, 
                          results_obj=None):
        """ 
        Evaluate the results of an experiment allowing sharing of data 
        between the different internal optimisers

        Parameters
        ----------
        cst_eval_idx : int
            Index of the optim/cost function that we will evaluate the 
            point against
        optim_requester_idx : int
            Index of the optim that requested the point being considered
        point_idx : int, optional, defaults to optim_requester_idx
            Subindex of the point inside the set of points that optim 
            optim_requester_idx requested
        results_obj : Qiskit results obj, optional, defaults to last got
            The experiment results to use
        """
        if results_obj is None:
            results_obj = self._last_results_obj
        if point_idx is None:
            point_idx = optim_requester_idx
        circ_name = self._parallel_id[optim_requester_idx,point_idx]
        cost_obj = self.cost_objs[cst_eval_idx]
        x = self._parallel_x[optim_requester_idx,point_idx]
        y = cost_obj.evaluate_cost(results_obj, name = circ_name)
        #print(f'{cst_eval_idx}'+' '+f'{optim_requester_idx}'+' '+f'{point_idx}'+':'+f'{circ_name}')
        return x, y
    

    def _gen_circuits_from_params(self, x_new, inplace = False):
        """
        Creates measurement circuits from supplied parameter points, assumes input 
        is of the form x_new = [[p11,p12...], [p21, p22.,,], ...] where pij is the 
        j'th parameter point requested bit circuit i. Input structure need not be square
                
        Parameters:
        ---------------
        x_new: 3d itterable array 
            Nested list of parameter points assumed at least 3d, need not be square
            
        inplace: default False (maybe overkill)
            If true changes internal executable circus. If false, simply returns 
            executable circuits
        """
        circs_to_exec = []
        cost_list = self.cost_objs
        
        if inplace:
            self._last_x_new = x_new
            self._parallel_id = {}
            self._parallel_x = {}

        for cst_idx, (cst, points) in enumerate(zip(cost_list, x_new)):
            for pt_idx, pt in enumerate(points):
                if pt is not None:
                    label = ut.safe_string.gen(4)
                    circs_to_exec += cst.bind_params_to_meas(pt, label)
                    self._parallel_x[cst_idx,pt_idx] = pt
                    self._parallel_id[cst_idx,pt_idx] = label
            # idx_points = [ut.safe_string.gen(4) for _ in points]                    
            # circs_to_exec += cst.bind_params_to_meas(points, idx_points)
            # self._parallel_x.update({(cst_idx,pt_idx):pt for pt_idx, pt in enumerate(points) })
            # self._parallel_id.update({(cst_idx,pt_idx):idx for pt_idx, idx in enumerate(idx_points) })
        if inplace:
            self.circs_to_exec = circs_to_exec
        return circs_to_exec        


    def _results_from_last_x(self):
        """
        If specific points were requested, then this returns an array of the same 
        dimentions. WARNING THIS IGNORES ALL CROSS SHARING    
        """
        results = []
        for cst_idx,cst in enumerate(self.cost_objs):
            sub_results = []
            for pt in range(len(self._last_x_new[cst_idx])):
                sub_results.append(self._cross_evaluation(cst_idx,cst_idx,pt)[1])
            results.append(sub_results)
        return results
               
    
    def init_optimisers(self, results_obj = None): 
        """ 
        Take results object to initalise the internal optimisers 

        Parameters
        ----------
        results_obj : Qiskit results obj
            The experiment results to use
        """
        if self._evaluated_init:
            raise Exception("Optimizers have already been initialized")
        if results_obj == None:
            results_obj = self._last_results_obj
        self._last_results_obj = results_obj
        nb_optim = len(self.optim_list)
        nb_init = len(self._last_x_new[0])
        if self._share_init:
            sharing_matrix = [(cc,0,run) for cc in range(nb_optim) for run in range(nb_init)]
        else:
            sharing_matrix = [(cc,cc,run) for cc in range(nb_optim) for run in range(nb_init)]
        #print(sharing_matrix)
        self.update(results_obj, sharing_matrix)
        self._evaluated_init = True

    def next_evaluation_circuits(self):
        """ 
        Return the set of executable (i.e. transpiled and bound) quantum 
        circuits that will carry out cost function evaluations at the 
        points requested by each of the internal optimisers
        
        Parameters
        ----------
        x_new : list of x vals, optional
            An iterable with exactly 1 param point per cost function, if None
            is passed the function will query the internal optimisers
        """
        self._parallel_id = {}
        self._parallel_x = {}
        if self._evaluated_init:
            x_new = [opt.next_evaluation_params() for opt in self.optim_list]
            if 'SPSA' not in [o._type for o in self.optim_list]:
                x_new = self._gen_padding_params(x_new)
        else:
            if self._share_init:
                # if init data is shared only get requests from first optim
                x_new = [self.optim_list[0].next_evaluation_params()]
            else:
                x_new = [opt.next_evaluation_params() for opt in self.optim_list]
        circs_to_exec = self._gen_circuits_from_params(x_new, inplace = True)
        
        # sanity check on number of circuits generated
        # if self.method in ['independent','shared']:
        #     assert len(self._parallel_id.keys())==len(self.cost_objs),('Should have '
        #         +f'{len(self.cost_objs)}'+' circuits, but instead have '
        #         +f'{len(self._parallel_id.keys())}')
        # elif self.method in ['random1','random2']:
        #     assert len(self._parallel_id.keys())==len(self.cost_objs)**2,('Should have '
        #         +f'{len(self.cost_objs)**2}'+' circuits, but instead have '
        #         +f'{len(self._parallel_id.keys())}')
        # elif self.method in ['left','right']:
        #     assert len(self._parallel_id.keys())==len(self.cost_objs)*(len(self.cost_objs)+1)//2,('Should have '
        #         +f'{len(self.cost_objs)*(len(self.cost_objs)+1)//2}'
        #         +' circuits, but instead have '+f'{len(self._parallel_id.keys())}')

        return circs_to_exec
            
    
    def update(self, results_obj = None, sharing_matrix = None):
        """ 
        Update the internal state of the optimisers, currently specific
        to Bayesian optimisers
            
        Parameters
        ----------
        results_obj : Qiskit results obj
            The experiment results to use
        """
        if results_obj == None:
            results_obj = self._last_results_obj
        else:
            self._last_results_obj = results_obj
        if sharing_matrix == None:
            sharing_matrix = self._sharing_matrix
        for evl, req, par in sharing_matrix:
            x, y = self._cross_evaluation(evl, req, par)
            self.optim_list[evl].update(x, y)
    
    def shot_noise(self, x_new, nb_trials = 8):
        """
        Calculates shot noise for each circuit for a single input parameter
        
        TODO: Allow for a list of parameter point (one per circuit)
        
        Parameters:
        ---------
        x_new: 
            A single parameter point to calculate all cost functions
        
        nb_trials: 
            The number of time the cost function is evaluated for the given point
        """
        if hasattr(x_new[0], '__iter__'):
            x_new = np.array([x_new] * nb_trials).transpose((1,0,2))
        else:
            x_new = [x_new] * nb_trials
            x_new = [x_new] * len(self.cost_objs)
        self.circs_to_exec = self._gen_circuits_from_params(x_new, inplace = True)
        return x_new


    @property
    def prefix(self):
        """ Special name for each instance"""
        return self._prefix
        
def check_cost_objs_consistency(cost_objs):
    """
    Carry out some error checking on the Cost objs passed to the class
    constructor. Fix small fixable errors and crash for bigger errors.
    
    Parameters
    ----------
    cost_objs : list of cost objs 
        The cost objs passed to __init__

    Returns
    -------
    new_cost_objs : list of cost objs
        Possibly slightly altered list of cost objs
    """

    # TODO: Only makes sense to do this if the cost objs are based on 
    # the WeightedPauliOps class. Should check that and skip otherwise

    new_cost_objs = []
    for idx,op in enumerate(cost_objs):

        if idx == 0:
            test_pauli_set = set([ p[1] for p in op.paulis ])
            num_qubits = op.num_qubits  
        else:
            assert op.num_qubits==num_qubits, ("Cost operators passed to"
                +" do not all have the same number of qubits.")

            if not len(op.paulis)==len(test_pauli_set):
                # the new qubit op has a different number of Paulis than the previous
                new_pauli_set = set([ p[1] for p in op.paulis ])
                if len(op.paulis)>len(test_pauli_set):
                    # the new operator set has more paulis the previous
                    missing_paulis = list(new_pauli_set - test_pauli_set)
                    paulis_to_add = [ [op.atol*10,p] for p in missing_paulis ]
                    wpo_to_add = wpo(paulis_to_add)
                    # iterate over previous qubit ops and add new paulis
                    for prev_op in qubit_ops:
                        prev_op.add(wpo_to_add)
                    # save new reference pauli set
                    test_pauli_set = new_pauli_set
                else:
                    # the new operator set has less paulis than the previous
                    missing_paulis = list(test_pauli_set - new_pauli_set)
                    paulis_to_add = [ [op.atol*10,p] for p in missing_paulis ]
                    wpo_to_add = wpo(paulis_to_add)
                    # add new paulis to current qubit op
                    op.add(wpo_to_add)


        new_cost_objs.append(op)

    return new_cost_objs

# Run optimiser is common to all? Maybe use an intermediate class 
#   that impliments ParallelRunner and has method run_optimizer? 
#   Logic is a bitch to usage is pretty useful
class SingleBO(ParallelRunner):
    """ Creates single BO optimiser that interfaces properly with batch"""
    def __init__(self, 
                 cost_obj,
                 optimizer_args):            
        optimizer = MethodBO
        super().__init__([cost_obj],
                         optimizer = optimizer,
                         optimizer_args = optimizer_args)
    def run_optimizer(self, nb_iter):
        """
        Uses method._run_with_cost to run single evaluation
        """
        method = self.optim_list[0]
        method._run_with_cost(nb_iter = nb_iter, 
                              cost = self.cost_objs[0])
    @property
    def best_x(self):
        """ Returns the best guess for current optimum"""
        return self.optim_list[0]._best_x
    
class SingleSPSA(ParallelRunner):
    """ Creates single SPSA optimiser that interfaces properly with batch (Still buggy)"""
    def __init__(self, 
                 cost_obj,
                 optimizer_args):
        print("Warning: SingleSPSA not debugged with Batch")
        optimizer = MethodSPSA
        super().__init__([cost_obj],
                         optimizer = optimizer,
                         optimizer_args = optimizer_args)
    def run_optimizer(self, nb_iter):
        """
        Uses method._run_with_cost to run single evaluation
        """
        method = self.optim_list[0]
        method._run_with_cost(nb_iter = nb_iter, 
                              cost = self.cost_objs[0])
    @property
    def best_x(self):
        """ Returns the best guess for current optimum"""
        return self.optim_list[0]._best_x
    