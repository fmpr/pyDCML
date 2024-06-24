import numpy as np

class Expression():
    
    def __init__(self, expr):
        if type(expr) != list:
            raise Exception('Expression expects an argument of type list ' \
                            +'containing at least one object of type Variable.')
        if len(expr) == 0:
            raise Exception('Argument of type list cannot be empty. ' \
                            +'It should contain at least one object of type Variable.')
        self.expr = expr
        
        
    def __mul__(self, other):
        if issubclass(type(other), Variable):
            self.expr = [self.expr[-1] + [other]]
        else:
            self.expr = [self.expr + other.expr]
        return self
        
        
    def __add__(self, other):
        self.expr = self.expr + other.expr
        return self
    
    
    def __str__(self):
        s = ''
        for term in self.expr:
            if type(term) == list:
                for var in term:
                    if type(var) == TransformedVariable:
                        s += var.transform_name + '(' + var.name + ')*'
                    elif type(var) == RandomEffect:
                        s += var.name + '_n*'
                    else:
                        s += var.name + '*'
                        
                s = s[:-1]
                
            s += ' + '
        
        s = s[:-3]
                
        return s
            
    
class Variable():
    
    def __init__(self, name):
        self.name = name
    
    
    def __add__(self, other):
        if issubclass(type(other), Variable):
            other = Expression([other])
        elif type(other) != Expression:
            raise Exception('Can only do addition between objects of type Variable or Expression')
        return Expression([[self]]) + other
    
    
    def __mul__(self, other):
        if issubclass(type(other), Variable):
            other = Expression([other])
        elif type(other) != Expression:
            raise Exception('Can only do multiplication between objects of type Variable or Expression')
        return Expression([self]) * other

        
class ObservedVariable(Variable):
    
    def __init__(self, name):
        super().__init__(name)
    
    
    def __str__(self):
        return self.name+'[Observed]'
        
        
class TransformedVariable(ObservedVariable):
    
    def __init__(self, name, transform_name):
        self.transform_name = transform_name
        super().__init__(name)
    
    
    def __str__(self):
        return self.name+'[Transformed:'+self.transform_name+']'
        
        
class FixedEffect(Variable):
    
    def __init__(self, name):
        super().__init__(name)
    
    
    def __str__(self):
        return self.name+'[FixedEffect]'
        
        
class RandomEffect(Variable):
    
    def __init__(self, name):
        super().__init__(name)
    
    
    def __str__(self):
        return self.name+'[RandomEffect]'
    
    
class NeuralNetwork(Variable):
    
    def __init__(self, name):
        self.inputs = {}
        super().__init__(name)
        
        
    def __call__(self, alt, *args):
        self.alt = alt
        self.inputs[alt] = args
        return Expression([self])
    
    
    def __str__(self):
        return self.name


class Specification():
    
    _valid_model_types = ['MNL','MXL','ContextMXL','LCCM']
    
    def __init__(self, model_type, utilities, debug=False):
        if model_type not in self._valid_model_types:
            raise Exception('Argument model_type must be one of the following: '
                            +str(self._valid_model_types))
        self.model_type = model_type
        self.utilities = utilities
        self.alt_names = list(utilities.keys())
        self.num_alternatives = len(utilities)
        
        # extract important info from utilities
        self.columns_to_extract = []
        self.neural_nets = {}
        self.alt_id_map = []
        self.param_id_map = []
        self.random_var_to_param_id = {}
        self.mixed_param_names = []  # names of parameters to be treated with a Mixed Logit formulation
        self.mixed_param_ids = []    # IDs of parameters to be treated with a Mixed Logit formulation
        self.fixed_param_names = []
        self.fixed_param_ids = []
        next_param_id = 0
        next_alt_id = 0
        for alt in self.utilities:
            v = self.utilities[alt]
            for term in v.expr:
                if type(term) == NeuralNetwork:
                    if alt not in self.neural_nets:
                        self.neural_nets[alt] = []
                    self.neural_nets[alt].append(term)
                    continue
                    
                fixed_effects = []
                random_effects = []
                attributes = []
                for factor in term:
                    if type(factor) == FixedEffect or type(factor) == RandomEffect:
                        if factor not in self.random_var_to_param_id:
                            self.random_var_to_param_id[factor] = next_param_id
                            next_param_id += 1
                        self.alt_id_map += [next_alt_id]
                        self.param_id_map += [self.random_var_to_param_id[factor]]
                        
                        if type(factor) == FixedEffect:
                            fixed_effects.append(factor)
                            if factor.name not in self.fixed_param_names:
                                self.fixed_param_names.append(factor.name)
                                self.fixed_param_ids.append(self.random_var_to_param_id[factor])
                        elif type(factor) == RandomEffect:
                            random_effects.append(factor)
                            if factor.name not in self.mixed_param_names:
                                self.mixed_param_names.append(factor.name)
                                self.mixed_param_ids.append(self.random_var_to_param_id[factor])
                    else:
                        attributes.append(factor)
                        self.columns_to_extract += [factor]
                        
                if len(fixed_effects) + len(random_effects) > 1:
                    raise Exception("The following term contains more than one parameter: "+str(term))
            
            next_alt_id += 1
        
        self.alt_id_map = np.array(self.alt_id_map)
        self.param_id_map = np.array(self.param_id_map)
        self.param_id_map_by_alt = [self.param_id_map[np.where(self.alt_id_map == i)[0]] for i in range(self.num_alternatives)]
        self.mixed_param_ids = np.array(self.mixed_param_ids)
        self.fixed_param_ids = np.array(self.fixed_param_ids)
        self.num_params = next_param_id
        if debug: print('columns_to_extract:', self.columns_to_extract)
        if debug: print('alt_id_map:', self.alt_id_map)
        if debug: print('param_id_map:', self.param_id_map)
        if debug: print('random_var_to_param_id:', self.random_var_to_param_id)
        
        
    def __str__(self):
        s = '----------------- '+self.model_type+' specification:\n'
        s += 'Alternatives: '+str(self.alt_names)+'\n'
        s += 'Utility functions:\n'
        for alt in self.utilities:
            s += '   V_'+alt+' = '+str(self.utilities[alt])+'\n'
        
        s += '\nNum. parameters to be estimated: %d\n' % (self.num_params,)
        s += 'Fixed effects params: '+str(self.fixed_param_names)
        if self.model_type == 'MXL':
            s += '\nRandom effects params: '+str(self.mixed_param_names)
            
        return s
    
    
class SearchSpace(Specification):
    
    def __init__(self, model_type, search_spaces):
        self.search_spaces = search_spaces
        
        # build utility functions that correspond to search spaces provided
        utilities = {alt:[] for alt in search_spaces}
        for alt in search_spaces:
            utilities[alt] = Expression([term.expr[0] for term in search_spaces[alt]])
            
        super().__init__(model_type, utilities)
        
        
    def __str__(self):
        s = '----------------- '+self.model_type+' search space:\n'
        s += 'Alternatives: '+str(self.alt_names)+'\n'
        s += 'Terms considered for each utility function:\n'
        for alt in self.utilities:
            s += '   V_'+alt+': '+str([str(term) for term in self.search_spaces[alt]])+'\n'
        
        s += '\nTotal number of parameters in search space: %d\n' % (self.num_params,)
        if self.model_type == 'MXL':
            s += 'Random effects params: '+str(self.mixed_param_names)+'\n'
        s += 'Fixed effects params: '+str(self.fixed_param_names)
            
        return s
        

class Dataset():

    def __init__(self, dataframe, choice_col, dcm_spec, 
                 _format='wide', obs_id_col=None, alt_id_col=None, resp_id_col=None, context_cols=None, availability_cols=None):
        print("Preparing dataset...")
        
        self.df = dataframe
        self.dcm_spec = dcm_spec
        self.num_observations = len(dataframe)
        self.num_alternatives = dcm_spec.num_alternatives
        self.context_cols = context_cols
        
        if obs_id_col != None:
            self.obs_ids = dataframe[obs_id_col].values.astype(int)
        else:
            self.obs_ids = np.arange(self.num_observations).astype(int)
        if alt_id_col != None:
            self.alt_ids = dataframe[alt_id_col].values.astype(int)
        else:
            self.alt_ids = None
        if resp_id_col != None:
            self.resp_ids = dataframe[resp_id_col].values.astype(int)
            self.num_resp = len(np.unique(self.resp_ids))
            self.num_menus = max([np.sum(self.resp_ids == resp_id) for resp_id in np.unique(self.resp_ids)])
        else:
            self.resp_ids = np.arange(self.num_observations).astype(int)
            self.num_menus = 1
            self.num_resp = self.num_observations
        if availability_cols != None:
            self.availability_cols = []
            for alt in self.dcm_spec.alt_names:
                self.availability_cols += [availability_cols[alt]]
        else:
            self.availability_cols = None
        
        print("\tModel type:", str(self.dcm_spec.model_type))
        print("\tNum. observations:", self.num_observations)
        print("\tNum. alternatives:", self.num_alternatives)
        print("\tNum. respondents:", self.num_resp)
        print("\tNum. menus:", self.num_menus)
        print("\tObservations IDs:", self.obs_ids)
        print("\tAlternative IDs:", self.alt_ids)
        print("\tRespondent IDs:", self.resp_ids)
        print("\tAvailability columns:", self.availability_cols)
        
        self.nnet_attr_names = []
        self.attr_to_nnet_map = []
        self.nnet_attr_to_alt_map = []
        for alt in self.dcm_spec.neural_nets:
            for nnet in self.dcm_spec.neural_nets[alt]:
                self.nnet_attr_names.extend([obs_var.name for obs_var in nnet.inputs[alt]])
                self.attr_to_nnet_map.extend([nnet.name for i in range(len(nnet.inputs[alt]))])
                self.nnet_attr_to_alt_map.extend([dcm_spec.alt_names.index(alt) for i in range(len(nnet.inputs[alt]))])

        self.nnet_attr_names = np.array(self.nnet_attr_names)
        self.attr_to_nnet_map = np.array(self.attr_to_nnet_map)
        self.nnet_attr_to_alt_map = np.array(self.nnet_attr_to_alt_map)
        
        self.attr_names = []
        self.fixed_attr_names = []
        self.fixed_param_names = []
        self.mixed_attr_names = []
        self.mixed_param_names = []
        for alt in self.dcm_spec.utilities:
            v = self.dcm_spec.utilities[alt]
            for term in v.expr:
                if type(term) == NeuralNetwork:
                    continue
                    
                param_factor = None
                obs_vars = []
                for factor in term:
                    if type(factor) == FixedEffect or type(factor) == RandomEffect:
                        param_factor = factor
                    elif type(factor) == TransformedVariable:
                        if factor.transform_name == 'log':
                            self.df['log('+factor.name+')'] = np.log(self.df[factor.name])
                            factor.name = 'log('+factor.name+')'
                        else:
                            raise Exception('Not implemented transform:', factor.transform_name)
                        obs_vars.append(factor.name)
                    elif type(factor) == ObservedVariable:
                        obs_vars.append(factor.name)
                    else:
                        raise Exception('Unknown/unsupported type in utility function:', type(factor))
                
                obs_factor = None
                if len(obs_vars) > 1:
                    obs_factor = '_'.join(obs_vars)
                    self.df[obs_factor] = self.df[obs_vars[0]]
                    for i in range(1, len(obs_vars)):
                        self.df[obs_factor] = self.df[obs_factor] * self.df[obs_vars[i]]
                elif len(obs_vars) == 1:
                    obs_factor = obs_vars[0]
                else:
                    self.df['ONES'] = np.ones(len(self.df))
                    obs_factor = 'ONES'
                
                self.attr_names.append(obs_factor)
                        
                if type(param_factor) == FixedEffect:
                    if param_factor.name not in self.fixed_attr_names:
                        self.fixed_param_names.append(param_factor.name)
                        if obs_factor != None:
                            self.fixed_attr_names.append(obs_factor)
                elif type(param_factor) == RandomEffect:
                    if param_factor.name not in self.mixed_attr_names:
                        self.mixed_param_names.append(param_factor.name)
                        if obs_factor != None:
                            self.mixed_attr_names.append(obs_factor)
        
        self.num_attr = len(self.attr_names)
        self.num_fixed_attr = len(self.fixed_attr_names)
        self.num_mixed_attr = len(self.mixed_attr_names)
        
        print("\tAttribute names:", self.attr_names)
        print("\tFixed effects attribute names:", self.fixed_attr_names)
        print("\tFixed effects parameter names:", self.fixed_param_names)
        print("\tRandom effects attribute names:", self.mixed_attr_names)
        print("\tRandom effects parameter names:", self.mixed_param_names)
        
        if _format == 'long':
            raise Exception("Support for format 'long' is not implemented yet!")
        elif _format == 'wide':
            alt_availability = []
            alt_attributes = []
            true_choices = []
            mask = []
            context = []
            nnet_inputs = []
            
            for resp_id in np.unique(self.resp_ids):
                alt_availability.append([])
                alt_attributes.append([])
                true_choices.append([])
                mask.append([])
                context.append([])
                nnet_inputs.append([])

                t = 0
                for ix,row in self.df[self.resp_ids == resp_id].iterrows():
                    if self.availability_cols != None:
                        alt_availability[-1].append(row[self.availability_cols])
                    else:
                        alt_availability[-1].append(np.ones(self.num_alternatives))
                    alt_attributes[-1].append(row[self.attr_names])
                    true_choices[-1].append(row[choice_col])
                    mask[-1].append(1)
                    if self.context_cols != None:
                        context[-1].append(row[self.context_cols])
                    if len(self.nnet_attr_names) > 0:
                        nnet_inputs[-1].append(row[self.nnet_attr_names])

                    t += 1

                # pad tensor with zeros and set mask = 0 for remaining menus (t)
                while t < self.num_menus:
                    alt_availability[-1].append(np.zeros(self.num_alternatives))
                    alt_attributes[-1].append(np.zeros(len(self.attr_names)))
                    true_choices[-1].append(0)
                    mask[-1].append(0)
                    if self.context_cols != None:
                        context[-1].append(np.zeros(len(self.context_cols)))
                    if len(self.nnet_attr_names) > 0:
                        nnet_inputs[-1].append(np.zeros(len(self.nnet_attr_names)))

                    t += 1


            self.alt_attributes = np.array(alt_attributes)
            print('\tAlternative attributes ndarray.shape:', self.alt_attributes.shape) 
            self.true_choices = np.array(true_choices)
            print('\tChoices ndarray.shape:', self.true_choices.shape) 
            self.alt_availability = np.array(alt_availability)
            print('\tAlternatives availability ndarray.shape:', self.alt_availability.shape) 
            self.mask = np.array(mask)
            print('\tData mask ndarray.shape:', self.mask.shape)
            self.context = np.array(context)
            print('\tContext data ndarray.shape:', self.context.shape)
            self.nnet_inputs = np.array(nnet_inputs)
            print('\tNeural nets data ndarray.shape:', self.nnet_inputs.shape)
            
        else:
            raise Exception('Argument _format must be either "wide" or "long".')
            
        print("Done!")
        
        
    def __str__(self):
        s = '----------------- DCM dataset:\n'
        s += 'Model type: '+str(self.dcm_spec.model_type)+'\n'
        s += 'Num. observations: '+str(self.num_observations)+'\n'
        s += 'Num. alternatives: '+str(self.num_alternatives)+'\n'
        s += 'Num. respondents: '+str(self.num_resp)+'\n'
        s += 'Num. menus: '+str(self.num_menus)+'\n'
        s += 'Num. fixed effects: '+str(self.num_fixed_attr)+'\n'
        s += 'Num. random effects: '+str(self.num_mixed_attr)+'\n'
        s += 'Attribute names: '+str(self.attr_names)#+'\n'
        #s += 'Fixed effects attribute names: '+str(self.fixed_attr_names)+'\n'
        #s += 'Fixed effects parameter names: '+str(self.fixed_param_names)+'\n'
        #s += 'Random effects attribute names: '+str(self.mixed_attr_names)+'\n'
        #s += 'Random effects parameter names: '+str(self.mixed_param_names)
            
        return s

    