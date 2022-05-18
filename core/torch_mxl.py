import torch
import torch.nn as nn
import torch.distributions as td
from torch.optim import Adam
import numpy as np
from core.lkj import LKJCholesky
import time
from matplotlib import pyplot as plt

class TorchMXL(nn.Module):
    def __init__(self, dcm_dataset, batch_size, use_cuda=True, use_inference_net=False):
        """
        Initializes the TorchMXL object.
        
        Parameters
        ----------
        dcm_dataset : Dataset
            The choice dataset.
        batch_size : int
            The size of each batch of observations used during variational inference.
        use_cuda : bool, optional
            Whether or not to use GPU-acceleration with CUDA (default is True).
        use_inference_net : bool, optional
            Whether or not to use an inference network for amortizing the cost of variational inference (default is False).
        """
        super(TorchMXL, self).__init__()
        
        self.dcm_dataset = dcm_dataset
        self.dcm_spec = dcm_dataset.dcm_spec
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.use_inference_net = use_inference_net
        
        self.num_observations = dcm_dataset.num_observations
        self.num_alternatives = dcm_dataset.num_alternatives
        self.num_resp = dcm_dataset.num_resp
        self.num_menus = dcm_dataset.num_menus
        self.num_params = dcm_dataset.dcm_spec.num_params
        self.num_fixed_params = len(dcm_dataset.dcm_spec.fixed_param_names)
        self.num_mixed_params = len(dcm_dataset.dcm_spec.mixed_param_names)
        self.alt_attributes = dcm_dataset.alt_attributes
        self.choices = dcm_dataset.true_choices
        self.alt_availability = dcm_dataset.alt_availability
        self.mask = dcm_dataset.mask
        self.context = dcm_dataset.context
        
        self.device = torch.device("cuda:0" if use_cuda and torch.cuda.is_available() else "cpu")
        
        # prepare data for running inference
        self.train_x = torch.tensor(self.alt_attributes, dtype=torch.float)#.to(self.device)
        self.context_info = torch.tensor(self.context, dtype=torch.float)#.to(self.device)
        self.train_y = torch.tensor(self.choices, dtype=torch.int)#.to(self.device)
        self.alt_av = torch.from_numpy(self.alt_availability)#.to(self.device)
        self.alt_av_mat = self.alt_availability.copy()
        self.alt_av_mat[np.where(self.alt_av_mat == 0)] = -1e9
        self.alt_av_mat -= 1
        self.alt_av_mat_cuda = torch.from_numpy(self.alt_av_mat).float()#.to(self.device)
        self.zeros_mat = torch.zeros(self.num_menus, self.batch_size, self.num_alternatives).to(self.device) # auxiliary matrix for model
        self.alt_ids_cuda = torch.from_numpy(self.dcm_spec.alt_id_map[:,np.newaxis].repeat(self.num_menus*self.num_resp,1).T.reshape(self.num_resp,self.num_menus,-1)).to(self.device)
        self.mask_cuda = torch.tensor(self.mask, dtype=torch.bool)#.to(self.device)
        
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # initialize variational distribution
        self.initialize_variational_distribution_q()
        
        
    def initialize_variational_distribution_q(self,):
        # q(alpha) - initialize mean and lower-cholesky factor of the covariance matrix
        self.alpha_mu = nn.Parameter(torch.zeros(self.num_fixed_params))
        self.alpha_cov_diag = nn.Parameter(torch.ones(self.num_fixed_params))
        self.alpha_cov_offdiag = nn.Parameter(torch.zeros(int((self.num_fixed_params*(self.num_fixed_params-1))/2)))
        self.tril_indices_alpha = torch.tril_indices(row=self.num_fixed_params, col=self.num_fixed_params, offset=-1)
        
        # q(zeta) - initialize mean and lower-cholesky factor of the covariance matrix
        self.zeta_mu = nn.Parameter(torch.zeros(self.num_mixed_params))
        self.zeta_cov_diag = nn.Parameter(torch.ones(self.num_mixed_params))
        self.zeta_cov_offdiag = nn.Parameter(torch.zeros(int((self.num_mixed_params*(self.num_mixed_params-1))/2)))
        self.tril_indices_zeta = torch.tril_indices(row=self.num_mixed_params, col=self.num_mixed_params, offset=-1)

        # q(Omega) - initialize means and variances of the diagonal and off-diagonal elements
        #            of the lower-cholesky factor of the covariance matrix 
        self.L_omega_diag_mu = nn.Parameter(torch.ones(self.num_mixed_params))
        self.L_omega_diag_sigma = nn.Parameter(torch.zeros(self.num_mixed_params))
        self.L_omega_offdiag_mu = nn.Parameter(torch.ones(int((self.num_mixed_params*(self.num_mixed_params-1))/2)))
        self.L_omega_offdiag_sigma = nn.Parameter(torch.zeros(int((self.num_mixed_params*(self.num_mixed_params-1))/2)))

        # q(beta_n) - initialize mean and lower-cholesky factor of the covariance matrix
        self.beta_mu = nn.Parameter(torch.zeros(self.num_resp, self.num_mixed_params))
        self.beta_cov_diag = nn.Parameter(torch.ones(self.num_mixed_params))
        self.beta_cov_offdiag = nn.Parameter(torch.zeros(int((self.num_mixed_params*(self.num_mixed_params-1))/2)))

        if self.dcm_spec.model_type == 'ContextMXL':
            # layers of neural net for context data
            self.context_hidden_dim = 10
            self.context_fc1 = nn.Linear(self.context.shape[-1], self.context_hidden_dim)
            self.context_bn1 = nn.BatchNorm1d(self.context_hidden_dim)
            self.context_fc2 = nn.Linear(self.context_hidden_dim, self.num_mixed_params + self.num_fixed_params)
            self.context_dropout = nn.Dropout(0.5)

        if self.use_inference_net:
            # layers of inference neural network for amortization
            self.kernel_size = self.num_params*self.num_alternatives+self.num_alternatives*2
            self.infnet_hidden_dim = 200
            self.cnn1 = torch.nn.Conv1d(1, self.infnet_hidden_dim, kernel_size=(self.kernel_size), stride=(self.kernel_size),
                                        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
            self.bn1 = nn.BatchNorm1d(1)
            self.bn2 = nn.BatchNorm1d(self.infnet_hidden_dim)
            self.fc1 = nn.Linear(self.infnet_hidden_dim, self.infnet_hidden_dim)
            self.fc2mu = nn.Linear(self.infnet_hidden_dim, self.num_mixed_params)
            self.dropout = nn.Dropout(0.5)
            self.pooling = nn.MaxPool1d(int(self.num_menus), stride=(int(self.num_menus)))
        
                
    def compute_variational_approximation_q(self, alt_attr, context_attr, obs_choices, alt_avail, alt_ids):
        """
        Computes the variational approximation q(z) to the true posterior distribution of the model p(z|x), where z denotes the latent variables in the model (e.g., the fixed and random effect parameters) and x denotes the observed data (e.g., alternative attributes and observed choices). When the inference network is used to amortize the cost of variational inference, then this method passes the observations through the inference neural network in order to obtain an approximation of the posterior q(beta_n).
        
        Parameters
        ----------
        alt_attr : Torch.tensor
            Torch tensor of shape (batch_size, num_menus, num_alternatives*(num_fixed_attr+num_mixed_attr)) containing the attributes for the different alternatives.
        context_attr : Torch.tensor
            Torch tensor of shape (batch_size, num_menus, num_context_attributes) containing the attributes descrbing the context for the different choice situations.
        obs_choices : Torch.tensor
            Torch tensor of shape (batch_size, num_menus) containing the observed choices (represented as integers in the set {0, ..., num_alternatives-1}).
        alt_avail : Torch.tensor
            Torch tensor of shape (batch_size, num_menus, num_alternatives) containing information about the availability of the different alternatives (represented as 0 or 1).
        alt_ids : Torch.tensor
            Torch tensor of shape (batch_size, num_menus, num_alternatives*(num_fixed_attr+num_mixed_attr)) mapping the attributes in alt_attr to the different alternatives (represented as integers in the set {0, ..., num_alternatives-1}).
        
        Returns
        ----------
        q_alpha : Torch.distribution
            Torch distribution with the posterior approximation over the global fixed effects preference parameters q(alpha)
        q_zeta : Torch.distribution.
            Torch distribution with the posterior approximation over the global mixed effects preference parameters q(zeta)
        q_L_Omega_diag : Torch.distribution.
            Torch distribution with the posterior approximation over the diagonal elements of the lower-Cholesky factorization of the covariance matrix q(Omega).
        q_L_Omega_offdiag : Torch.distribution
            Torch distribution with the posterior approximation over the off-diagonal elements of the lower-Cholesky factorization of the covariance matrix q(Omega).
        q_beta : Torch.distribution
            Torch distribution with the posterior approximation over the (local) per-respondent preference parameters q(beta_n) for each respondent n.
        """
        # q(alpha) - construct posterior approximation on alpha
        alpha_cov_tril = torch.zeros((self.num_fixed_params, self.num_fixed_params), device=self.device)
        alpha_cov_tril[self.tril_indices_alpha[0], self.tril_indices_alpha[1]] = self.alpha_cov_offdiag
        alpha_cov_tril += torch.diag_embed(self.softplus(self.alpha_cov_diag))
        q_alpha = td.MultivariateNormal(self.alpha_mu, scale_tril=torch.tril(alpha_cov_tril))
        
        # q(zeta) - construct posterior approximation on zeta
        zeta_cov_tril = torch.zeros((self.num_mixed_params, self.num_mixed_params), device=self.device)
        zeta_cov_tril[self.tril_indices_zeta[0], self.tril_indices_zeta[1]] = self.zeta_cov_offdiag
        zeta_cov_tril += torch.diag_embed(self.softplus(self.zeta_cov_diag))
        q_zeta = td.MultivariateNormal(self.zeta_mu, scale_tril=torch.tril(zeta_cov_tril))
        
        # q(Omega) - construct posterior approximation on Omega using multiple independent Gaussians
        q_L_Omega_diag = td.Normal(self.softplus(self.L_omega_diag_mu), self.softplus(self.L_omega_diag_sigma))
        q_L_Omega_offdiag = td.Normal(self.L_omega_offdiag_mu, self.softplus(self.L_omega_offdiag_sigma))
        
        # q(beta_n) - construct posterior approximation on beta_n
        beta_cov_tril = torch.zeros((self.num_mixed_params, self.num_mixed_params), device=self.device)
        beta_cov_tril[self.tril_indices_zeta[0], self.tril_indices_zeta[1]] = self.beta_cov_offdiag
        beta_cov_tril += torch.diag_embed(self.softplus(self.beta_cov_diag))
        if self.use_inference_net:
            # prepare input data for inference neural network
            one_hot = torch.zeros(self.num_resp, self.num_menus, self.num_alternatives, device=self.device, dtype=torch.float)
            one_hot = one_hot.scatter(2, obs_choices.unsqueeze(2).long(), 1)
            inference_data = torch.cat([one_hot, alt_attr, alt_avail.float()], dim=-1)
            inference_data = inference_data.flatten(1,2).unsqueeze(1)
            
            # compute the hidden units
            hidden = self.bn1(inference_data)
            hidden = self.cnn1(hidden)
            hidden = self.relu(self.pooling(hidden))
            hidden = self.bn2(hidden)
            hidden = self.relu(self.fc1(hidden.flatten(1,2)))
            mu_loc = self.fc2mu(hidden)
            q_beta = td.MultivariateNormal(mu_loc, scale_tril=torch.tril(beta_cov_tril))
        else:
            q_beta = td.MultivariateNormal(self.beta_mu, scale_tril=torch.tril(beta_cov_tril))
        
        if self.dcm_spec.model_type == 'ContextMXL':
            # pass context data through the context neural net
            hidden = self.relu(self.context_fc1(context_attr))
            hidden = self.context_dropout(hidden)
            beta_offsets = self.context_fc2(hidden)
            
            return q_alpha, q_zeta, q_L_Omega_diag, q_L_Omega_offdiag, q_beta, beta_offsets
        
        return q_alpha, q_zeta, q_L_Omega_diag, q_L_Omega_offdiag, q_beta, None
        
        
    def elbo(self, alt_attr, context_attr, obs_choices, alt_avail, obs_mask, alt_ids, indices):
        """
        Computes the stochastic approximation to the evidence lower bound (ELBO) used by variational inference to optimize the parameters of the variational approximation q(z) to the true posterior distribution p(z|x).
        
        Parameters
        ----------
        alt_attr : Torch.tensor
            Torch tensor of shape (batch_size, num_menus, num_alternatives*(num_fixed_attr+num_mixed_attr)) containing the attributes for the different alternatives.
        context_attr : Torch.tensor
            Torch tensor of shape (batch_size, num_menus, num_context_attributes) containing the attributes descrbing the context for the different choice situations.
        obs_choices : Torch.tensor
            Torch tensor of shape (batch_size, num_menus) containing the observed choices (represented as integers in the set {0, ..., num_alternatives-1}).
        alt_avail : Torch.tensor
            Torch tensor of shape (batch_size, num_menus, num_alternatives) containing information about the availability of the different alternatives (represented as 0 or 1).
        obs_mask : Torch.tensor
            Torch tensor of shape (batch_size, num_menus) describing which menus in alt_attr and obs_choices are to be considered (represented as 0 or 1) - this is useful for panel data where different respondents have different numbers of choice situations.
        alt_ids : Torch.tensor
            Torch tensor of shape (batch_size, num_menus, num_alternatives*(num_fixed_attr+num_mixed_attr)) mapping the attributes in alt_attr to the different alternatives (represented as integers in the set {0, ..., num_alternatives-1}).
        
        Returns
        ----------
        elbo : Torch.tensor
            Value of the ELBO based on the current variational distribution q(z).
        """
        
        # ----- get posterior approximations -----
        q_alpha, q_zeta, q_L_Omega_diag, q_L_Omega_offdiag, q_beta, beta_offsets = self.compute_variational_approximation_q(alt_attr, context_attr, obs_choices, alt_avail, alt_ids)
        
        # ----- sample from posterior approximations -----
        alpha = q_alpha.rsample()
        zeta = q_zeta.rsample()
        beta = q_beta.rsample()
        L_Omega_diag = q_L_Omega_diag.rsample()
        L_Omega_offdiag = q_L_Omega_offdiag.rsample()
        L_Omega = torch.zeros((self.num_mixed_params, self.num_mixed_params), device=self.device)
        L_Omega[self.tril_indices_zeta[0], self.tril_indices_zeta[1]] = L_Omega_offdiag
        L_Omega += torch.diag_embed(self.softplus(L_Omega_diag))
        
        # ----- gather paramters for computing the utilities -----
        beta_resp = self.gather_parameters_for_MNL_kernel(alpha, beta[indices])
        
        # ----- compute utilities -----
        utilities = self.compute_utilities(beta_resp, alt_attr, alt_avail, alt_ids)

        # ----- (expected) log-likelihood -----
        loglik = td.Categorical(logits=utilities).log_prob(obs_choices.transpose(0,1))
        loglik = torch.where(obs_mask.T, loglik, loglik.new_zeros(())) # use mask to filter out missing menus
        loglik = loglik.sum()
        
        # ----- define priors -----
        alpha_prior = td.MultivariateNormal(torch.zeros(self.num_fixed_params, device=self.device), 
                                           scale_tril=torch.tril(1*torch.eye(self.num_fixed_params, device=self.device)))
        
        zeta_prior = td.MultivariateNormal(torch.zeros(self.num_mixed_params, device=self.device), 
                                           scale_tril=torch.tril(1*torch.eye(self.num_mixed_params, device=self.device)))
        
        beta_prior = td.MultivariateNormal(zeta, scale_tril=L_Omega)

        # vector of variances for each of the d variables - used in LKJ prior
        theta_prior = td.HalfCauchy(1*torch.ones(self.num_mixed_params, device=self.device))

        # lower cholesky factor of a correlation matrix
        eta = 1*torch.ones(1, device=self.device)  # implies a uniform distribution over correlation matrices
        L_Sigma_prior = LKJCholesky(self.num_mixed_params, eta)

        # decompose L_Sigma into L_Omega and theta for scoring w.r.t. priors
        Omega = torch.mm(L_Omega, L_Omega.T) 
        theta = torch.diag(Omega) 
        theta_sqrt = theta.sqrt()
        L_Sigma = torch.mul(L_Omega / torch.outer(theta_sqrt, theta_sqrt), theta_sqrt)
        
        # ----- compute KL-divergence terms -----
        kld = 0.
        
        # KL[q(alpha) || p(alpha)]
        kld += td.kl_divergence(q_alpha, alpha_prior)
            
        # KL[q(zeta) || p(zeta)]
        kld += td.kl_divergence(q_zeta, zeta_prior)
        
        # KL[q(beta_n) || p(beta_n)]
        kld += td.kl_divergence(q_beta, beta_prior).sum()
        
        # KL[q(Omega) || p(Omega)]
        kld += q_L_Omega_diag.log_prob(L_Omega_diag).sum() + q_L_Omega_offdiag.log_prob(L_Omega_offdiag).sum() 
        kld += -L_Sigma_prior.log_prob(L_Sigma).sum() - theta_prior.log_prob(theta).sum()
        
        # ----- compute ELBO -----
        # ELBO = -E[loglik] + KL[q || prior]
        elbo = -loglik + kld
        
        # compute accuracy based on utilities
        acc = utilities.argmax(-1) == obs_choices.transpose(0,1)
        acc = torch.where(obs_mask.T, acc, acc.new_zeros(()))
        acc = acc.sum() / obs_mask.sum()
        
        # remember values (e.g. to show progress)
        self.loglik = loglik
        self.kld = kld
        self.acc = acc
        
        return elbo
    
    
    def infer(self, num_epochs=10000, true_alpha=None, true_beta=None, true_beta_resp=None):
        """
        Performs variational inference (amortized variational inference if use_inference_net is set to True). 
        
        Parameters
        ----------
        num_epochs : int, optional
            Number of passes/iterations through the dataset to be performed during ELBO maximization (default is 10000).
        true_alpha : np.array, optional
            Numpy array with true values of the global fixed-effect preference parameters for comparison (useful for investigating the progress of variational inference in cases when the true values of the preference parameters are known). If provided, then this method outputs additional information during ELBO maximization.
        true_beta : np.array, optional
            Numpy array with true values of the global random-effect preference parameters for comparison (useful for investigating the progress of variational inference in cases when the true values of the preference parameters are known). If provided, then this method outputs additional information during ELBO maximization.
        true_beta_resp : np.array, optional
            Numpy array with true values of the per-respondent preference parameters for comparison (useful for investigating the progress of variational inference in cases when the true values of the preference parameters are known). If provided, then this method outputs additional information during ELBO maximization.
        
        Returns
        ----------
        results : dict
            Python dictionary containing the results of variational inference. 
        """
        self.to(self.device)
        
        #print("Initial ELBO value: %.1f" % self.loss(self.train_x, self.context_info, self.train_y, self.alt_av_mat_cuda, self.mask_cuda, self.alt_ids_cuda).item())

        optimizer = Adam(self.parameters(), lr=1e-2)

        self.train() # enable training mode
        
        tic = time.time()
        alpha_errors = []
        beta_errors = []
        betaInd_errors = []
        for epoch in range(num_epochs):
            permutation = torch.randperm(self.num_resp)
            
            for i in range(0, self.num_resp, self.batch_size):
                
                indices = permutation[i:i+self.batch_size]
                batch_x, batch_context, batch_y = self.train_x[indices], self.context_info[indices], self.train_y[indices]
                batch_alt_av_mat, batch_mask_cuda, batch_alt_ids = self.alt_av_mat_cuda[indices], self.mask_cuda[indices], self.alt_ids_cuda[indices]
                
                batch_x = batch_x.to(self.device)
                batch_context = batch_context.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_alt_av_mat = batch_alt_av_mat.to(self.device)
                batch_mask_cuda = batch_mask_cuda.to(self.device)
            
                optimizer.zero_grad()

                elbo = self.elbo(batch_x, batch_context, batch_y, batch_alt_av_mat, batch_mask_cuda, batch_alt_ids, indices)

                elbo.backward()
                optimizer.step()
                
                if not epoch % 100:
                    msg = "[Epoch %5d] ELBO: %.0f; Loglik: %.0f; Acc.: %.3f" % (epoch, elbo.item(), self.loglik, self.acc)
                    if np.all(true_alpha != None):
                        alpha_rmse = np.sqrt(np.mean((true_alpha - self.alpha_mu.detach().cpu().numpy())**2)) if len(true_alpha) > 0 else np.inf
                        alpha_errors += [alpha_rmse]
                        msg += "; Alpha RMSE: %.3f" % (alpha_rmse,)
                    if self.dcm_spec.model_type != 'MNL' and np.all(true_beta != None):
                        beta_rmse = np.sqrt(np.mean((true_beta - self.zeta_mu.detach().cpu().numpy())**2)) if len(true_beta) > 0 else np.inf
                        beta_errors += [beta_rmse]
                        msg += "; Beta RMSE: %.3f" % (beta_rmse,)
                    if self.dcm_spec.model_type != 'MNL' and np.all(true_beta_resp != None):
                        params_resps_rmse = np.sqrt(np.mean((true_beta_resp - self.beta_mu.detach().cpu().numpy())**2)) if len(true_beta_resp) > 0 else np.inf
                        betaInd_errors += [params_resps_rmse]
                        msg += "; BetaInd RMSE: %.3f" % (params_resps_rmse,)

                    print(msg)

        toc = time.time() - tic
        print('Elapsed time:', toc, '\n')
            
        # prepare python dictionary of results to output
        results = {}
        results["Estimation time"] = toc
        results["Est. alpha"] = self.alpha_mu.detach().cpu().numpy()
        if self.dcm_spec.model_type != 'MNL':
            results["Est. zeta"] = self.zeta_mu.detach().cpu().numpy()
            if self.use_inference_net: 
                q_alpha, q_zeta, q_L_Sigma_diag, q_L_Sigma_offdiag, q_beta, beta_offsets = self.compute_variational_approximation_q(self.train_x.to(self.device), self.context_info.to(self.device), self.train_y.to(self.device), self.alt_av_mat_cuda.to(self.device), self.alt_ids_cuda.to(self.device))
                results["Est. beta_n"] = q_beta.loc.detach().cpu().numpy()
            else:
                results["Est. beta_n"] = self.beta_mu.detach().cpu().numpy()
        results["ELBO"] = elbo.item()
        results["Loglikelihood"] = self.loglik.item()
        results["Accuracy"] = self.acc.item()
        
        # show quick summary of results
        if np.all(true_alpha != None): print("True alpha:", true_alpha)
        print("Est. alpha:", self.alpha_mu.detach().cpu().numpy())
        
        for i in range(len(self.dcm_spec.fixed_param_names)):
            print("\t%s: %.3f" % (self.dcm_spec.fixed_param_names[i], results["Est. alpha"][i]))
        print()

        if self.dcm_spec.model_type != 'MNL':
            if np.all(true_beta != None): print("True zeta:", true_beta)
            print("Est. zeta:", self.zeta_mu.detach().cpu().numpy())
        
            for i in range(len(self.dcm_spec.mixed_param_names)):
                print("\t%s: %.3f" % (self.dcm_spec.mixed_param_names[i], results["Est. zeta"][i]))
            print()

        if np.all(true_alpha != None) or np.all(true_beta != None) or np.all(true_beta_resp != None):
            if np.all(true_alpha != None): plt.plot(alpha_errors)
            if np.all(true_beta != None): plt.plot(beta_errors)
            if np.all(true_beta_resp != None): plt.plot(betaInd_errors)
            plt.legend(['alpha rmse','beta_rmse','beta_resps_rmse'])
            plt.show();
        
        return results
        
        
    def gather_parameters_for_MNL_kernel(self, alpha, beta):
        next_fixed = 0
        next_mixed = 0
        reordered_pars = []
        alpha_resp = alpha.repeat(self.batch_size,1)
        for par_id in range(self.num_params):
            if par_id in self.dcm_spec.fixed_param_ids:
                reordered_pars.append(alpha_resp[:,next_fixed].unsqueeze(-1))
                next_fixed += 1
            elif par_id in self.dcm_spec.mixed_param_ids:
                reordered_pars.append(beta[:,next_mixed].unsqueeze(-1))
                next_mixed += 1
            else:
                raise Exception("This should not happen!!")
        reordered_pars = torch.cat(reordered_pars, dim=-1)
        beta_resp = torch.cat([reordered_pars[:,self.dcm_spec.param_id_map_by_alt[i]] for i in range(self.num_alternatives)], dim=-1)
        return beta_resp
    
    
    def compute_utilities(self, beta_resp, alt_attr, alt_avail, alt_ids):
        # compute utilities for each alternative
        utilities = torch.scatter_add(self.zeros_mat,
                                      2, 
                                      alt_ids.transpose(0,1), 
                                      torch.mul(alt_attr.transpose(0,1), beta_resp))
    
        # adjust utility for unavailable alternatives
        utilities += alt_avail.transpose(0,1)
        
        return utilities
    
    