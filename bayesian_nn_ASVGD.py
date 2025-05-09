import theano.tensor as T
import theano
import numpy as np
from scipy.spatial.distance import pdist, squareform
import random
import time
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

'''
    Sample code to reproduce our results
    for the Bayesian neural network example.
    Our settings are almost the same as Hernandez-Lobato and Adams (ICML15)
    https://jmhldotorg.files.wordpress.com/2015/05/pbp-icml2015.pdf
    Our implementation is also based on their Python code.

    p(y | W, X, \gamma) = \prod_i^N  N(y_i | f(x_i; W), \gamma^{-1})
    p(W | \lambda) = \prod_i N(w_i | 0, \lambda^{-1})
    p(\gamma) = Gamma(\gamma | a0, b0)
    p(\lambda) = Gamma(\lambda | a0, b0)

    The posterior distribution is as follows:
    p(W, \gamma, \lambda)
    = p(y | W, X, \gamma) p(W | \lambda) p(\gamma) p(\lambda)
    To avoid negative values of \gamma and \lambda,
    we update loggamma and loglambda instead.

    Build closely (by Viktor Stein) on the code by Qiang Liu & Dilin Wang
'''


class svgd_bayesnn:

    '''
        We define a one-hidden-layer-neural-network specifically. We leave extension of deep neural network as our future work.
        
        Input
            -- X_train: training dataset, features
            -- y_train: training labels
            -- batch_size: sub-sampling batch size
            -- max_iter: maximum iterations for the training procedure
            -- M: number of particles are used to fit the posterior distribution
            -- n_hidden: number of hidden units
            -- a0, b0: hyper-parameters of Gamma distribution
            -- master_stepsize, auto_corr: parameters of adgrad
    '''
    def __init__(self, X_train, y_train,  batch_size = 100, max_iter = 1000, M = 20, n_hidden = 50, a0 = 1, b0 = 0.1, master_stepsize = 1e-3, auto_corr = 0.9):
        self.n_hidden = n_hidden
        self.d = X_train.shape[1]   # number of data, dimension 
        self.M = M
        
        num_vars = self.d * n_hidden + n_hidden * 2 + 3  # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 2 variances
        self.theta = np.zeros([self.M, num_vars])  # particles, will be initialized later
        
        '''
            We keep the last 10% (maximum 500) of training data points for model developing
        '''
        size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
        X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
        X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]

        '''
            The data sets are normalized so that the input features and the targets have zero mean and unit variance
        '''
        self.std_X_train = np.std(X_train, 0)
        self.std_X_train[ self.std_X_train == 0 ] = 1
        self.mean_X_train = np.mean(X_train, 0)
        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)
        
        '''
            Theano symbolic variables
            Define the neural network here
        '''
        X = T.matrix('X') # Feature matrix
        y = T.vector('y') # labels
        
        w_1 = T.matrix('w_1') # weights between input layer and hidden layer
        b_1 = T.vector('b_1') # bias vector of hidden layer
        w_2 = T.vector('w_2') # weights between hidden layer and output layer
        b_2 = T.scalar('b_2') # bias of output
        
        N = T.scalar('N') # number of observations
        
        log_gamma = T.scalar('log_gamma')   # variances related parameters
        log_lambda = T.scalar('log_lambda')
        
        ###
        prediction = T.dot(T.nnet.relu(T.dot(X, w_1)+b_1), w_2) + b_2
        
        ''' define the log posterior distribution '''
        log_lik_data = -0.5 * X.shape[0] * (T.log(2*np.pi) - log_gamma) - (T.exp(log_gamma)/2) * T.sum(T.power(prediction - y, 2))
        log_prior_data = (a0 - 1) * log_gamma - b0 * T.exp(log_gamma) + log_gamma
        log_prior_w = -0.5 * (num_vars-2) * (T.log(2*np.pi)-log_lambda) - (T.exp(log_lambda)/2)*((w_1**2).sum() + (w_2**2).sum() + (b_1**2).sum() + b_2**2)  \
                       + (a0-1) * log_lambda - b0 * T.exp(log_lambda) + log_lambda
        
        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        log_posterior = (log_lik_data * N / X.shape[0] + log_prior_data + log_prior_w)
        dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda = T.grad(log_posterior, [w_1, b_1, w_2, b_2, log_gamma, log_lambda])
        
        # automatic gradient
        logp_gradient = theano.function(
             inputs = [X, y, w_1, b_1, w_2, b_2, log_gamma, log_lambda, N],
             outputs = [dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda]
        )

        # prediction function
        self.nn_predict = theano.function(inputs = [X, w_1, b_1, w_2, b_2], outputs = prediction)

        '''
            Training with SVGD
        '''
        # normalization
        X_train, y_train = self.normalization(X_train, y_train)
        N0 = X_train.shape[0]  # number of observations

        ''' initializing all particles '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.init_weights(a0, b0)
            # use better initialization for gamma
            ridx = np.random.choice(range(X_train.shape[0]),
                                    np.min([X_train.shape[0], 1000]),
                                    replace=False)
            y_hat = self.nn_predict(X_train[ridx,:], w1, b1, w2, b2)
            loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
            self.theta[i, :] = self.pack_weights(w1, b1, w2, b2, loggamma, loglambda)

        grad_theta = np.zeros([self.M, num_vars])  # gradient
        # adagrad with momentum
        fudge_factor = 1e-6
        historical_grad = 0
        k1 = random.randint(0, 19)
        k2 = random.randint(0, 19)
        plot = False
        for iter in range(max_iter):
            # sub-sampling
            batch = [i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size)]
            for i in range(self.M):
                w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
                dw1, db1, dw2, db2, dloggamma, dloglambda = logp_gradient(X_train[batch, :], y_train[batch], w1, b1, w2, b2, loggamma, loglambda, N0)
                grad_theta[i, :] = self.pack_weights(dw1, db1, dw2, db2, dloggamma, dloglambda)

            # calculating the kernel matrix
            kxy, dxkxy = self.svgd_kernel(h=-1)  
            grad_theta = (np.matmul(kxy, grad_theta) + dxkxy) / self.M   # \Phi(x)

            # adagrad 
            if iter == 0:
                historical_grad += np.multiply(grad_theta, grad_theta)
            else:
                historical_grad = auto_corr * historical_grad + (1 - auto_corr) * np.multiply(grad_theta, grad_theta)
                adj_grad = np.divide(grad_theta, fudge_factor+np.sqrt(historical_grad))
                self.theta += master_stepsize * adj_grad 
            # plot two dimensions
            if plot and not iter % 50:
                plt.figure(figsize=(6, 6))
                plt.scatter(self.theta[k1, :], self.theta[k2, :], s=10, alpha=0.7)
                plt.title(f"Iteration {iter}, dimensions = {k1},{k2}")
                plt.axis('equal')   # ensure x and y scales are the same
                plt.xlim([-2, 2])
                plt.ylim([-2, 2])
                plt.grid(True)
                plt.show()
        '''
            Model selection by using a development set
        '''
        X_dev = self.normalization(X_dev) 
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
            pred_y_dev = self.nn_predict(X_dev, w1, b1, w2, b2) * self.std_y_train + self.mean_y_train
            # likelihood
            def f_log_lik(loggamma): return np.sum(  np.log(np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) * np.exp( -1 * (np.power(pred_y_dev - y_dev, 2) / 2) * np.exp(loggamma) )) )
            # The higher probability is better    
            lik1 = f_log_lik(loggamma)
            # one heuristic setting
            loggamma = -np.log(np.mean(np.power(pred_y_dev - y_dev, 2)))
            lik2 = f_log_lik(loggamma)
            if lik2 > lik1:
                self.theta[i,-2] = loggamma  # update loggamma


    def normalization(self, X, y = None):
        X = (X - np.full(X.shape, self.mean_X_train)) / \
            np.full(X.shape, self.std_X_train)
            
        if y is not None:
            y = (y - self.mean_y_train) / self.std_y_train
            return (X, y)  
        else:
            return X
    
    '''
        Initialize all particles
    '''
    def init_weights(self, a0, b0):
        w1 = 1.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, self.n_hidden)
        b1 = np.zeros((self.n_hidden,))
        w2 = 1.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden)
        b2 = 0.
        loggamma = np.log(np.random.gamma(a0, b0))
        loglambda = np.log(np.random.gamma(a0, b0))
        return (w1, b1, w2, b2, loggamma, loglambda)
    
    '''
        Calculate kernel matrix and its gradient: K, \nabla_x k
    ''' 
    def svgd_kernel(self, h = -1):
        sq_dist = pdist(self.theta)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0: # if h < 0, using median trick
            h = np.median(pairwise_dists)  
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))

        # compute the rbf kernel
        
        Kxy = np.exp( -pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:,i] + np.multiply(self.theta[:,i],sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy)
    
    
    '''
        Pack all parameters in our model
    '''    
    def pack_weights(self, w1, b1, w2, b2, loggamma, loglambda):
        params = np.concatenate([w1.flatten(), b1, w2, [b2], [loggamma],[loglambda]])
        return params
    
    '''
        Unpack all parameters in our model
    '''
    def unpack_weights(self, z):
        w = z
        w1 = np.reshape(w[:self.d*self.n_hidden], [self.d, self.n_hidden])
        b1 = w[self.d*self.n_hidden:(self.d+1)*self.n_hidden]
    
        w = w[(self.d+1)*self.n_hidden:]
        w2, b2 = w[:self.n_hidden], w[-3] 
        
        # the last two parameters are log variance
        loggamma, loglambda= w[-2], w[-1]
        
        return (w1, b1, w2, b2, loggamma, loglambda)

    
    '''
        Evaluating testing rmse and log-likelihood, which is the same as in PBP 
        Input:
            -- X_test: unnormalized testing feature set
            -- y_test: unnormalized testing labels
    '''
    def evaluation(self, X_test, y_test):
        # normalization
        X_test = self.normalization(X_test)

        # average over the output
        pred_y_test = np.zeros([self.M, len(y_test)])
        prob = np.zeros([self.M, len(y_test)])

        '''
            Since we have M particles, we use a Bayesian view to calculate rmse and log-likelihood
        '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
            pred_y_test[i, :] = self.nn_predict(X_test, w1, b1, w2, b2) * self.std_y_train + self.mean_y_train
            prob[i, :] = np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) * np.exp( -1 * (np.power(pred_y_test[i, :] - y_test, 2) / 2) * np.exp(loggamma) )
        pred = np.mean(pred_y_test, axis=0)

        # evaluation
        svgd_rmse = np.sqrt(np.mean((pred - y_test)**2))
        svgd_ll = np.mean(np.log(np.mean(prob, axis = 0)))

        return (svgd_rmse, svgd_ll)




class asvgd_bayesnn:

    '''
        We define a one-hidden-layer-neural-network specifically.
        We leave extension of deep neural network as our future work.

        Input
            -- X_train: training dataset, features
            -- y_train: training labels
            -- batch_size: sub-sampling batch size
            -- max_iter: maximum iterations for the training procedure
            -- M: number of particles are used to fit the posterior distribution
            -- n_hidden: number of hidden units
            -- a0, b0: hyper-parameters of Gamma distribution
            -- master_stepsize, auto_corr: parameters of adgrad
    '''

    def __init__(self, X_train, y_train,  batch_size=100, max_iter=1000,
                 M=20, n_hidden=50, a0=1, b0=0.1, master_stepsize=1e-6,
                 auto_corr=0.9,
                 eps=.1,  # Wasserstein regularization
                 beta=None,  # constant damping parameter
                 gradient_restart=True,
                 speed_restart=True):
        self.n_hidden = n_hidden
        self.d = X_train.shape[1]   # number of data, dimension
        self.M = M

        # w1: d*n_hidden; b1: n_hidden; w2 = n_hidden; b2 = 1; 2 variances
        num_vars = self.d * n_hidden + n_hidden * 2 + 3
        # particles, will be initialized later
        self.theta = np.zeros([self.M, num_vars])

        '''
            We keep the last 10% (maximum 500) of training data points
            for model developing
        '''
        size_dev = min(int(np.round(0.1 * X_train.shape[0])), 500)
        X_dev, y_dev = X_train[-size_dev:], y_train[-size_dev:]
        X_train, y_train = X_train[:-size_dev], y_train[:-size_dev]

        '''
            The data sets are normalized so that the input features and
            the targets have zero mean and unit variance
        '''
        self.std_X_train = np.std(X_train, 0)
        self.std_X_train[self.std_X_train == 0] = 1
        self.mean_X_train = np.mean(X_train, 0)
        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        '''
            Theano symbolic variables
            Define the neural network here
        '''
        X = T.matrix('X')  # Feature matrix
        y = T.vector('y')  # labels

        w_1 = T.matrix('w_1')  # weights between input layer and hidden layer
        b_1 = T.vector('b_1')  # bias vector of hidden layer
        w_2 = T.vector('w_2')  # weights between hidden layer and output layer
        b_2 = T.scalar('b_2')  # bias of output

        N = T.scalar('N')  # number of observations

        log_gamma = T.scalar('log_gamma')   # variances related parameters
        log_lambda = T.scalar('log_lambda')

        # evaluate the two-layer NN
        prediction = T.dot(T.nnet.relu(T.dot(X, w_1)+b_1), w_2) + b_2

        ''' define the log posterior distribution '''
        log_lik_data = -0.5 * X.shape[0] * (T.log(2*np.pi) - log_gamma) \
                       - (T.exp(log_gamma)/2) * T.sum(T.power(prediction - y, 2))
        log_prior_data = (a0 - 1) * log_gamma - b0 * T.exp(log_gamma) + log_gamma
        log_prior_w = -0.5 * (num_vars-2) * (T.log(2*np.pi)-log_lambda) \
                      - (T.exp(log_lambda)/2)*((w_1**2).sum() + (w_2**2).sum()
                                               + (b_1**2).sum() + b_2**2)  \
                       + (a0-1) * log_lambda - b0 * T.exp(log_lambda) \
                       + log_lambda

        # sub-sampling mini-batches of data, where (X, y) is the batch data, and N is the number of whole observations
        log_posterior = (log_lik_data * N / X.shape[0] + log_prior_data + log_prior_w)
        dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda = T.grad(log_posterior, [w_1, b_1, w_2, b_2, log_gamma, log_lambda])

        # automatic gradient
        logp_gradient = theano.function(
             inputs=[X, y, w_1, b_1, w_2, b_2, log_gamma, log_lambda, N],
             outputs=[dw_1, db_1, dw_2, db_2, d_log_gamma, d_log_lambda]
        )

        # prediction function
        self.nn_predict = theano.function(inputs=[X, w_1, b_1, w_2, b_2],
                                          outputs=prediction)

        '''
            Training with ASVGD
        '''
        # normalization
        X_train, y_train = self.normalization(X_train, y_train)
        N0 = X_train.shape[0]  # number of observations

        ''' initializing all particles '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.init_weights(a0, b0)
            # use better initialization for gamma
            ridx = np.random.choice(range(X_train.shape[0]),
                                    np.min([X_train.shape[0], 1000]),
                                    replace=False)
            y_hat = self.nn_predict(X_train[ridx, :], w1, b1, w2, b2)
            loggamma = -np.log(np.mean(np.power(y_hat - y_train[ridx], 2)))
            self.theta[i, :] = self.pack_weights(w1, b1, w2, b2,
                                                 loggamma, loglambda)

        grad_theta = np.zeros([self.M, num_vars])  # gradient 
        # adagrad with momentum
        adagrad = True
        print(f'Adagrad = {adagrad}')
        fudge_factor = 1e-6
        historical_grad = 0

        # New ASVGD stuff
        Y = np.zeros((self.M, num_vars))  # initial acceleration
        V = np.zeros((self.M, num_vars))  # initial velocities
        # restart_counter: for computing acceleration parameter for each particle
        # will be reset to 1 when a restart is triggered.
        restart_counter = np.ones(self.M)  # each particle starts with counter=1
        alpha_k = np.zeros(self.M)  # acceleration parameter for each particle
        thetas = np.zeros((max_iter, self.M, num_vars))
        #
        k1 = random.randint(0, 19)
        k2 = random.randint(0, 19)
        plot = False
        for iter in range(max_iter):
            # sub-sampling
            batch = [i % N0 for i in range(iter * batch_size, (iter + 1) * batch_size)]
            for i in range(self.M):
                w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
                dw1, db1, dw2, db2, dloggamma, dloglambda = logp_gradient(X_train[batch,:], y_train[batch], w1, b1, w2, b2, loggamma, loglambda, N0)
                grad_theta[i, :] = self.pack_weights(dw1, db1, dw2, db2, dloggamma, dloglambda)

            # calculating the kernel matrix
            kxy, dxkxy, sigma = self.svgd_kernel(h=-1)

            # ASVGD
            thetas[iter, :, :] = self.theta
            V = self.M * np.linalg.solve(kxy + self.M * eps * np.eye(self.M), Y)
            # adaptive speed restart
            if speed_restart and iter > 1:
                norm_diff_current = np.linalg.norm(self.theta - thetas[iter-1, :, :], axis=1)  # ||x_{k+1}^i - x_k^i||
                norm_diff_prev = np.linalg.norm(thetas[iter-1, :, :] - thetas[iter-2, :, :], axis=1)  # ||x_k^i - x_{k-1}^i||
                # For each particle, check if current step < previous
                mask = norm_diff_current < norm_diff_prev
                # If yes, restart for the particle; else, increment its counter
                restart_counter[mask] = 0
                restart_counter[~mask] += 1
                alpha_k = (restart_counter) / (restart_counter + 3)
            elif beta is not None:
                alpha_k = beta*np.ones(self.M)  # constant damping
            else:
                alpha_k = iter / (iter + 3) * np.ones(self.M)  # no restart
            if gradient_restart:
                B = grad_theta + self.theta  # (N x d)
                # First term: sum_{i,j} K[i,j] * (f(X_i) + X_i) dot V_j
                term1 = np.sum((B.dot(V.T)) * kxy)
                # Second term: sum_{i,j} K[i,j] * (X_j dot V_j).
                term2 = np.sum(np.sum(kxy, axis=0) * np.sum(self.theta * V, axis=1))
                phi = term1 - term2
                if phi < 0:
                    restart_counter = np.ones(self.M)
            Y = alpha_k[:, None] * Y
            W = kxy @ np.multiply(V @ V.T, kxy) - np.multiply(kxy@V@V.T, kxy) + self.M * kxy
            W_laplacian = np.diag(W.sum(axis=1)) - W
            Y += np.sqrt(master_stepsize) / self.M * grad_theta
            Y += np.sqrt(master_stepsize) / (2 * self.M**2 * sigma**2) * W_laplacian @ self.theta

            # adagrad
            if adagrad:
                if iter == 0:
                    historical_grad = historical_grad + np.multiply(Y, Y)
                else:
                    historical_grad = auto_corr * historical_grad + (1 - auto_corr) * np.multiply(Y, Y)
                    adj_grad = np.divide(Y, fudge_factor+np.sqrt(historical_grad))
                    self.theta += np.sqrt(master_stepsize) * adj_grad
            else:
                self.theta += np.sqrt(master_stepsize) * Y

            # plot two dimensions
            if plot and not iter % 50:
                plt.figure(figsize=(6, 6))
                plt.scatter(self.theta[k1, :], self.theta[k2, :], s=10, alpha=0.7)
                plt.title(f"Iteration {iter}, dimensions = {k1},{k2}")
                plt.axis('equal')   # ensure x and y scales are the same
                plt.xlim([-2., 2.])
                plt.ylim([-2., 2.])
                plt.grid(True)
                plt.show()
            # early stopping
            if np.linalg.norm(grad_theta) < self.M * 1e-3:
                print('Early stopping triggered by ASVGD')
                break
        

        '''
            Model selection by using a development set
        '''
        X_dev = self.normalization(X_dev)
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
            pred_y_dev = self.nn_predict(X_dev, w1, b1, w2, b2) * self.std_y_train + self.mean_y_train
            # likelihood
            def f_log_lik(loggamma): return np.sum(np.log(np.sqrt(np.exp(loggamma)) /np.sqrt(2*np.pi) * np.exp( -1 * (np.power(pred_y_dev - y_dev, 2) / 2) * np.exp(loggamma) )) )
            # The higher probability is better
            lik1 = f_log_lik(loggamma)
            # one heuristic setting
            loggamma = -np.log(np.mean(np.power(pred_y_dev - y_dev, 2)))
            lik2 = f_log_lik(loggamma)
            if lik2 > lik1:
                self.theta[i, -2] = loggamma  # update loggamma

    def normalization(self, X, y = None):
        X = (X - np.full(X.shape, self.mean_X_train)) / \
            np.full(X.shape, self.std_X_train)

        if y is not None:
            y = (y - self.mean_y_train) / self.std_y_train
            return (X, y)
        else:
            return X

    def init_weights(self, a0, b0):
        '''
            Initialize all particles
        '''
        w1 = 1.0 / np.sqrt(self.d + 1) * np.random.randn(self.d, self.n_hidden)
        b1 = np.zeros((self.n_hidden,))
        w2 = 1.0 / np.sqrt(self.n_hidden + 1) * np.random.randn(self.n_hidden)
        b2 = 0.
        loggamma = np.log(np.random.gamma(a0, b0))
        loglambda = np.log(np.random.gamma(a0, b0))
        return (w1, b1, w2, b2, loggamma, loglambda)

    def svgd_kernel(self, h=-1):
        '''
            Calculate kernel matrix and its gradient: K, \nabla_x k
        '''
        sq_dist = pdist(self.theta)
        pairwise_dists = squareform(sq_dist)**2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(self.theta.shape[0]+1))

        # compute the rbf kernel

        Kxy = np.exp(-pairwise_dists / h**2 / 2)

        dxkxy = -np.matmul(Kxy, self.theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(self.theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(self.theta[:, i], sumkxy)
        dxkxy = dxkxy / (h**2)
        return (Kxy, dxkxy, h)

    def pack_weights(self, w1, b1, w2, b2, loggamma, loglambda):
        '''
            Pack all parameters in our model
        '''
        params = np.concatenate([w1.flatten(), b1, w2,
                                 [b2], [loggamma], [loglambda]])
        return params

    def unpack_weights(self, z):
        '''
            Unpack all parameters in our model
        '''
        w = z
        w1 = np.reshape(w[:self.d*self.n_hidden], [self.d, self.n_hidden])
        b1 = w[self.d*self.n_hidden:(self.d+1)*self.n_hidden]

        w = w[(self.d+1)*self.n_hidden:]
        w2, b2 = w[:self.n_hidden], w[-3]

        # the last two parameters are log variance
        loggamma, loglambda = w[-2], w[-1]

        return (w1, b1, w2, b2, loggamma, loglambda)

    def evaluation(self, X_test, y_test):
        '''
            Evaluating testing rmse and log-likelihood,
            which is the same as in PBP
            Input:
                -- X_test: unnormalized testing feature set
                -- y_test: unnormalized testing labels
        '''
        # normalization
        X_test = self.normalization(X_test)

        # average over the output
        pred_y_test = np.zeros([self.M, len(y_test)])
        prob = np.zeros([self.M, len(y_test)])

        '''
            Since we have M particles, we use a Bayesian view to calculate
            rmse and log-likelihood
        '''
        for i in range(self.M):
            w1, b1, w2, b2, loggamma, loglambda = self.unpack_weights(self.theta[i, :])
            pred_y_test[i, :] = self.nn_predict(X_test, w1, b1, w2, b2)
            * self.std_y_train + self.mean_y_train
            prob[i, :] = np.sqrt(np.exp(loggamma)) / np.sqrt(2*np.pi)
            *np.exp(-1 * (np.power(pred_y_test[i, :] - y_test, 2) / 2)
                    * np.exp(loggamma))
        pred = np.mean(pred_y_test, axis=0)

        # evaluation
        asvgd_rmse = np.sqrt(np.mean((pred - y_test)**2))
        asvgd_ll = np.mean(np.log(np.mean(prob, axis=0)))

        return (asvgd_rmse, asvgd_ll)


if __name__ == '__main__':
    datasets = ['wine', 'concrete', 'energy', 'housing', 'kin8nm', 'naval',
                'power', 'protein', 'yacht']
    for dataset in datasets:
        np.random.seed(1)
        random.seed(1)

        # load dataset
        path = 'C:\\Users\\vglom\\Documents\\TU Berlin\\SHK Steidl\\!Wuchen\\' \
               + 'Accelerated gradient flows\\Code\\Code that works\\data\\'
        data = np.loadtxt(f"{path}{dataset}.txt")
        X_input = data[:, :-1]
        y_input = data[:, -1]

        # build train/test split
        train_ratio = 0.9
        permutation = np.arange(X_input.shape[0])
        random.shuffle(permutation)
        size_train = int(np.round(len(permutation) * train_ratio))
        idx_train, idx_test = permutation[:size_train], permutation[size_train:]
        X_train, y_train = X_input[idx_train], y_input[idx_train]
        X_test, y_test = X_input[idx_test], y_input[idx_test]

        # parameters
        batch_size, n_hidden = 100, 50
        maxiters = [10, 25, 50, 100, 200, 300, 400, 500, 1000, 1500, 2000, 5000]
        n_runs = 20

        # initialize result containers: runs x iters x metrics
        results_svgd = np.zeros((n_runs, len(maxiters), 3))
        results_asvgd = np.zeros((n_runs, len(maxiters), 3))

        for run in tqdm(range(n_runs)):
            np.random.seed(run)
            random.seed(run)
            for k, iters in enumerate(maxiters):
                # standard SVGD
                start = time.time()
                svgd = svgd_bayesnn(X_train, y_train,
                                    batch_size=batch_size,
                                    n_hidden=n_hidden,
                                    max_iter=iters)
                svg_time = time.time() - start
                svg_rmse, svg_ll = svgd.evaluation(X_test, y_test)
                results_svgd[run, k, :] = [svg_rmse, svg_ll, svg_time]

                # accelerated SVGD (ASVGD)
                start = time.time()
                asvgd = asvgd_bayesnn(X_train, y_train,
                                      batch_size=batch_size,
                                      n_hidden=n_hidden,
                                      max_iter=iters)
                asvgd_time = time.time() - start
                asg_rmse, asg_ll = asvgd.evaluation(X_test, y_test)
                results_asvgd[run, k, :] = [asg_rmse, asg_ll, asvgd_time]

        # save raw arrays
        np.save(f'results_SVGD_{dataset}.npy', results_svgd)
        np.save(f'results_ASVGD_{dataset}.npy', results_asvgd)

        # compute statistics
        stats = {}
        for name, arr in [('SVGD', results_svgd), ('ASVGD', results_asvgd)]:
            means = np.mean(arr, axis=0)
            stds = np.std(arr, axis=0)
            stats[name] = {'mean': means, 'std': stds}

        # print tables
        for name in ['SVGD', 'ASVGD']:
            df_mean = pd.DataFrame(stats[name]['mean'], index=maxiters,
                                   columns=[f'{name}_RMSE', f'{name}_LL',
                                            f'{name}_time']).round(3)
            df_std = pd.DataFrame(stats[name]['std'],  index=maxiters,
                                  columns=[f'{name}_RMSE', f'{name}_LL',
                                           f'{name}_time']).round(3)
            print(f"{name} Mean results over {n_runs} runs:")
            print(df_mean)
            print(f"\n{name} Std. dev. over {n_runs} runs:")
            print(df_std)
            print("\n" + "="*60 + "\n")

        # plotting comparison with error bars for each metric
        metrics = ['RMSE', 'LL', 'time']
        ylabels = ['RMSE', 'Test Log-Likelihood', 'Time (seconds)']
        titles = [f'RMSE averaged over {n_runs} runs, {dataset}',
                  f'Log-Likelihood averaged over {n_runs} runs, {dataset}',
                  f'run time averaged over {n_runs} runs, {dataset}']

        for i, metric in enumerate(metrics):
            plt.errorbar(maxiters,
                         stats['SVGD']['mean'][:, i],
                         yerr=stats['SVGD']['std'][:, i],
                         marker='o', linestyle='-', label='SVGD')
            plt.errorbar(maxiters,
                         stats['ASVGD']['mean'][:, i],
                         yerr=stats['ASVGD']['std'][:, i],
                         marker='s', linestyle='--', label='ASVGD')
            plt.xlabel('iterations')
            plt.ylabel(ylabels[i])
            plt.legend()
            plt.title(titles[i])
            plt.savefig(f'Comparison_{metric}_{dataset}.png',
                        dpi=300, bbox_inches='tight')
            plt.show()
