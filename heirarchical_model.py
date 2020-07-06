from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib
from sklearn.metrics import r2_score
import theano
import theano.tensor as T
import pymc3 as pm
from sklearn import preprocessing
import numpy as np

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, x, y=None):
        return self
    def transform(self, data_array):
        return data_array[:, self.columns]

class LabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.labeler = preprocessing.LabelEncoder()
        print(type(self.labeler))
    def fit(self,X, y=None, **fit_params):
        return self.labeler.fit(X)
    def transform(self,X, y='deprecated', **fit_params):
        return self.labeler.transform(X).reshape(-1,1)
    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)
    
class HLR(BaseEstimator):
    """
    Hierarchical linear regression model designed to work with pipeline
    """
    def __init__(self, step = 'NUTS'):
        self.cached_model = None
        self.shared_vars = None
        self.trace = None
        self.step = step
    def fit(self, X, y):
        """
        Train the HLR model
        
        Parameters
        X : numpy array, shape [n_samples, n_features]
        
        cat: numpy array, shape [n_samples, ]
        
        y : numpy array, shape [n_samples, ]
        """
        if ~(X[:,X.shape[1]-1] % 1 ==0).all():
            print('WARNING: Setting the category to column '+ str(X.shape[1]-1) + " despite % 1 of the column having non-zero values!")
        cat = X[:,-1].reshape(-1,1)
        cat = cat.astype('int32')
        self.num_cat = len(np.unique(cat))
        self.num_samples = X.shape[0]
        self.num_feat = X.shape[1]-1
        
        #Step 1: instantiate the model if not already defined
        if self.cached_model is None:
            self.create_model()
        
        #Step 2: Inference
        self.inference(X,y)
    def create_model(self):
        """
        Creates a pymc3 model and sets the model to self.cached_model
        
        Parameters
        --------------
        X: numpy array, shape [n_samples, n_features+1]
        NOTE - the last column of X is the category column
        """
        #initialize theano shared input variables
        model_input_x = [theano.shared(np.zeros([self.num_samples, 1])) for i in range(0,self.num_feat)]
        model_input_cat = theano.shared(np.zeros([self.num_samples, 1], dtype='int'))
        model_output = theano.shared(np.zeros([self.num_samples, 1]))
        self.shared_vars = {
            'model_input_x': model_input_x,
            'model_output': model_output,
            'model_input_cat': model_input_cat
        }
        
        model = pm.Model()
        with model:
            mu_beta = pm.Normal('mu_beta', mu=0., sd=0.5, shape=(self.num_feat))
            sigma_beta = pm.HalfCauchy('sigma_beta', beta=1, shape=(self.num_feat))
            
            alpha = pm.Normal('alpha', mu=0, sd=1, shape=(1))
            
            beta = [pm.Normal('beta_'+str(i), mu=mu_beta[i], sd=sigma_beta[i], shape=(self.num_cat)) for i in range(0,self.num_feat)]
            
            eps = pm.HalfCauchy('eps', beta=1)
            
            y_est = alpha + sum([beta_i[model_input_cat]*model_input_x[i] for i,beta_i in enumerate(beta)])
            
            y_like = pm.Normal('y_like', mu=y_est, sd=eps, observed=model_output)
            
        self.cached_model = model
    def inference(self,X,y):
        self.set_theano_shared(X,y)
        if self.step == 'NUTS':
            with self.cached_model:
                step = pm.NUTS()
                self.trace = pm.sample(3000, step, njobs=1)
        elif self.step == 'ADVI':
            with self.cached_model:
                self.approx = pm.fit(
                    n=40000,
                    method=pm.ADVI(),
                    )
            self.trace = self.approx.sample(7500)
    
    def ppc_predict(self,X):
        if self.trace is None:
            raise Exception('Run fit before trying to predict new data')
        
        self.set_theano_shared(X)
        
        ppc = pm.sample_ppc(
            self.trace[1000:], # specify the trace and exclude the first 1000 samples 
            model=self.cached_model, # specify the trained model
            samples=10000) #for each point in X_test, create 10000 samples
        return ppc
    def predict(self,X):
        """
        Predict using the linear hierarchical model
        
        Parameters
        --------------
        X: numpy array, shape [n_samples, n_features]
        cats: numpy array, shape [n_samples, 1]
        """
        
        pred = self.ppc_predict(X)['y_like'].mean(axis=0)
        std = self.ppc_predict(X)['y_like'].std(axis=0)
        return pred, std
    
    def set_theano_shared(self, X, y = None):
        if y is None:
            y = np.zeros([self.num_samples,1])
            
        if ~(X[:,X.shape[1]-1] % 1 ==0).all():
            print('WARNING: Setting the category to column '+ str(X.shape[1]-1) + " despite % 1 of the column having non-zero values!")
            
        cat = X[:,-1].reshape(-1,1)
        cat = cat.astype('int32')
        self.num_cat = len(np.unique(cat))
        self.num_samples = X.shape[0]
        self.num_feat = X.shape[1]-1
        
        [self.shared_vars['model_input_x'][i].set_value(X[:,i].reshape(-1,1)) for i in range(0,len(self.shared_vars['model_input_x']))]
        self.shared_vars['model_output'].set_value(y)
        self.shared_vars['model_input_cat'].set_value(cat)
        
    def score(self,X, y):
        """
        Returns the coefficient of determination R^2 of the prediction.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        cats: array-like, shape = (n_samples, 1)

        y : array-like, shape = (n_samples, 1)
        """
        return r2_score(y, self.predict(X)[0])
    def save(self, file_prefix):
        """
        Saves the trace, and param files with the given file_prefix.

        Parameters
        ----------
        file_prefix: str
        """
        fileObject = open(file_prefix + "trace.pickle", 'wb')
        joblib.dump(self.trace, fileObject)
        fileObject.close()

        fileObject = open(file_prefix + "params.pickle", 'wb')
        joblib.dump(
            {"num_cat": self.num_cat, "num_feat": self.num_feat},
            fileObject
        )
        fileObject.close()
    
    def load(self, file_prefix):
        """
        Loads a saved version of the advi_trace, v_params, and param files with the given file_prefix.

        Parameters
        ----------
        file_prefix: str
        """
        self.trace = joblib.load(file_prefix + "trace.pickle")

        params = joblib.load(file_prefix + "params.pickle")
        self.num_cat = params["num_cat"]
        self.num_pred = params["num_feat"]
        
    def plot_elbo(self):
        if self.step == 'ADVI':
            plt.plot(self.approx.hist)
            plt.yscale('log')
            #plt.xscale('log')
            plt.ylabel('ELBO')
            plt.xlabel('iteration');
        else:
            raise Exception("Can't plot elbo for " + self.step + ". This method only exist for ADVI.")
