import pandas as pd
import numpy as np
import time
import warnings
import inspect
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from collections import Counter
from pandas.api.types import is_float_dtype, is_integer_dtype


class iclltuning:
    """This is the class representation of this library. Invoke the methods of this class to perform hyperparameter tuning on the models fed to the Imbalanced Classification via Layered Learning (ICLL) model.

    **Layer 1** follows the same definition as mentioned within the original paper: "Layer 1 (L1) denotes the event that the observation belongs to the Mixed or Pure Minority Group."
    
    **Layer 2** follows the same definition as mentioned within the original paper: "Given that L1 occurs (i.e. observation belongs to the mixed or pure minority group), Layer 2 (L2) denotes the event that the observation belongs to the pure minority class."
    
    :param model_l1: ML model to be fit on **Layer 1**. A model must be passed if you don't want hyperparameter tuning to be performed on **Layer 1** task. 

        |
        | **Requirements:**

        1. This model must have a callable :py:meth:`!fit` method.
        2. This model must have a callable :py:meth:`!predict` in the case of hard classification or :py:meth:`!predict_proba` in the case of soft classification. Either one provided is fine.
        
        |
        | In the situation where both the model and cross-validation object are both provided, the model assigned to the cross-validation object will take precedence in the ICLL algorithm.
    :type model_l1: object, **optional**

    :param model_l2: ML model to be fit on **Layer 2**. A model must be passed if you don't want hyperparameter tuning to be performed on **Layer 2** task. 
    
        |
        | **Requirements:**

        1. This model must have a callable :py:meth:`!fit` method.
        2. This model must have a callable :py:meth:`!predict` in the case of hard classification or :py:meth:`!predict_proba` in the case of soft classification. Either one provided is fine.
        
        |
        | In the situation where both the model and cross-validation object are both provided, the model assigned to the cross-validation object will take precedence in the ICLL algorithm.
    :type model_l2: object, **optional**

    :param cv_obj_l1: :py:obj:`GridSearchCV` or :py:obj:`RandomizedSearchCV` object to be fit on **Layer 1**.

        .. note::

            :py:attr:`refit` parameter of the :py:obj:`GridSearchCV` or :py:obj:`RandomizedSearchCV` object **must NOT be initialized with a value of False**. If it is instantiated a value of False, then the **ICLL model will NOT be trained**, and a :py:exc:`ValueError` will be raised instead.

    :type cv_obj_l1: :py:obj:`GridSearchCV` or :py:obj:`RandomizedSearchCV` object, **optional**

    :param cv_obj_l2: :py:obj:`GridSearchCV` or :py:obj:`RandomizedSearchCV` object to be fit on **Layer 2**.

        .. note::

            :py:attr:`refit` parameter of the :py:obj:`GridSearchCV` or :py:obj:`RandomizedSearchCV` object **must be NOT be initialized with a value of False**. If it is instantiated a value of False, then the **ICLL model will NOT be trained**, and a :py:exc:`ValueError` will be raised instead.

    :type cv_obj_l2: :py:obj:`GridSearchCV` or :py:obj:`RandomizedSearchCV` object, **optional**

    :param smote_obj_l1: SMOTE object to be passed for resampling on **Layer 1**.
    :type smote_obj_l1: :py:class:`imblearn.over_sampling._smote.base.SMOTE`, **optional**

    :param smote_obj_l2: SMOTE object to be passed for resampling on **Layer 2**.
    :type smote_obj_l2: :py:class:`imblearn.over_sampling._smote.base.SMOTE`, **optional**

    :param sample_wt_l1: Array of sample weights to be assigned for **Layer 1** training task. 
        
        .. note::

            1. If :py:attr:`cv_obj_l1` parameter is passed, then hyperparameter tuning will be carried out with sample weights passed, and as a result, the best estimator as found out by :py:attr:`cv_obj_l1` will be trained with the provided sample weights. Note that both the :py:attr:`cv_obj_l1` object and its attached estimator must both have the :py:attr:`sample_weight` attribute for the :py:meth:`!fit` method. **Otherwise, hyperparameter tuning will be carried out without supplying the sample weights.**
            2. If :py:attr:`cv_obj_l1` parameter is not passed, then :py:attr:`model_l1` parameter will be trained with sample weights of :py:attr:`sample_wt_l1`. Note that the estimator must have the :py:attr:`sample_weight` attribute for the :py:meth:`!fit` method. **Otherwise, estimator training will be carried out wihtout supplying the sample weights.**
            3. If both parameters :py:attr:`smote_obj_l1` and :py:attr:`sample_wt_l1` are passed, then resampling will **NOT be performed on Layer 1**, and sample weights will be passed during model training/ hyperparameter tuning.

    :type sample_wt_l1: numpy.ndarray/ List / Pandas Series, **optional**

    :param sample_wt_l2: Array of sample weights to be assigned for **Layer 2** training task. 

        .. note::

            1. If :py:attr:`cv_obj_l2` parameter is passed, then hyperparameter tuning will be carried out with sample weights passed, and as a result, the best estimator as found out by :py:attr:`cv_obj_l2` will be trained with the provided sample weights. Note that both the :py:attr:`cv_obj_l2` object and its attached estimator must both have the :py:attr:`sample_weight` attribute for the :py:meth:`!fit` method. **Otherwise, hyperparameter tuning will be carried out wihtout supplying the sample weights.**
            2. If :py:attr:`cv_obj_l2` parameter is not passed, then :py:attr:`model_l2` parameter will be trained with sample weights of :py:attr:`sample_wt_l2`. Note that the estimator must have the :py:attr:`sample_weight` attribute for the :py:meth:`!fit` method. **Otherwise, estimator training will be carried out wihtout supplying the sample weights.**
            3. If both parameters :py:attr:`smote_obj_l2` and :py:attr:`sample_wt_l2` are passed, then resampling will **NOT be performed on Layer 2**, and sample weights will be passed during model training/ hyperparameter tuning.

            |
            | **IMPORTANT**

            In ICLL algorithm, **Layer 2** is trained to distinguish instances belonging to Mixed clusters from those belonging to Pure Minority clusters. Instances assigned to Pure Majority clusters are excluded from the training process in this layer. Consequently, only the sample weights corresponding to instances in Mixed or Pure Minority clusters are passed to the estimator or the :py:obj:`GridSearchCV` or :py:obj:`RandomizedSearchCV` object in **Layer 2**.

            **WARNING**: Only use this parameter if you fully understand its effect on the training process.

    :type sample_wt_l2: numpy.ndarray/ List / Pandas Series, **optional**

    :param custom_distance_threshold: Provide a value for this parameter if you want to override the original implementation of the computation of distance threshold (flat clusters are obtained at a threshold of the log-transformed intra-cluster distance of **µ + σ**)
    :type custom_distance_threshold: float, **optional**

    """
    def __init__(self,
                 model_l1=None,
                 model_l2=None,
                 cv_obj_l1=None,
                 cv_obj_l2=None,
                 smote_obj_l1=None,
                 smote_obj_l2=None,
                 sample_wt_l1=None,
                 sample_wt_l2=None,
                 custom_distance_threshold=None
                 ):
        """Constructor of the class iclltuning
        """
        self.__model_l1 = model_l1
        self.__model_l2 = model_l2
        
        self.__cv_obj_l1 = cv_obj_l1
        self.__cv_obj_l2 = cv_obj_l2

        self.__smote_obj_l1 = smote_obj_l1
        self.__smote_obj_l2 = smote_obj_l2

        self.__sample_wt_l1 = sample_wt_l1
        self.__sample_wt_l2 = sample_wt_l2

        self.__custom_distance_threshold = custom_distance_threshold

        # Initialize model variables
        self.__final_layer1_model = None
        self.__final_layer2_model = None

        # Initialize status of boolean variables
        self.__cv_obj_l1_bool=False
        self.__cv_obj_l2_bool=False
        self.__apply_resample_l1=False
        self.__apply_resample_l2=False


    def __layer_1_training(self,X,y,layer_wt):
        """Private class method to train the model assigned to Layer 1.

        :param X: Independent variables data found by clustering assigned in Layer 1 for model training
        :type X: Pandas DataFrame or numpy.ndarray

        :param y: Target variable. Target variable values must be in 1/0 format
        :type y: Pandas Series or numpy.ndarray

        :return: Returns None
        :rtype: None
        
        :meta private:
        """

        # Check whether a cross-validation object is provided
        start=time.time()

        if self.__cv_obj_l1_bool:
            sig_cv = inspect.signature(self.__cv_obj_l1.fit)
            sig_model = inspect.signature(self.__cv_obj_l1.estimator.fit)

            if ("sample_weight" in sig_cv.parameters) and ("sample_weight" in sig_model.parameters) and (not self.__sample_wt_l1 is None):
                self.__cv_obj_l1.fit(X,y,sample_weight=layer_wt)

            else:
                self.__cv_obj_l1.fit(X,y)
                    
        else:
            # Check if the model has a parameter of 'sample_weight' present in fit()
            sig = inspect.signature(self.__model_l1.fit)

            if ("sample_weight" in sig.parameters) and (not self.__sample_wt_l1 is None):
                self.__model_l1.fit(X,y,sample_weight=layer_wt)
            
            else:
                self.__model_l1.fit(X,y)
        
        end=time.time()

        self.__l1_time=(end-start)


    def __layer_2_training(self,X,y,layer_wt):
        """Private class method to train the model assigned to Layer 2.

        :param X: Independent variables data found by clustering assigned in Layer 2 for model training
        :type X: Pandas DataFrame or numpy.ndarray

        :param y: Target variable. Target variable values must be in 1/0 format
        :type y: Pandas Series or numpy.ndarray

        :return: Returns None
        :rtype: None

        :meta private:
        """

        # Check whether a cross-validation object is provided
        start=time.time()
        
        if self.__cv_obj_l2_bool:
            sig_cv = inspect.signature(self.__cv_obj_l2.fit)
            sig_model = inspect.signature(self.__cv_obj_l2.estimator.fit)

            if ("sample_weight" in sig_cv.parameters) and ("sample_weight" in sig_model.parameters) and (not self.__sample_wt_l2 is None):
                self.__cv_obj_l2.fit(X,y,sample_weight=layer_wt)
            
            else:
                self.__cv_obj_l2.fit(X,y)
        
        else:
            # Check if the model has a parameter of 'sample_weight' present in fit()
            sig = inspect.signature(self.__model_l2.fit)

            if ("sample_weight" in sig.parameters) and (not self.__sample_wt_l2 is None):
                self.__model_l2.fit(X,y,sample_weight=layer_wt)
            
            else:
                self.__model_l2.fit(X,y)

        end=time.time()

        self.__l2_time=(end-start)


    @classmethod
    def __three_way_grouping(cls, clusters, y):
        """This is a private class method which determines which cluster is an instance a part of (minority/ majority and mixed).

        :param clusters: A nested list which contains instances belonging to which clusters as obtained by hierarchical clustering
        :type clusters: list

        :param y: Class label of instances
        :type y: numpy.ndarray

        :return c_mix_all_ind: Indices of the observations which belong to either the mixed or minority clusters
        :rtype: numpy.ndarray

        :return cluster_summary_dict: A dictionary which indicates count of observations which belong to majority/minority/mixed clusters.
        :rtype: dict

        :meta private:
        """
        
        # c_maj_list = list of clusters containing pure negative class instances (c_maj)
        # c_min_list = list of clusters containing pure positive class instances (c_min)
        # c_mix_list = list of clusters containing mix of positive and negative class instances (c_mix)
        c_maj_list, c_min_list, c_mix_list = [], [], []

        # Loop through individual clusters
        for clst in clusters:
            try:
                # Get the corresponding target label of each instance in that cluster
                y_clt = y[np.asarray(clst)]

                # Get a dictionary with key as 0/1 and value as the count of instances belonging to 0/1
                if len(Counter(y_clt)) == 1:
                    # Since dictionary length is 1, which means that the cluster is a c_maj or c_min cluster
                    # So, all elements must belong to only one class label, which is why it is sufficient enough to check the first element of the list y_clt
                    if y_clt[0] == 0:
                        c_maj_list.append(clst)
                    else:
                        c_min_list.append(clst)

                # Dictionary length is 2, which means the cluster is c_mix
                else:
                    c_mix_list.append(clst)

            except ValueError:
                raise ValueError('Error when creating three-way groups.')

        cluster_summary_dict={
            'MAJORITY_CLUSTER_COUNT':len(c_maj_list),
            'MINORITY_CLUSTER_COUNT':len(c_min_list),
            'MIXED_CLUSTER_COUNT':len(c_mix_list)
        }

        # Create a 1-D array of the indices of all instances which belong to c_mix. np.concatenate() returns a 1-D flattened array, and np.sort() sorts the indices in ascending order
        if len(c_mix_list) > 0:
            c_mix_ind = np.sort(np.concatenate(c_mix_list))
        else:
            c_mix_ind = np.array([])

        if len(c_min_list) > 0:
            c_min_ind = np.sort(np.concatenate(c_min_list))
        else:
            c_min_ind = np.array([])

        # Get the indices of the observations which belong to either the c_mix or c_min clusters
        c_mix_all_ind = np.sort(np.concatenate([c_mix_ind,c_min_ind]))

        return c_mix_all_ind,cluster_summary_dict


    @classmethod
    def __clustering(cls, X, custom_distance_threshold):
        """
        Perform agglomerative heirarchical clustering on the given data.

        Calculation Notes:
        pdist() calculates the pairwise euclidean distance between observations. 
        
        Let 'm' be the number of observations.
        Since a distance matrix is usually a 2D m x m matrix, it is avoided and instead only the upper triangular matrix is stored, discarding all symmetric elements.

        The length of the array will then be m*(m-1)/2.
        
        :param X: Independent variables data on which agglomerative heirarchical clustering will be performed.
        :type X: Pandas DataFrame or numpy.ndarray

        :param custom_distance_threshold: A user provided threshold to get the cut-off distance at which the final set of flat clusters would be formed.
        :type custom_distance_threshold: float, **optional**

        :return clusters: Clusters of indices of data observations
        :rtype clusters: numpy.ndarray

        :return original_distance: Log-transformed intra-cluster distance of µ + σ. This would be the cut-off for obtaining flat clsuters.
        :rtype original_distance: float

        :meta private:
        """
        d = pdist(X)  # PAIRWISE DISTANCE BETWEEN INSTANCES

        Z = linkage(d, method='ward')  # calculate the instances links for the agglomerative clustering

        Z[:, 2] = np.log(1 + Z[:, 2])
        sZ = np.std(Z[:, 2])  # standard deviation of the distance between each cluster linkage
        mZ = np.mean(Z[:, 2])  # mean of the distance between each cluster linkage

        original_distance= sZ+mZ

        if pd.isna(custom_distance_threshold):
            cut_off_distance = original_distance
        else:
            cut_off_distance=custom_distance_threshold

        clustLabs = fcluster(Z, cut_off_distance,criterion='distance')  # for the clusters using merger distance threshold as mZ+stds*sZ

        #: 'clusters' variable is a list of list. The number of elements in the 'clusters' variable is equal to the number of clusters formed.
        #: Each nested list houses the indices of the observation belonging to that particular cluster. Egs: Cluster 1 has observations whose indices are [3,7].
        clusters = []
        for lab in np.unique(clustLabs):  # produce a clusters data structure to work with
            clusters.append(np.where(clustLabs == lab)[0])

        return clusters,original_distance


    def fit(self, X, y):
        """Fit the ICLL model on your dataset.

        :param X: Independent variables data for model training

            .. note::

                Since the first step of ICLL model is to perform the agglomerative hierarchical clustering, it is expected that all elements of :py:attr:`X` will only contain **non-null integer or floating** data types. Failure to adhere to this will result in a :py:exc:`ValueError` being raised instead.

        :type X: Pandas DataFrame or numpy.ndarray

        :param y: Target variable. Target variable values must be in 1/0 format
        :type y: Pandas Series or numpy.ndarray

        :return: Fitted estimator.
        :rtype: Object

        :meta public:
        """
        # Set initial status of boolean variables
        self.__dual_model=False

        # Set initial values of core variables
        self.__decisive_action = ""

        self.__computed_distance = np.nan
        self.__clustering_info_dict ={
            'MAJORITY_CLUSTER_COUNT':np.nan,
            'MINORITY_CLUSTER_COUNT':np.nan,
            'MIXED_CLUSTER_COUNT':np.nan
        }
        self.__l1_time=np.nan
        self.__l2_time=np.nan

        self.__final_layer1_model = None
        self.__final_layer2_model = None


        # Perform checks to ensure that the class object has been initialized with correct attribute datatypes
        if not ((type(self.__cv_obj_l1).__name__ in ["GridSearchCV","RandomizedSearchCV"]) or pd.isna(self.__cv_obj_l1)):
            raise ValueError("Invalid value for parameter 'cv_obj_l1'. This parameter only accepts an object from class 'GridSearchCV' or 'RandomizedSearchCV'. This parameter can also accept None if hyperparameter tuning on Layer 1 model is not desired.")
        else:
            if pd.isna(self.__cv_obj_l1):
                pass
            else:
                self.__cv_obj_l1_bool=True

                # Check if refit parameter is initialized with False
                if self.__cv_obj_l1.refit is False:
                    raise ValueError("'refit' parameter in 'cv_obj_l1' MUST NOT be initialized with a value of False!")

        
        if not ((type(self.__cv_obj_l2).__name__ in ["GridSearchCV","RandomizedSearchCV"]) or pd.isna(self.__cv_obj_l2)):
            raise ValueError("Invalid value for parameter 'cv_obj_l2'. This parameter only accepts an object from class 'GridSearchCV' or 'RandomizedSearchCV'. This parameter can also accept None if hyperparameter tuning on Layer 2 model is not desired.")
        else:
            if pd.isna(self.__cv_obj_l2):
                pass
            else:
                self.__cv_obj_l2_bool=True

                # Check if refit parameter is initialized with False
                if self.__cv_obj_l2.refit is False:
                    raise ValueError("'refit' parameter in 'cv_obj_l2' MUST NOT be initialized with a value of False!")



        if not ((hasattr(self.__model_l1, "fit") and callable(self.__model_l1.fit)) or pd.isna(self.__model_l1)):
            raise ValueError("Model to be trained on Layer 1 does not have a callable fit(X,y) method")
        else:
            if (pd.isna(self.__model_l1) and (not self.__cv_obj_l1_bool)):
                raise ValueError("Both parameters 'model_l1' and 'cv_obj_l1' for Layer 1 task are empty! Atleast one should be provided.")
            else:
                pass

        if not ((hasattr(self.__model_l2, "fit") and callable(self.__model_l2.fit)) or pd.isna(self.__model_l2)):
            raise ValueError("Model to be trained on Layer 2 does not have a callable fit(X,y) method")
        else:
            if (pd.isna(self.__model_l2) and (not self.__cv_obj_l2_bool)):
                raise ValueError("Both parameters 'model_l2' and 'cv_obj_l2' for Layer 2 task are empty! Atleast one should be provided.")
            else:
                pass

        if not ((type(self.__smote_obj_l1).__name__=="SMOTE") or pd.isna(self.__smote_obj_l1)):
            raise ValueError("Resampling method for layer 1 does not belong to imblearn package -> SMOTE. This parameter can also accept None if resampling tuning on Layer 1 training data is not desired.")
        else:
            if pd.notna(self.__smote_obj_l1):
                self.__apply_resample_l1=True


        if not ((type(self.__smote_obj_l2).__name__=="SMOTE") or pd.isna(self.__smote_obj_l2)):
            raise ValueError("Resampling method for layer 2 does not belong to imblearn package -> SMOTE. This parameter can also accept None if resampling tuning on Layer 2 training data is not desired.")
        else:
            if pd.notna(self.__smote_obj_l2):
                self.__apply_resample_l2=True

        if not (isinstance(self.__sample_wt_l1,(list,np.ndarray,pd.Series)) or (self.__sample_wt_l1 is None)):
            raise ValueError("Parameter 'sample_wt_l1' is not a List/numpy.ndarray/Pandas Series data type. This parameter can also accept None if no sample weights are to be assigned during model training of Layer 1.")
        else:
            if (self.__sample_wt_l1 is None):
                pass
            else:
                self.__sample_wt_l1 = np.array(self.__sample_wt_l1)

                if self.__sample_wt_l1.ndim != 1:
                    raise ValueError("Parameter 'sample_wt_l1' must be one-dimensional.")
                else:
                    # Override resampling on Layer 1
                    self.__apply_resample_l1 = False

        if not (isinstance(self.__sample_wt_l2,(list,np.ndarray,pd.Series)) or (self.__sample_wt_l2 is None)):
            raise ValueError("Parameter 'sample_wt_l2' is not a List/numpy.ndarray/Pandas Series data type. This parameter can also accept None if no sample weights are to be assigned during model training of Layer 2.")
        else:
            if (self.__sample_wt_l2 is None):
                pass
            else:
                self.__sample_wt_l2 = np.array(self.__sample_wt_l2)

                if self.__sample_wt_l2.ndim != 1:
                    raise ValueError("Parameter 'sample_wt_l2' must be one-dimensional.")
                else:
                    # Override resampling on Layer 2
                    self.__apply_resample_l2 = False

        if not isinstance(X,(pd.DataFrame,np.ndarray)):
            raise ValueError("Parameter 'X' is not a Pandas Dataframe or a numpy.ndarray datatype!")

        if not isinstance(y,(pd.Series,np.ndarray)):
            raise ValueError("Parameter 'y' is not a Pandas Series or a numpy.ndarray datatype!")

        if not (isinstance(self.__custom_distance_threshold,(int,float)) or pd.isna(self.__custom_distance_threshold)):
            raise ValueError("Parameter 'custom_distance_threshold' is not an integer or float datatype. This parameter can also accept None if the originally computed intra-cluster distance is to be retained.")

        # Check if a negative / zero distance is passed
        if pd.isna(self.__custom_distance_threshold):
            pass
        else:
            if (self.__custom_distance_threshold<=0) or (not np.isfinite(self.__custom_distance_threshold)):
                raise ValueError("Parameter 'custom_distance_threshold' must contain a positive, non-zero finite value!")

        # Convert pandas series (if given) to numpy array
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must contain the same number of samples; got {X.shape[0]} samples in X and {y.shape[0]} in y.")

        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError("'X' must be two-dimensional and contain at least one feature!")

        if X.shape[0] < 2:
            raise ValueError("'X' must contain at least two samples!")
        
        # Check if y contains more than one dimension.
        if y.ndim != 1:
            raise ValueError("Parameter 'y' must be one-dimensional.")

        # Check if the pandas dataframe/ numpy array of X contains integer and floating datatypes only
        if isinstance(X, pd.DataFrame):
            # Check if all cells have integer or float values
            if not all(is_integer_dtype(dtype) or is_float_dtype(dtype) for dtype in X.dtypes):
                raise ValueError("All values in 'X' must have integer or float dtypes!")

            # Check if 'X' only contains finite values
            if not np.isfinite(X.values).all():
                raise ValueError("'X' must not contain any missing or infinite values!")

        else:
            X = np.asarray(X)
            if X.dtype.kind not in "iuf":
                raise ValueError("All values in 'X' must have integer or float dtypes!")

            # Check if 'X' only contains finite values
            if not np.isfinite(X).all():
                raise ValueError("'X' must not contain any missing or infinite values!")


        # Check if y contains only labels of 0 and 1.
        if (not np.isin(y, [0, 1]).all() or not np.any(y == 0) or not np.any(y == 1)):
            raise ValueError("Parameter 'y' must contain both binary class labels 0 and 1.")

        if self.__sample_wt_l1 is None:
            pass
        else:
            # If self.__sample_wt_l1 is passed, then check if the length is the same as the length of 'X'
            if len(self.__sample_wt_l1)!=len(X):
                raise ValueError(f"sample_wt_l1 and X must contain the same number of samples; got {X.shape[0]} samples in X and {len(self.__sample_wt_l1)} in sample_wt_l1.")

        if self.__sample_wt_l2 is None:
            pass
        else:
            # If self.__sample_wt_l2 is passed, then check if the length is the same as the length of 'X'
            if len(self.__sample_wt_l2)!=len(X):
                raise ValueError(f"sample_wt_l2 and X must contain the same number of samples; got {X.shape[0]} samples in X and {len(self.__sample_wt_l2)} in sample_wt_l2.")

        # Perform clustering
        self.__clusters, self.__computed_distance = self.__clustering(X,self.__custom_distance_threshold)

        # Perform grouping of instances
        self.__all_ind_arr, self.__clustering_info_dict = self.__three_way_grouping(self.__clusters, y)

        # Case 1: The base case. All types of clusters are formed.
        if (self.__clustering_info_dict['MAJORITY_CLUSTER_COUNT']>0) and (self.__clustering_info_dict['MIXED_CLUSTER_COUNT']>0) and (self.__clustering_info_dict['MINORITY_CLUSTER_COUNT']>0):
            self.__decisive_action = 'C1'

        # Case 2: No pure majority clusters are formed.
        elif (self.__clustering_info_dict['MAJORITY_CLUSTER_COUNT']==0) and (self.__clustering_info_dict['MIXED_CLUSTER_COUNT']>0) and (self.__clustering_info_dict['MINORITY_CLUSTER_COUNT']>0):
            self.__decisive_action = 'C2'

        # Case 3: No mixed clusters are formed.
        elif (self.__clustering_info_dict['MAJORITY_CLUSTER_COUNT']>0) and (self.__clustering_info_dict['MIXED_CLUSTER_COUNT']==0) and (self.__clustering_info_dict['MINORITY_CLUSTER_COUNT']>0):
            self.__decisive_action = 'C3'

        # Case 4: No pure minority clusters are formed
        elif (self.__clustering_info_dict['MAJORITY_CLUSTER_COUNT']>0) and (self.__clustering_info_dict['MIXED_CLUSTER_COUNT']>0) and (self.__clustering_info_dict['MINORITY_CLUSTER_COUNT']==0):
            self.__decisive_action = 'C4'
            warnings.warn("No pure minority clusters was formed! ICLL model training will carry on as usual.")

        # Case 5: Only mixed clusters are formed
        elif (self.__clustering_info_dict['MAJORITY_CLUSTER_COUNT']==0) and (self.__clustering_info_dict['MIXED_CLUSTER_COUNT']>0) and (self.__clustering_info_dict['MINORITY_CLUSTER_COUNT']==0):
            self.__decisive_action = 'C5'

        # Case 6: Base case (will never happen)
        else:
            self.__decisive_action = 'C6'

        # Raise warnings and stop ICLL model training if case 'C2' arises
        if self.__decisive_action == 'C2':
            warnings.warn(f"No pure majority clusters were formed! Try to instantiate ICLL again by providing a new distance threshold for parameter 'custom_distance_threshold'. The computed distance threshold is in this run is {self.__computed_distance}")

        # Raise warnings and stop model training if case 'C3' arises
        elif self.__decisive_action == 'C3':
            warnings.warn("No mixed clusters were formed! Since heirarchical clustering was able to seperate both classes effectively, so the ICLL model approach for this predictive task is redundant.")

        # Raise warnings and stop model training if case 'C5' arises
        elif self.__decisive_action == 'C5':
            warnings.warn("Only a mixed cluster was formed! ICLL model training will not carry on further.")

        #This block will never trigger
        elif self.__decisive_action == 'C6':
            warnings.warn("only a single type of pure majority/ pure mixed/ pure minority cluster was formed!  ICLL model training will not carry on further.")

        # Continue with ICLL model training as usual
        else:

            # Since __all_ind_arr variable contains indices of observations belonging to either c_mix or c_min clusters, those observations are tagged as 1 for L1 predictive task
            # The observations belonging to the majority class can get their label flipped from 0 to 1 if they belong to a c_mix cluster.
            y_l1 = y.copy()
            y_l1[self.__all_ind_arr] = 1

            if not self.__apply_resample_l1:
                self.__layer_1_training(X, y_l1,self.__sample_wt_l1)

            else:
                try:
                    X_L1, y_L1 = self.__smote_obj_l1.fit_resample(X, y_l1)
                    self.__layer_1_training(X_L1, y_L1,self.__sample_wt_l1)

                except ValueError as e:
                    warnings.warn(f"Layer 1 SMOTE failed due to: ({e})\n Performing model fit of Layer 1 without resampling...",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                    self.__layer_1_training(X, y_l1,self.__sample_wt_l1)

            if isinstance(X, pd.DataFrame):
                X_l2 = X.iloc[self.__all_ind_arr, :]
            else:
                X_l2 = X[self.__all_ind_arr]

            y_l2 = y[self.__all_ind_arr]

            # Choose only the corresponding sample weights of instances in Layer 2 (ie: instances belonging to Mixed and Pure Minority)
            if self.__sample_wt_l2 is None:
                modded_sample_wt_l2 = None
            else:
                modded_sample_wt_l2 = self.__sample_wt_l2[self.__all_ind_arr]

            if not self.__apply_resample_l2:
                self.__layer_2_training(X_l2, y_l2,modded_sample_wt_l2)

            else:
                try:
                    X_L2, y_L2 = self.__smote_obj_l2.fit_resample(X_l2, y_l2)
                    self.__layer_2_training(X_L2, y_L2,modded_sample_wt_l2)

                except ValueError as e:
                    warnings.warn(f"Layer 2 SMOTE failed due to: ({e})\n Performing model fit of Layer 2 without resampling...",
                        RuntimeWarning,
                        stacklevel=2,
                    )

                    self.__layer_2_training(X_l2, y_l2,modded_sample_wt_l2)

            if self.__cv_obj_l1_bool:
                self.__final_layer1_model=self.__cv_obj_l1.best_estimator_
            else:
                self.__final_layer1_model=self.__model_l1

            if self.__cv_obj_l2_bool:
                self.__final_layer2_model=self.__cv_obj_l2.best_estimator_
            else:
                self.__final_layer2_model=self.__model_l2

            # Since both the models have finished training, update the status
            self.__dual_model = True
            
        return self


    @staticmethod
    def __layer_validator(layer_name):
        """This function is to simply check whether the input is of a string datatype and the value is either 'L1' or 'L2'

        :param layer_name: The layer for which functionalities must be accessed
        :type layer_name: str

        :return: Function will return True if the check is passed
        :rtype: bool

        :meta private:
        """
        if not isinstance(layer_name,str):
            raise ValueError("Parameter 'layer_name' is not of a string datatype!")
        else:
            if layer_name.upper() not in ['L1','L2']:
                raise ValueError("Acceptable values for 'layer_name' parameter is only 'L1' or 'L2'! Please specify the proper layer name!")
            else:
                return True

    @staticmethod
    def __positive_class_probability(estimator, X_test, layer_name):
        """
        A validation function for getting the prediction probabilities of an estimator.

        :param estimator: The particular estimator of the layer for which we want probabilistic predictions.
        :type estimator: Object

        :param X_test: The test dataset over which probabilistic predictions are to be made
        :type X_test: Pandas DataFrame or numpy.ndarray

        :param layer_name: The name of the layer for which validation is to be carried out.
        :type layer_name: string

        :return: Probability of the sample for belonging to the **positive class**
        :rtype: numpy.ndarray

        :meta private:
        """
        # Check whether the 'classes_' method exists and can be called for the models in both the layers
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            raise ValueError(f"{layer_name} estimator does not expose fitted classes_.")

        # Check whether the 'predict_proba' method exists and can be called for the models in both the layers
        predict_proba = getattr(estimator, "predict_proba", False)
        if not callable(predict_proba):
            raise ValueError(f"Estimator for {layer_name} must have a callable predict_proba() method!")

        classes = np.asarray(classes)
        probabilities = np.asarray(predict_proba(X_test))

        if (classes.ndim != 1) or (not np.isin(classes, [0, 1]).all()):
            raise ValueError(f"{layer_name} estimator must be fitted with binary classes 0 and 1.")

        if (probabilities.ndim != 2) or (probabilities.shape != (X_test.shape[0], classes.size)):
            raise ValueError(f"{layer_name} estimator returned an unexpected probability shape.")

        # Store the indices of the array of only the positive class
        positive_columns = np.flatnonzero(classes == 1)

        # np.ndarray.size returns the count of elements throughout the entire array
        if positive_columns.size == 1:

            # positive_columns[0] is guaranteed to return the probabilities of only the positive class
            return probabilities[:, positive_columns[0]]

        else:
            raise ValueError(f"{layer_name} estimator must have exactly one probability column for class 1.")


    def predict(self, X):
        r"""Predict class labels for samples in X.

        This class method is only valid when both the models for **Layer 1** and **Layer 2** training have a valid :py:meth:`!predict` method.
        
        Let :math:`f_{L1}(x_i)` and :math:`f_{L2}(x_i)` denote the models trained in the layers L1 and L2, respectively. The the prediction by ICLL model is given by: :math:`f_{L1}(x_i) \cdot f_{L2}(x_i)`

        Thus, a sample instance would only receive a predicted class label of 1 only when the predicted class label of that instance for both the models :math:`f_{L1}(x_i)` and :math:`f_{L2}(x_i)` is 1.

        .. note::
            
            The ICLL method :py:meth:`predict` will not work if any of the following scenarios arise by clustering:
        
            | **Scenario 1**: No pure majority clusters are formed.
            | **Scenario 2**: No mixed clusters are formed.
            | **Scenario 3**: Only mixed clusters are formed.

            A :py:exc:`ValueError` will be raised instead if any of the above three scenarios is encountered.

        :param X: The data matrix for which we want to get the predictions.
        :type X: Pandas DataFrame or numpy.ndarray

        :return: Vector containing the class labels for each sample. This would be an ndarray of shape (n_samples,)
        :rtype: numpy.ndarray

        :meta public:
        """
        # Check whether the 'predict' method exists and can be called for the models in both the layers
        if not callable(getattr(self.__final_layer1_model, "predict", False)):
            raise ValueError(f"Estimator in Layer 1 must have a callable predict() method!")

        if not callable(getattr(self.__final_layer2_model, "predict", False)):
            raise ValueError(f"Estimator in Layer 2 must have a callable predict() method!")

        # Situation where the fitted models are available on both the layers
        if self.__dual_model:
            yh_l1, yh_l2 = self.__final_layer1_model.predict(X), self.__final_layer2_model.predict(X)

            yh_f = np.asarray([x1 * x2 for x1, x2 in zip(yh_l1, yh_l2)])

            return yh_f

        else:
            raise ValueError("ICLL model has not been trained as clustering produced no pure majority clusters, no mixed clusters, or only mixed clusters.")


    def predict_proba(self, X):
        r"""Predict probability estimates.

        This class method is only valid when both the models for Layer 1 and Layer 2 training have a valid :py:meth:`!predict_proba` method.
        
        Let :math:`f_{L1}(x_i)` and :math:`f_{L2}(x_i)` denote the models trained in the layers L1 and L2, respectively. The the prediction by ICLL model is given by: :math:`f_{L1}(x_i) \cdot f_{L2}(x_i)`

        .. note::
            
            The ICLL method :py:meth:`predict_proba` will not work if any of the following scenarios arise by clustering:
        
            | **Scenario 1**: No pure majority clusters are formed.
            | **Scenario 2**: No mixed clusters are formed.
            | **Scenario 3**: Only mixed clusters are formed.

            A :py:exc:`ValueError` will be raised instead if any of the above three scenarios is encountered.

            |
            | **IMPORTANT**

            The ICLL model will always return the final probabilities based upon the probabilites of the positive class retured by each layer's estimator. So, to identify the probabilities returned by the positive class, the estimator corresponding to both the layers must necessarily have a :py:attr:`classes_` attribute.
            
            If the method does not have a :py:attr:`classes_` attribute, a :py:exc:`ValueError` will be raised instead.

        :param X: The data matrix for which we want to get the predictions.
        :type X: Pandas DataFrame or numpy.ndarray

        :return: Probability of the sample for belonging to the **positive class**
        :rtype: numpy.ndarray

        :meta public:
        """

        # Situation where the fitted models are available on both the layers
        if self.__dual_model:
            yh_l1_p = self.__positive_class_probability(self.__final_layer1_model,X,'Layer 1')
            yh_l2_p = self.__positive_class_probability(self.__final_layer2_model,X,'Layer 2')

            yh_fp = np.asarray([x1 * x2 for x1, x2 in zip(yh_l1_p, yh_l2_p)])
            
            return yh_fp

        else:
            raise ValueError("ICLL model has not been trained as clustering produced no pure majority clusters, no mixed clusters, or only mixed clusters.")


    def get_computed_dist_threshold(self):
        """Returns the log-transformed intra-cluster distance of **µ + σ**

        :return: log-transformed intra-cluster distance of **µ + σ**
        :rtype: float
        """
        return self.__computed_distance


    def get_cluster_type_info(self):
        """Get the count of each type of cluster formed (pure majority, mixed and pure minority).

        :return: A dictionary where the key is the name of the type of cluster, and its corresponding value is the count of such clusters.
        :rtype: dict
        """
        return self.__clustering_info_dict

    
    def get_layer_search_time(self,layer_name):
        """Pass 'L1' to parameter :py:attr:`layer_name` to return the time taken for hyperparameter tuning on Layer 1 when a :py:obj:`GridSearchCV` or :py:obj:`RandomizedSearchCV` object is provided. 
        Otherwise, it returns the time taken to train the **Layer 1** model.

        Pass 'L2' to parameter :py:attr:`layer_name` to return the time taken for hyperparameter tuning on Layer 2 when a :py:obj:`GridSearchCV` or :py:obj:`RandomizedSearchCV` object is provided. 
        Otherwise, it returns the time taken to train the **Layer 2** model.

        .. note::
            
            The ICLL method :py:meth:`get_layer_search_time` will not work if any of the following scenarios arise by clustering:
        
            | **Scenario 1**: No pure majority clusters are formed.
            | **Scenario 2**: No mixed clusters are formed.
            | **Scenario 3**: Only mixed clusters are formed.

            A :py:exc:`ValueError` will be raised instead if any of the above three scenarios is encountered.

        :param layer_name: The name of the layer for which functionalities are to be accessed.
        :type layer_name: str

        :return: Time measured in seconds.
        :rtype: float

        :meta public:
        """
        # Validate parameter 'layer_name'
        self.__layer_validator(layer_name)

        # Convert to uppercase
        layer_name=layer_name.upper()

        # Situation where both the fitting times are available for access
        if self.__dual_model:
            if layer_name=='L1':
                return self.__l1_time
            else:
                return self.__l2_time
        
        else:
            raise ValueError("ICLL model has not been trained as clustering produced no pure majority clusters, no mixed clusters, or only mixed clusters.")     


    def get_fitted_cv_layer_object(self,layer_name):
        """Specify 'L1' or 'L2' to parameter :py:attr:`layer_name` to retrieve the fitted object for that layer. If a GridSearchCV/RandomizedSearchCV object was provided, the fitted CV object is returned.

        .. note::
            
            The ICLL method :py:meth:`get_fitted_cv_layer_object` will not work if any of the following scenarios arise by clustering:
        
            | **Scenario 1**: No pure majority clusters are formed.
            | **Scenario 2**: No mixed clusters are formed.
            | **Scenario 3**: Only mixed clusters are formed.

            A :py:exc:`ValueError` will be raised instead if any of the above three scenarios is encountered.

        :param layer_name: The name of the layer for which functionalities are to be accessed.
        :type layer_name: str

        :return: Fitted :py:obj:`GridSearchCV` or :py:obj:`RandomizedSearchCV` object if provided at the time of object initialization.
        :rtype: object
        """
        # Validate parameter 'layer_name'
        self.__layer_validator(layer_name)

        # Convert to uppercase
        layer_name=layer_name.upper()

        # Situation where both the CV objects are available for access
        if self.__dual_model:
            if (layer_name=='L1'):
                if (self.__cv_obj_l1_bool):
                    return self.__cv_obj_l1
                else:
                    raise ValueError("Parameter 'cv_obj_l1' was empty, CV object cannot be accessed!")
            
            else:
                if (self.__cv_obj_l2_bool):
                    return self.__cv_obj_l2
                else:
                    raise ValueError("Parameter 'cv_obj_l2' was empty, CV object cannot be accessed!")

        else:
            raise ValueError("ICLL model has not been trained as clustering produced no pure majority clusters, no mixed clusters, or only mixed clusters.")


    def get_fitted_model_layer_object(self,layer_name):
        """Specify 'L1' or 'L2' for :py:attr:`layer_name` to retrieve the fitted model from that layer. If the layer uses a :py:obj:`GridSearchCV` or :py:obj:`RandomizedSearchCV` object, the model from :py:attr:`best_estimator_` is returned. Otherwise, the fitted model stored in :py:attr:`model_l1` or :py:attr:`model_l2` is returned.

        .. note::
            
            The ICLL method :py:meth:`get_fitted_model_layer_object` will not work if any of the following scenarios arise by clustering:
        
            | **Scenario 1**: No pure majority clusters are formed.
            | **Scenario 2**: No mixed clusters are formed.
            | **Scenario 3**: Only mixed clusters are formed.

            A :py:exc:`ValueError` will be raised instead if any of the above three scenarios is encountered.

        :param layer_name: The name of the layer for which functionalities are to be accessed.
        :type layer_name: str

        :return: Fitted model
        :rtype: object
        """
        # Validate parameter 'layer_name'
        self.__layer_validator(layer_name)

        # Convert to uppercase
        layer_name=layer_name.upper()

        # Situation where both the fitted models are available for access
        if self.__dual_model:
            if (layer_name=='L1'):
                return self.__final_layer1_model            
            else:
                return self.__final_layer2_model
            
        else:
            raise ValueError("ICLL model has not been trained as clustering produced no pure majority clusters, no mixed clusters, or only mixed clusters.")
