import pandas as pd
import numpy as np
import time
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV


class NoGreyZoneError(ValueError):
    pass


class ICLLTuning:
    """This is the class representation of this library. Invoke the methods of this class to perform hyperparameter tuning on the models fed to the Imbalanced Classification via Layered Learning (ICLL) method

    'Layer 1' follows the same definition as mentioned within the original paper: "Layer 1 (L1) denotes the event that the observation belongs to the Mixed or Pure Minority Group"
    'Layer 2' follows the same definition as mentioned within the original paper: "Given that L1 occurs (i.e. observation belongs to the mixed or pure minority group), Layer 2 (L2) denotes the event that the observation belongs to the pure minority class."
    
    :param model_l1: ML model to be fit on **Layer 1**. A model must be passed if you don't want hyperparameter tuning to be performed on Layer 1 task. 
    
    Requirements:
    1. This model must have a callable fit(X,y) method.
    2. This model must have a callable predict(X) in the case of hard classification or predict_proba(X) in the case of soft classification. Either one provided is fine.
    
    In the situation where both the model and cross-validation object are both provided, the model assigned to the cross-validation object will take precedence in the ICLL algorithm.
    :type model_l1: object, optional

    :param model_l2: ML model to be fit on **Layer 2**. A model must be passed if you don't want hyperparameter tuning to be performed on Layer 2 task. 
    
    Requirements:
    1. This model must have a callable fit(X,y) method.
    2. This model must have a callable predict(X) in the case of hard classification or predict_proba(X) in the case of soft classification. Either one provided is fine.
    
    In the situation where both the model and cross-validation object are both provided, the model assigned to the cross-validation object will take precedence in the ICLL algorithm.
    :type model_l2: object, optional

    :param cv_obj_l1: Grid Search CV/ Randomized Search CV object to be fit on Layer 1
    :type cv_obj_l1: 'sklearn.tree._classes.DecisionTreeClassifier' or 'sklearn.model_selection._search.RandomizedSearchCV' object, optional

    :param cv_obj_l2: Grid Search CV/ Randomized Search CV object to be fit on Layer 2
    :type cv_obj_l2: 'sklearn.tree._classes.DecisionTreeClassifier' or 'sklearn.model_selection._search.RandomizedSearchCV' object, optional

    :param smote_obj_l1: SMOTE object to be passed for resampling on Layer 1
    :type smote_obj_l1: imblearn.over_sampling._smote.base.SMOTE, optional

    :param smote_obj_l2: SMOTE object to be passed for resampling on Layer 2
    :type smote_obj_l2: imblearn.over_sampling._smote.base.SMOTE, optional

    :param custom_distance_threshold: Provide a value for this parameter if you want to override the original implementation of the computation of distance threshold (flat clusters are obtained at a threshold of the log-transformed intra-cluster distance of µ + σ)
    :type custom_distance_threshold: float, optional

    :param precedence_layer: Action to be taken if all the instances are assigned to clusters which comprise of a mix of minority and majority class observations. By default, the model (or cross-validation object in case if no model is provided) to layer 2 is used. Supply 'l1' if the model (or cross-validation object in case if no model is provided) of layer 1 is to be used instead.
    :type precedence_layer: str, optional

    TO DO:


    """
    def __init__(self,
                 model_l1,
                 model_l2,
                 cv_obj_l1,
                 cv_obj_l2,
                 smote_obj_l1,
                 smote_obj_l2,
                 custom_distance_threshold,
                 precedence_layer='l2'
                 ):
        
        self.model_l1 = model_l1
        self.model_l2 = model_l2
        self.cv_obj_l1 = cv_obj_l1
        self.cv_obj_l2 = cv_obj_l2
        self.smote_obj_l1 = smote_obj_l1
        self.smote_obj_l2 = smote_obj_l2
        self.custom_distance_threshold = custom_distance_threshold
        self.precedence_layer = precedence_layer

        self.clusters = []
        self.grey_ind_arr = np.array([])
        self.clustering_status=''
        self.l1_time=np.nan
        self.l2_time=np.nan


    def predict(self, X):

        yh_l1, yh_l2 = self.final_layer1_model.predict(X), self.final_layer2_model.predict(X)

        yh_f = np.asarray([x1 * x2 for x1, x2 in zip(yh_l1, yh_l2)])

        return yh_f

    def predict_proba(self, X):

        yh_l1_p = self.final_layer1_model.predict_proba(X)
        try:
            yh_l1_p = np.array([x[1] for x in yh_l1_p])
        except IndexError:
            yh_l1_p = yh_l1_p.flatten()
        yh_l2_p = self.final_layer2_model.predict_proba(X)
        yh_l2_p = np.array([x[1] for x in yh_l2_p])

        yh_fp = np.asarray([x1 * x2 for x1, x2 in zip(yh_l1_p, yh_l2_p)])

        return yh_fp


    def predict_l2(self, X):

        return self.final_layer2_model.predict(X)


    def predict_l1(self, X):

        return self.final_layer1_model.predict(X)


    def predict_proba_l2(self, X):

        yh_l2_p = self.final_layer2_model.predict_proba(X)
        yh_l2_p = np.array([x[1] for x in yh_l2_p])

        return yh_l2_p


    def predict_proba_l1(self, X):

        yh_l1_p = self.final_layer1_model.predict_proba(X)
        yh_l1_p = np.array([x[1] for x in yh_l1_p])

        return yh_l1_p


    def layer_2_training(self,X,y):

        # Check whether a cross-validation object is provided
        start=time.time()
        if self.cv_obj_l2_bool:
            self.cv_obj_l2.fit(X,y)
        else:
            self.model_l2.fit(X,y)
        end=time.time()

        self.l2_time=(end-start)


    def layer_1_training(self,X,y):

        # Check whether a cross-validation object is provided
        start=time.time()
        if self.cv_obj_l1_bool:
            self.cv_obj_l1.fit(X,y)
        else:
            self.model_l1.fit(X,y)
        end=time.time()

        self.l1_time=(end-start)


    @classmethod
    def three_way_grouping(cls, clusters, y):

        """
        whites = list of clusters containing pure negative class instances (c_maj)

        black = list of clusters containing pure positive class instances (c_min)

        greys = list of clusters containing mix of positive and negative class instances (c_mix)
        """
        whites, blacks, greys = [], [], []
        decision_flag = ''

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
                        whites.append(clst)
                    else:
                        blacks.append(clst)

                # Dictionary length is 2, which means the cluster is c_mix
                else:
                    greys.append(clst)
            except ValueError:
                raise ValueError('Error when creating three-way groups.')

        if (len(whites)==0) and (len(blacks)==0):
            decision_flag='dual_empty'
        elif len(blacks)==0:
            decision_flag='c_min'
        elif len(greys) == 0:
            decision_flag='c_mix'
        elif len(whites)==0:
            decision_flag='c_maj'

        # Create a 1-D array of the indices of all instances which belong to c_mix. np.concatenate() returns a 1-D flattened array, and np.sort() sorts the indices in ascending order
        grey_ind = np.sort(np.concatenate(greys))

        if len(blacks) > 0:
            black_ind = np.sort(np.concatenate(blacks))
        else:
            black_ind = np.array([])

        # Get the indices of the observations which belong to either the c_mix or c_min clusters
        greyb_ind = np.sort(np.concatenate([grey_ind,black_ind]))

        return greyb_ind,decision_flag

    @classmethod
    def clustering(cls, X, custom_distance_threshold):
        
        """
        pdist() calculates the pairwise euclidean distance between observations. 
        
        Let 'm' be the number of observations.
        Since a distance matrix is usually a 2D m x m matrix, it is avoided and instead only the upper triangular matrix is stored, discarding all symmetric elements.

        The length of the array will then be m*(m-1)/2.
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


        """
        'clusters' variable is a list of list. The number of elements in the 'clusters' variable is equal to the number of clusters formed.

        Each nested list houses the indices of the observation belonging to that particular cluster. Egs: Cluster 1 has observations whose indices are [3,7].
        """
        clusters = []
        for lab in np.unique(clustLabs):  # produce a clusters data structure to work with
            clusters.append(np.where(clustLabs == lab)[0])

        return clusters,original_distance


    def fit(self, X: pd.DataFrame, y):
        """
        Fit the ICLL model on your dataset.

        :param X: Independent variables data for model training
        :type X: Pandas DataFrame or numpy.ndarray

        :param y: Target variable. Target variable values must be in 1/0 format
        :type y: Pandas Series or numpy.ndarray

        :return: Returns None
        :rtype: None
        """
        # Initialize boolean variables
        self.cv_obj_l1_bool=False
        self.cv_obj_l2_bool=False
        apply_resample_l1=False
        apply_resample_l2=False

        # Perform checks to ensure that the class object has been initialized with correct attribute datatypes
        if not (isinstance(self.cv_obj_l1,GridSearchCV,RandomizedSearchCV) or pd.isna(self.cv_obj_l1)):
            raise ValueError("Invalid value for parameter 'cv_obj_l1'. This parameter only accepts an object from class 'GridSearchCV' or 'RandomizedSearchCV'. This parameter can also accept None if hyperparameter tuning on Layer 1 model is not desired.")
        else:
            if pd.isna(self.cv_obj_l1):
                pass
            else:
                self.cv_obj_l1_bool=True
        
        if not (isinstance(self.cv_obj_l2,GridSearchCV,RandomizedSearchCV) or pd.isna(self.cv_obj_l2)):
            raise ValueError("Invalid value for parameter 'cv_obj_l2'. This parameter only accepts an object from class 'GridSearchCV' or 'RandomizedSearchCV'. This parameter can also accept None if hyperparameter tuning on Layer 2 model is not desired.")
        else:
            if pd.isna(self.cv_obj_l2):
                pass
            else:
                self.cv_obj_l2_bool=True


        if not ((hasattr(self.model_l1, "fit") and callable(self.model_l1.fit)) or pd.isna(self.model_l1)):
            raise ValueError("Model to be trained on Layer 1 does not have a callable fit(X,y) method")
        else:
            if (pd.isna(self.model_l1) and (not self.cv_obj_l1_bool)):
                raise ValueError("Both parameters 'model_l1' and 'cv_obj_l1' for Layer 1 task are empty! Atleast one should be provided.")
            else:
                pass

        if not ((hasattr(self.model_l2, "fit") and callable(self.model_l2.fit)) or pd.isna(self.model_l2)):
            raise ValueError("Model to be trained on Layer 2 does not have a callable fit(X,y) method")
        else:
            if (pd.isna(self.model_l2) and (not self.cv_obj_l2_bool)):
                raise ValueError("Both parameters 'model_l2' and 'cv_obj_l2' for Layer 2 task are empty! Atleast one should be provided.")
            else:
                pass

        if not (isinstance(self.smote_obj_l1, SMOTE) or pd.isna(self.smote_obj_l1)):
            raise ValueError("Resampling method for layer 1 does not belong to imblearn package -> SMOTE. This parameter can also accept None if resampling tuning on Layer 1 training data is not desired.")
        else:
            apply_resample_l1=True

        if not (isinstance(self.smote_obj_l2, SMOTE) or pd.isna(self.smote_obj_l2)):
            raise ValueError("Resampling method for layer 2 does not belong to imblearn package -> SMOTE. This parameter can also accept None if resampling tuning on Layer 2 training data is not desired.")
        else:
            apply_resample_l2=True

        if not isinstance(X,(pd.DataFrame,np.ndarray)):
            raise ValueError("Parameter 'X' is not a Pandas Dataframe or a numpy.ndarray datatype!")

        if not isinstance(y,(pd.Series,np.ndarray)):
            raise ValueError("Parameter 'y' is not a Pandas Series or a numpy.ndarray datatype!")

        if not (isinstance(self.custom_distance_threshold,(int,float)) or pd.isna(self.custom_distance_threshold)):
            raise ValueError("Parameter 'custom_distance_threshold' is not an integer or float datatype. This parameter can also accept None if the originally computed intra-cluster distance is to be retained.")

        if not isinstance(self.precedence_layer,str):
            raise ValueError("Parameter 'precedence_layer' was not provided a string datatype.")

        if isinstance(y, pd.Series):
            y = y.values

        self.clusters, self.original_distance = self.clustering(X,self.custom_distance_threshold)

        self.grey_ind_arr, self.clustering_status = self.three_way_grouping(self.clusters, y)

        # Since grey_ind_arr variable contains indices of observations belonging to either c_mix or c_min clusters, those observations are tagged as 1 for L1 predictive task
        # The observations belonging to the majority class can get their label flipped from 0 to 1 if they belong to a c_mix cluster.
        y_l1 = y.copy()
        y_l1[self.grey_ind_arr] = 1

        X_l2 = X.loc[self.grey_ind_arr, :]
        y_l2 = y[self.grey_ind_arr]

        if not apply_resample_l1:
            self.layer_1_training(X, y_l1)

        else:
            try:
                X_L1, y_L1 = self.smote_obj_l1.fit_resample(X, y_l1)
                self.layer_1_training(X_L1, y_L1)

            except ValueError as e:
                print(e)
                print("\n")
                print("Performing model fit of Layer 1 without resampling...")

                self.layer_1_training(X, y_l1)


        if not apply_resample_l2:
            self.layer_2_training(X_l2, y_l2)

        else:
            try:
                X_L2, y_L2 = self.smote_obj_l2.fit_resample(X_l2, y_l2)
                self.layer_2_training(X_L2, y_L2)

            except ValueError as e:
                print(e)
                print("\n")
                print("Performing model fit of Layer 2 without resampling...")

                self.layer_2_training(X_l2, y_l2)

        if self.cv_obj_l1_bool:
            self.final_layer1_model=self.cv_obj_l1.best_estimator_
        else:
            self.final_layer1_model=self.model_l1

        if self.cv_obj_l2_bool:
            self.final_layer2_model=self.cv_obj_l2.best_estimator_
        else:
            self.final_layer2_model=self.model_l2
        