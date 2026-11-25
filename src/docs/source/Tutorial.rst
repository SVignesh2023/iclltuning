ICLL Working Tutorial
=====================

This page illustrates how to use the **iclltuning** library by an short tutorial.

For the purposes of this tutorial, we shall be using the **Bank Term Deposit Predictions** dataset. A link to the same is available here: `Bank Term Deposit Predictions <https://www.kaggle.com/datasets/thedevastator/bank-term-deposit-predictions>`_.

.. note::

    This tutorial demonstrates how to use the **iclltuning library** and is **NOT intended to evaluate the ICLL model performance**. Therefore, the reported scores may be low because no extensive feature engineering, feature selection, or careful hyperparameter tuning was performed. Only basic data preprocessing was applied, and the model was trained solely to illustrate the library's workflow and usage.

First, lets load in the necessary packages:

.. code-block:: python

    import pandas as pd
    import numpy as np
    from sklearn.model_selection import GridSearchCV,StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from imblearn.over_sampling import SMOTE

    from iclltuning import iclltuning

Now, let us load the dataset. 

The target variable in this dataset is present as 'yes'/'no' format. We convert this into 1/0 format for ease of model training.

.. code-block:: python

    train_df=pd.read_csv(r"D:\Code_files\Bank_Term_Deposit\train.csv",low_memory=False,delimiter=',')
    train_df['y']=np.where(train_df['y']=='no',0,1)


View the top five rows of the dataset:

.. code-block:: text

        age           job  marital  education default  balance housing loan  contact  day month  duration  campaign  pdays  previous poutcome  y
    0   58    management  married   tertiary      no     2143     yes   no  unknown    5   may       261         1     -1         0  unknown  0
    1   44    technician   single  secondary      no       29     yes   no  unknown    5   may       151         1     -1         0  unknown  0
    2   33  entrepreneur  married  secondary      no        2     yes  yes  unknown    5   may        76         1     -1         0  unknown  0
    3   47   blue-collar  married    unknown      no     1506     yes   no  unknown    5   may        92         1     -1         0  unknown  0
    4   33       unknown   single    unknown      no        1      no   no  unknown    5   may       198         1     -1         0  unknown  0


This dataset is a moderately imbalanced dataset as **11.7%** of the instances belong to the positive class.

.. code-block:: python
    
    (train_df['y'].value_counts(normalize=True)*100).round(1).astype(str)+'%'


Output::

            y
    0    88.3%
    1    11.7%
    Name: proportion, dtype: object


Let us perform some basic cleaning of the dataset, mainly to convert categorical values to numerical values.

.. code-block:: python

    def convertData(func_df):
        func_df['marital']=np.where(func_df['marital']=='married',1,0)
        func_df['default']=np.where(func_df['default']=='yes',1,0)
        func_df['housing']=np.where(func_df['housing']=='yes',1,0)
        func_df['loan']=np.where(func_df['loan']=='yes',1,0)
        func_df['poutcome']=np.where(func_df['poutcome']=='success',1,0)

        return func_df

    # Select only the following features for model training
    selected_columns=[
        'age','marital','default',
        'housing','loan','day',
        'campaign','pdays','previous','poutcome'
    ]

    X_train=train_df[selected_columns].copy()
    y_train=train_df['y'].copy()

    # Call the function to transform the data
    X_train=convertData(X_train)


This tutorial will cover the following combinations of parameter initializations:

.. list-table::
   :header-rows: 1
   :widths: 25 25 20 20 20

   * - Layer Name
     - Model
     - Cross-Validation
     - Resampling
     - Sample Weights
   * - Layer 1 (**L1**)
     - :py:obj:`RandomForestClassifier`
     - ✅
     - ❌
     - ✅
   * - Layer 2 (**L2**)
     - :py:obj:`LogisiticRegression`
     - ❌
     - ✅
     - ❌

The :py:obj:`RandomForestClassifier` model will be used for the classification task in **Layer 1**. The :py:obj:`LogisiticRegression` Model will be used for classification task in **Layer 2**.

We shall **NOT tune the probability threshold of the** :py:obj:`LogisiticRegression` **model**, as this is not the purpose of this tutorial. So. we will be sticking with the **default probability threshold of 0.5**.

.. code-block:: python

    rf_model=RandomForestClassifier(class_weight='balanced')
    lr_model=LogisticRegression(penalty='elasticnet',l1_ratio=0.5,class_weight='balanced',random_state=0,solver='saga',max_iter=10000)


We shall use :py:obj:`GridSearchCV` object to find the optimal hyperparameter configurations of the :py:obj:`RandomForestClassifier` model applied on **Layer 1**.

.. code-block:: python

    # Define the Hyperparameter Grid for the Random Forest classifier
    rf_param_grid={
        'n_estimators' : [10,50,100],
        'max_depth':[3,5,7],
        'max_features' : ['sqrt',None]
    }

    # Define K-Fold object
    skf_obj=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)

    # Initialize Grid Search CV object
    cv_obj=GridSearchCV(
        estimator=rf_model,
        param_grid=rf_param_grid,
        scoring='recall',
        refit=True,
        n_jobs=-1,
        cv=skf_obj
    )


Define a :py:obj:`SMOTE` object to perform resampling on **Layer 2**.

.. code-block:: python

    sm_obj = SMOTE(random_state=0)


Define a sample weights to assign to the training instances in **Layer 1**.

.. code-block:: python

    # Set weights
    smpl_weights=y_train.copy()

    # Optimize the model for a hypothetical situation where the population psoitive class ratio is actually 5%.
    smpl_weights=np.array(np.where(smpl_weights==1,0.05/0.11,(1-0.05)/(1-0.11)))


Now, we shall finally initialize our ICLL model:

.. code-block:: python

    # Initialize the ICLL model
    icll_model=iclltuning(
        model_l1=None, # No model is requeired in Layer 1 as it is already present in the GridSearchCV object
        model_l2=lr_model, # Logistic Regression Model for Layer 2
        cv_obj_l1=cv_obj, # GridSearchCV object to perform hyperparameter tuning on Random Forest model in Layer 1
        cv_obj_l2=None, # No hyperparameter tuning required for Layer 2
        smote_obj_l1=None, # No Resampling for Layer 1
        smote_obj_l2=sm_obj, # Resampling to be applied to Layer 2
        sample_wt_l1=smpl_weights, # Assign sample weights in training Layer 1 model
        sample_wt_l2=None, # No sample weights to be assigned for training Layer 2 model
        custom_distance_threshold=None # Let ICLL model compute the optimal threshold to construct flat clusters
    )

    # Fit ICLL model to the training data
    icll_model.fit(X_train,y_train)

Now, that the ICLL model is trained, let us make predictions on our test data.

Load in the testing data:

.. code-block:: python

    # Read the test data
    test_df=pd.read_csv(r"D:\Code_files\Bank_Term_Deposit\test.csv",low_memory=False,delimiter=',')

    # Convert the target label to 1/0 format
    test_df['y']=np.where(test_df['y']=='no',0,1)

    # Transform the data
    X_test=test_df[selected_columns].copy()
    y_test=test_df['y'].copy()

    X_test=convertData(X_test)

Use :py:meth:`predict` to have ICLL model give the hard classification label for each instance.

.. code-block:: python

    icll_model.predict(X_test)


Similiarly, use :py:meth:`predict_proba` to have ICLL model give the soft classification label (probability) for each test instance.

.. code-block:: python

    icll_model.predict_proba(X_test)


**Output:**

.. code-block:: text

    array([0.3350838 , 0.0595488 , 0.09453571, ..., 0.03996633, 0.18961206,
       0.09201792])


With the :py:obj:`iclltuning` object in place, there are multiple other methods that we can access.

1. Since a :py:obj:`GridSearchCV` object was used in **Layer 1**, we can obtain the fitted object, and access its methods as you normally would.

.. code-block:: python

    fitted_cv_obj=icll_model.get_fitted_cv_layer_object('L1') # Supply 'L1' as GridSearchCV object was used in Layer 1.

    cv_fit_results=pd.DataFrame(fitted_cv_obj.cv_results_)
    cv_fit_results

**Output:**

.. code-block:: text

        mean_fit_time  std_fit_time  mean_score_time  std_score_time  param_max_depth param_max_features  param_n_estimators                                                         params  split0_test_score  split1_test_score  split2_test_score  split3_test_score  split4_test_score  mean_test_score  std_test_score  rank_test_score
    0       0.296676      0.048235         0.027802        0.014005                3               sqrt                  10   {'max_depth': 3, 'max_features': 'sqrt', 'n_estimators': 10}           0.854073           0.825884           0.854540           0.839122           0.835981         0.841920        0.011020               10
    1       0.986788      0.142360         0.037510        0.000497                3               sqrt                  50   {'max_depth': 3, 'max_features': 'sqrt', 'n_estimators': 50}           0.832581           0.846441           0.833671           0.831646           0.833801         0.835628        0.005463               13
    2       1.800487      0.134114         0.069442        0.008968                3               sqrt                 100  {'max_depth': 3, 'max_features': 'sqrt', 'n_estimators': 100}           0.827908           0.844572           0.828687           0.840212           0.828972         0.834070        0.006942               15
    3       0.504022      0.092023         0.016963        0.003834                3               None                  10     {'max_depth': 3, 'max_features': None, 'n_estimators': 10}           0.860146           0.862327           0.863417           0.861548           0.863396         0.862167        0.001230                3
    4       2.157512      0.223643         0.039396        0.004407                3               None                  50     {'max_depth': 3, 'max_features': None, 'n_estimators': 50}           0.858433           0.862327           0.863261           0.865909           0.863707         0.862727        0.002448                2

2. Since a :py:obj:`GridSearchCV` object was used in **Layer 1**, we can obtain the total time taken in seconds to perform hyperparameter tuning by using the :py:meth:`get_layer_search_time` method.

   | Similarly, we can obtain the time taken to train the :py:obj:`LogisticRegression` model in **Layer 2** in the same fashion.

.. code-block:: python
    
    
    t1=icll_model.get_layer_search_time('L1') # Supply 'L1' to get the time taken to perform hyperparameter tuning in Layer 1.
    t2=icll_model.get_layer_search_time('L2') # Supply 'L2' to get the time taken to train the Logistic Regression model in Layer 2.

    print(f"Time taken to perform hyperparameter tuning on Random Forest model: {np.round(t1,2)} seconds.")
    print(f"Time taken for Logistic Regression model to train: {np.round(t2/60,2)} minutes.")

**Output:**

.. code-block:: text

    Time taken to perform hyperparameter tuning on Random Forest model: 29.31 seconds.
    Time taken for Logistic Regression model to train: 1.41 minutes.


3. To get the distance threshold at which flat clusters were obtained, use the :py:meth:`get_computed_dist_threshold` method:

.. code-block:: python

    icll_model.get_computed_dist_threshold()

**Output:**

.. code-block:: text

    np.float64(1.7534645676717928)

Now with this information, if you want to obtain a different set of clusters and subsequently retrain the model, you can supply the value of the parameter :py:attr:`custom_distance_threshold`.


4. To get the count of each type of clusters formed, simply call the :py:meth:`get_cluster_type_info` method:

.. code-block:: python

    cluster_type_info=icll_model.get_cluster_type_info()
    print(cluster_type_info)


**Output:**

.. code-block:: text

    {'MAJORITY_CLUSTER_COUNT': 3124, 'MINORITY_CLUSTER_COUNT': 633, 'MIXED_CLUSTER_COUNT': 2250}

5. We can use the :py:meth:`get_fitted_model_layer_object` to access the methods of a fitted model in either layer.

   | Since a :py:obj:`GridSearchCV` object was used in **Layer 1**, using :py:meth:`get_fitted_model_layer_object` with *'L1'* returns the model stored in :py:meth:`best_estimator_`.
   |
   | The :py:meth:`best_estimator_` method houses the estimator selected by the search—highest score or smallest loss on the left-out data.

.. code-block:: python

    # Get the best estimator as decided by GridSearchCV object for Layer 1 by supplying 'L1'
    best_rf_model=icll_model.get_fitted_model_layer_object('L1')

    # Get the feature importance
    feature_importance_list=best_rf_model.feature_importances_

    importance_info_df=pd.DataFrame({'Feature Name': selected_columns,'Importance':feature_importance_list})
    print(importance_info_df)

**Output:**

.. code-block:: text

        Feature Name  Importance
    0          age    0.035334
    1      marital    0.000000
    2      default    0.000000
    3      housing    0.011766
    4         loan    0.000000
    5          day    0.000048
    6     campaign    0.421679
    7        pdays    0.476379
    8     previous    0.000000
    9     poutcome    0.054794

Now, to access the trained :py:obj:`LogisticRegression` model in **Layer 2** by supplying *'L2'* as the input.

.. code-block:: python

    # Get the trained Logistic Regression model in Layer 2 by spplying 'L2'
    fitted_lr_model=icll_model.get_fitted_model_layer_object('L2')

    coeff_list=fitted_lr_model.coef_[0]
    coef_df=pd.DataFrame({'Feature Name': selected_columns, 'Model Coefficients':coeff_list})

.. code-block:: text

        Feature Name  Model Coefficients
    0          age            0.007315
    1      marital           -0.822288
    2      default           -0.831480
    3      housing           -1.197592
    4         loan           -1.430569
    5          day           -0.005099
    6     campaign           -0.062564
    7        pdays            0.004941
    8     previous            0.271866
    9     poutcome            1.196456

**This concludes the end of this tutorial.**
