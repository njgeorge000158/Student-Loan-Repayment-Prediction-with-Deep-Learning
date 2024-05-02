# Student-Loan-Repayment-Prediction-with-Deep-Learning

----

## Table of Contents (student_loans_colab_colab.ipynb)

----

# <br><br> **Section 1: Extraction**
> ## <br> **1.1: Read the CSV data into a Pandas DataFrame**
> ## <br> **1.2: Display Student Loan DataFrame**
# <br><br> **Section 2: Preprocessing**
> ## <br> **2.1: Create the labels series (y) from the “spam” column, and then create the features (X) DataFrame from the remaining columns.**
>> ### **Separate the Y Variable, The Labels**
>> ### **Separate the X Variable, the Features**
> ## <br> **2.2: Split the Data into Training and Testing Datasets by Using train_test_split.**
> ## <br> **2.3: Use the StandardScaler to Scale the X Variables**
>> ### **Create a StandardScaler Instance**
>> ### **Fit the StandardScaler**
>> ### **Scale the Data**
# <br><br> **Section 3: Compile, Train, Evaluate, and Export the Model**
> ## <br> **3.1: Compile Model**
>> ### **Model Definition**
>> ### **Instantiate the Model**
>> ### **Layers**
>> ### **Model Summary**
>> ### **Compile**
> ## <br> **3.2: Fit and Train Model**
> ## <br> **3.3: Evaluate Model**
> ## <br> **3.4: Save and Export Model**
# <br><br> **Section 4: Predict Loan Repayment Success**
> ## <br> **4.1: Reload Model**
> ## <br> **4.2: Predictions**
> ## <br> **4.3: Compare Predictions and Actual Values**

----

## Table of Contents (student_loans_hyperparameters_optimization_colab.ipynb)

----

# <br><br> **Section 1: Extraction and Transformation**
> ## <br> **1.1: Read the CSV data into a Pandas DataFrame**
> ## <br> **1.2: Display Spam DataFrame**
> ## <br> **1.3: Create the labels series (y) from the “spam” column, and then create the features (X) DataFrame from the remaining columns.**
>> ### **Separate the Y Variable, The Labels**
>> ### **Review the Y Series**
>> ### **Check the Balance of the Labels Variable (y) by Using the value_counts Function.**
>> ### **Separate the X Variable, the Features**
>> ### **Review the X DataFrame**
> ## <br> **1.4: Split the Data into Training and Testing Datasets by Using train_test_split.**
> ## <br> **1.5: Use the StandardScaler to Scale the X Variables**
>> ### **Scale Training and Test Data as Numpy Arrays**
>> ### **Create Scaled X Variable DataFrames**
>> ### **Display Scaled Training and Testing Data**
# <br><br> **Section 2: Undersampled and OverSampled Spam Data**
> ## <br> **2.1: Instantiate the Random Undersampler Instance**
> ## <br> **2.2: Instantiate the Random Oversampler Instance**
> ## <br> **2.4: Instantiate the SMOTE Instance**
> ## <br> **2.5: Instantiate the SMOTEENN Instance**
> ## <br> **2.6: Check the Balance of the Labels Variable (y) by Using the value_counts Function.**
> ## <br> **2.7: Display Normalized Resampled Training and Testing Data**
# <br><br> **Section 3: Model Optimization**
> ## <br> **3.1: Logistic Regression**
>> ### **Original**
>> ### **Random Undersampling**
>> ### **Random Oversampling**
>> ### **Cluster Centroids**
>> ### **SMOTE**
>> ### **SMOTEEN**
> ## <br> **3.2: Decision Tree**
>> ### **Original**
>> ### **Random Undersampling**
>> ### **Random Oversampling**
>> ### **Cluster Centroids**
>> ### **SMOTE**
>> ### **SMOTEEN**
> ## <br> **3.3: Random Forest**
>> ### **Original**
>> ### **Random Undersampling**
>> ### **Random Oversampling**
>> ### **Cluster Centroids**
>> ### **SMOTE**
>> ### **SMOTEEN**
> ## <br> **3.4: Support Vector Machine (SVM)**
>> ### **Original**
>> ### **Random Undersampling**
>> ### **Random Oversampling**
>> ### **Cluster Centroids**
>> ### **SMOTE**
>> ### **SMOTEEN**
> ## <br> **3.5: K-Nearest Neighbor (KNN)**
>> ### **Original**
>> ### **Random Undersampling**
>> ### **Random Oversampling**
>> ### **Cluster Centroids**
>> ### **SMOTE**
>> ### **SMOTEEN**
# <br><br> **Section 4: Save Grid Search Models To Files**
> ## <br> **4.1: Logistic Regression**
>> ### **Random Undersampling**
>> ### **Random Oversampling**
>> ### **Cluster Centroids**
>> ### **SMOTE**
>> ### **SMOTEEN**
> ## <br> **4.2: Decision Tree**
>> ### **Random Undersampling**
>> ### **Random Oversampling**
>> ### **Cluster Centroids**
>> ### **SMOTE**
>> ### **SMOTEEN**
> ## <br> **4.3: Random Forest**
>> ### **Random Undersampling**
>> ### **Random Oversampling**
>> ### **Cluster Centroids**
>> ### **SMOTE**
>> ### **SMOTEEN**
> ## <br> **4.4: Support Vector Machine (SVM)**
>> ### **Random Undersampling**
>> ### **Random Oversampling**
>> ### **Cluster Centroids**
>> ### **SMOTE**
>> ### **SMOTEEN**
> ## <br> **4.5: K-Nearest Neighbor (KNN)**
>> ### **Random Undersampling**
>> ### **Random Oversampling**
>> ### **Cluster Centroids**
>> ### **SMOTE**
>> ### **SMOTEEN**

----

## Copyright

Nicholas J. George © 2023. All Rights Reserved.
