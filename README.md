![student_loan](https://github.com/njgeorge000158/Student-Loan-Repayment-with-Deep-Learning/assets/137228821/d5a521ef-5abc-4f46-9813-14499c90ba08)

----

# **Student Loan Repayment Prediction with Deep Learning**

## **Overview of the Analysis**

The purpose of this analysis is to create a regression model using deep learning techniques to predict if a particular company will succeed in designating the appropriate credit rating for a student loan applicant. The model draws on a dataset of 1,599 applicants that have received funding.

## **Results**

### Data Preprocessing

- The variable, credit_ranking, is the target of the regression model.

- The variables – payment_history, location_parameter, stem_degree_score, gpa_ranking, alumni_success, study_major_code, time_to_completion, finance_workshop_score, cohort_ranking, total_loan_score, financial_aid_score – are the features of the model.

### Compiling, Training, and Evaluating the Model

- To achieve the target performance, I configured the neural network configuration based on the results of an optimization script, student_loans_optimization_search.ipynb.

<img width="483" alt="Screenshot 2024-04-17 at 1 24 19 PM" src="https://github.com/njgeorge000158/Student-Loan-Repayment-Prediction-with-Deep-Learning/assets/137228821/e53a7e31-2db4-4085-ba42-3564a256b89f">

- The model has an input layer, a hidden layer, and the output layer with two dropouts.  The input layer and hidden layer consist of 98 and 45 neurons, respectively, and use tanh activation functions.  Because this is a regression model, the output layer has 1 neuron and uses a linear activation function.  The structure maintains the ability to learn patterns effectively while striking a balance between complexity and overfitting.

<img width="697" alt="Screenshot 2024-04-17 at 1 25 40 PM" src="https://github.com/njgeorge000158/Student-Loan-Repayment-Prediction-with-Deep-Learning/assets/137228821/809e03f6-c5d9-446f-bdeb-942a200ae2ba">

<img width="831" alt="Screenshot 2024-04-17 at 1 17 34 PM" src="https://github.com/njgeorge000158/Student-Loan-Repayment-Prediction-with-Deep-Learning/assets/137228821/8efa395d-237e-407c-906b-49635fc67a53">

Once implemented, the optimized model attained a mean squared error (mse) and loss of 33.3%.

<img width="1011" alt="Screenshot 2024-04-17 at 1 19 07 PM" src="https://github.com/njgeorge000158/Student-Loan-Repayment-Prediction-with-Deep-Learning/assets/137228821/81925ad2-9453-4f08-b60b-c6996c2870d4">

When I compared the a sample of the actual values to the model predictions, I observed that the values where either the same or off by one.

<img width="161" alt="Screenshot 2024-04-17 at 1 31 48 PM" src="https://github.com/njgeorge000158/Student-Loan-Repayment-Prediction-with-Deep-Learning/assets/137228821/0f4236b5-8b69-42f4-9af0-766f053aecc1">

## **Summary**

Overall, the sequential neural network model successfully predicts the student loan applicant credit ranking with a mean square error of 33.3%, which translates into the actual score or an off-by-one score.

----

### Copyright

Nicholas J. George © 2024. All Rights Reserved.
