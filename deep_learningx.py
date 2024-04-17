#!/usr/bin/env python
# coding: utf-8

# In[1]:


#*******************************************************************************************
 #
 #  File Name:  deep_learningx.py
 #
 #  File Description:
 #      This Python script, deep_learningx.py, contains Python functions for processing 
 #      deep learning models. Here is the list:
 #
 #      set_hyperparameters_dictionary
 #      return_neural_network_model
 #      return_best_neural_network_model_hyperparameters
 #
 #
 #  Date            Description                             Programmer
 #  ----------      ------------------------------------    ------------------
 #  04/11/2024      Initial Development                     Nicholas J. George
 #
 #******************************************************************************************/

import deep_learningx_constants as dlc

import keras_tuner as kt
import tensorflow as tf

from silence_tensorflow import silence_tensorflow
silence_tensorflow()


# In[2]:


CONSTANT_LOCAL_FILE_NAME = 'deep_learningx.py'


# In[3]:


#*******************************************************************************************
 #
 #  Function Name:  set_hyperparameters_dictionary
 #
 #  Function Description:
 #      This function sets the hyperparameters dictionary for optimizing 
 #      deep learning models.
 #
 #  Return Type: n/a
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dictionary
 #          parameters_dictionary
 #                          The parameter is the new parameter dictionary.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def set_hyperparameters_dictionary(parameters_dictionary):
    
    dlc.parameters_dictionary = parameters_dictionary


# In[4]:


#*******************************************************************************************
 #
 #  Function Name:  return_neural_network_model
 #
 #  Function Description:
 #      This function returns a neural network model for the optimization process.
 #
 #  Return Type: n/a
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  object  hp              The parameter is hyperparameter object for keras_tuner.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_neural_network_model(hp):

    input_features_integer = dlc.parameters_dictionary['input_features']

    neural_net_sequential_model = tf.keras.models.Sequential()

    activation_choice \
        = hp.Choice('activation', dlc.parameters_dictionary['activation_choice_list'])

    neural_net_sequential_model.add \
        (tf.keras.layers.Dense \
            (units = hp.Int('first_units', \
                            min_value = dlc.parameters_dictionary['first_node_min_value'], \
                            max_value = dlc.parameters_dictionary['first_node_max_value'], \
                            step = dlc.parameters_dictionary['first_step']), \
             activation = activation_choice, \
             input_dim = dlc.parameters_dictionary['input_features']))

    for index in range(hp.Int('num_layers', 
                              dlc.parameters_dictionary['min_layers'], 
                              dlc.parameters_dictionary['max_layers'])):

        neural_net_sequential_model.add \
            (tf.keras.layers.Dense \
                (units = hp.Int('units_' + str(index),
                                 min_value = dlc.parameters_dictionary['node_min_value'], 
                                 max_value = dlc.parameters_dictionary['node_max_value'], 
                                 step = dlc.parameters_dictionary['step']),
                                 activation = activation_choice))

    neural_net_sequential_model.add \
        (tf.keras.layers.Dense(units = 1, activation = dlc.parameters_dictionary['output_activation']))

    learning_rate_float \
            = hp.Float('learning_rate', 
                       min_value = dlc.parameters_dictionary['min_learning_rate'], 
                       max_value = dlc.parameters_dictionary['max_learning_rate'], 
                       sampling = dlc.parameters_dictionary['learning_sampling'])

    neural_net_sequential_model.compile \
            (loss = dlc.parameters_dictionary['loss'], 
             optimizer = dlc.parameters_dictionary['optimizer'], 
             metrics = [dlc.parameters_dictionary['metrics']])


    return neural_net_sequential_model


# In[5]:


#*******************************************************************************************
 #
 #  Function Name:  return_best_neural_network_model_hyperparameters
 #
 #  Function Description:
 #      This function returns the objective score, loss score, and hyperparameters
 #      for the optimal neural network based on the hyperparameters dictionary.
 #
 #  Return Type: dictionary
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  nparray x_train_scaled_nparray
 #                          The parameter is the scaled feature training data for the model.
 #  nparray x_test_scaled_nparray
 #                          The parameter is the scaled feature test data for the model.
 #  nparray y_train_nparray The parameter is the target training data for the model.
 #  nparray y_test_nparray  The parameter is the target test data for the model.
 #  integer max_epochs_integer
 #                          The parameter is the maximum number of epochs for training
 #                          the neural network model.
 #  integer hyperband_iterations_integer
 #                          The parameter is the number of iterations for the hyperband
 #                          tuning.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_best_neural_network_model_hyperparameters \
        (x_train_scaled_nparray, \
         x_test_scaled_nparray, \
         y_train_nparray, \
         y_test_nparray,
         max_epochs_integer,
         hyperband_iterations_integer):

    tuner_hyperband \
            = kt.Hyperband \
                (return_neural_network_model,
                 objective \
                     = kt.Objective \
                         (dlc.parameters_dictionary['objective'], 
                          direction = dlc.parameters_dictionary['objective_direction']),
                 max_epochs = max_epochs_integer,
                 hyperband_iterations = hyperband_iterations_integer,
                 overwrite = True)

    tuner_hyperband.search \
            (x_train_scaled_nparray,
             y_train_nparray,
             epochs = max_epochs_integer,
             validation_data = (x_test_scaled_nparray, y_test_nparray))


    best_model = tuner_hyperband.get_best_models()[0]
            
    best_hyperparameters = tuner_hyperband.get_best_hyperparameters()[0]
        
    best_model_loss_float, best_model_objective_float \
        = best_model.evaluate(x_test_scaled_nparray, y_test_nparray, verbose = 2)

            
    best_hyperparameters_dictionary \
        = {'objective': best_model_objective_float,
           'loss': best_model_loss_float,
           'hyperparameters': best_hyperparameters.values}
            

    return best_hyperparameters_dictionary


# In[ ]:




