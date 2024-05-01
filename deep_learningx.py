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
 #      return_nn_sequential_model
 #      return_keras_tuner
 #      return_best_nn_sequential_model_hyperparameters
 #
 #
 #  Date            Description                             Programmer
 #  ----------      ------------------------------------    ------------------
 #  04/11/2024      Initial Development                     Nicholas J. George
 #
 #******************************************************************************************/

import keras_tuner as kt
import tensorflow as tf

from silence_tensorflow import silence_tensorflow
silence_tensorflow()


# In[2]:


CONSTANT_LOCAL_FILE_NAME = 'deep_learningx.py'


# In[3]:


HYPERPARAMETERS_DICTIONARY \
    = {'tuner_type': 'hyperband',
       'best_model_count': 5,
       'hyperband_iterations': 2,
       'patience': 100,
       'max_epochs': 1000,
       'restore_best_weights': True,
       'activation_choice_list': ['relu'],
       'input_features': 10,
       'objective': 'val_accuracy',
       'objective_direction': 'max',
       'input_layer_units_range': (100, 100),
       'input_units_step': 1,
       'hidden_layers': 5,
       'hidden_layer_units_range_list': \
           [(50, 50), (40, 40), (30, 30), (20, 20), (10, 10)],
       'hidden_units_step': 1,
       'min_learning_rate': 25e-4,
       'max_learning_rate': 25e-4,
       'learning_rate_step': 1e-6,
       'learning_sampling': 'linear',
       'output_activation_choice_list': ['sigmoid'],
       'output_layer_units': 1,
       'loss': 'binary_crossentropy',
       'optimizer': 'adam',
       'metrics': 'accuracy'}


# In[4]:


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
 #          hyperparameters_dictionary
 #                          The parameter is the new hyperparameters dictionary.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def set_hyperparameters_dictionary(hyperparameters_dictionary):

    global HYPERPARAMETERS_DICTIONARY
    
    HYPERPARAMETERS_DICTIONARY = hyperparameters_dictionary


# In[5]:


#*******************************************************************************************
 #
 #  Function Name:  return_nn_sequential_model
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

def return_nn_sequential_model(hp):

    input_features_integer = HYPERPARAMETERS_DICTIONARY['input_features']

    neural_net_sequential_model = tf.keras.models.Sequential()

    activation_choice \
        = hp.Choice('activation', HYPERPARAMETERS_DICTIONARY['activation_choice_list'])

    neural_net_sequential_model.add \
        (tf.keras.layers.Dense \
            (units = hp.Int('first_units', \
                            min_value = HYPERPARAMETERS_DICTIONARY['input_layer_units_range'][0], \
                            max_value = HYPERPARAMETERS_DICTIONARY['input_layer_units_range'][1], \
                            step = HYPERPARAMETERS_DICTIONARY['input_units_step']), \
             activation = activation_choice, \
             input_dim = HYPERPARAMETERS_DICTIONARY['input_features']))

    
    if HYPERPARAMETERS_DICTIONARY['hidden_layers'] != 0:
        
        for index in range(HYPERPARAMETERS_DICTIONARY['hidden_layers']):

            neural_net_sequential_model.add \
                (tf.keras.layers.Dense \
                    (units \
                         = hp.Int \
                             ('units_' + str(index + 1),
                              min_value \
                                  = HYPERPARAMETERS_DICTIONARY['hidden_layer_units_range_list'][index][0], 
                              max_value \
                                  = HYPERPARAMETERS_DICTIONARY['hidden_layer_units_range_list'][index][1], 
                              step = HYPERPARAMETERS_DICTIONARY['hidden_units_step']),
                              activation = activation_choice))         

    
    output_activation_choice \
        = hp.Choice('activation', HYPERPARAMETERS_DICTIONARY['output_activation_choice_list'])
    
    neural_net_sequential_model.add \
        (tf.keras.layers.Dense \
             (units = HYPERPARAMETERS_DICTIONARY['output_layer_units'], 
              activation = output_activation_choice))

    learning_rate_float \
            = hp.Float('learning_rate', 
                       min_value = HYPERPARAMETERS_DICTIONARY['min_learning_rate'], 
                       max_value = HYPERPARAMETERS_DICTIONARY['max_learning_rate'],
                       step = HYPERPARAMETERS_DICTIONARY['learning_rate_step'],
                       sampling = HYPERPARAMETERS_DICTIONARY['learning_sampling'])

    neural_net_sequential_model.compile \
            (loss = HYPERPARAMETERS_DICTIONARY['loss'], 
             optimizer = HYPERPARAMETERS_DICTIONARY['optimizer'], 
             metrics = [HYPERPARAMETERS_DICTIONARY['metrics']])


    return neural_net_sequential_model


# In[6]:


#*******************************************************************************************
 #
 #  Function Name:  return_keras_tuner
 #
 #  Function Description:
 #      This function returns the specified keras tuner.
 #
 #  Return Type: keras tuner
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  n/a     n/a             n/a
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_keras_tuner():

    keras_tuner = None

    if HYPERPARAMETERS_DICTIONARY['tuner_type'] == 'hyperband':
    
        keras_tuner \
            = kt.Hyperband \
                (return_nn_sequential_model,
                 objective \
                     = kt.Objective \
                         (HYPERPARAMETERS_DICTIONARY['objective'], 
                          direction = HYPERPARAMETERS_DICTIONARY['objective_direction']),
                 max_epochs = HYPERPARAMETERS_DICTIONARY['max_epochs'],
                 hyperband_iterations = HYPERPARAMETERS_DICTIONARY['hyperband_iterations'],
                 overwrite = True)

    elif HYPERPARAMETERS_DICTIONARY['tuner_type'] == 'grid_search':

        keras_tuner \
            = kt.GridSearch \
                (return_nn_sequential_model,
                 objective \
                     = kt.Objective \
                         (HYPERPARAMETERS_DICTIONARY['objective'], 
                          direction = HYPERPARAMETERS_DICTIONARY['objective_direction']),
                 overwrite = True)

    elif HYPERPARAMETERS_DICTIONARY['tuner_type'] == 'random_search':

        keras_tuner \
            = kt.RandomSearch \
                (return_nn_sequential_model,
                 objective \
                     = kt.Objective \
                         (HYPERPARAMETERS_DICTIONARY['objective'], 
                          direction = HYPERPARAMETERS_DICTIONARY['objective_direction']),
                 overwrite = True)

    elif HYPERPARAMETERS_DICTIONARY['tuner_type'] == 'bayesian_optimization':
        
        keras_tuner \
            = kt.BayesianOptimization \
                (return_nn_sequential_model,
                 objective \
                     = kt.Objective \
                         (HYPERPARAMETERS_DICTIONARY['objective'], 
                          direction = HYPERPARAMETERS_DICTIONARY['objective_direction']),
                 overwrite = True)

    return keras_tuner


# In[7]:


#*******************************************************************************************
 #
 #  Function Name:  return_best_nn_sequential_model_hyperparameters
 #
 #  Function Description:
 #      This function returns the objective score, loss score, and hyperparameters
 #      for the optimal neural network models based on the hyperparameters dictionary.
 #
 #  Return Type: list
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
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_best_nn_sequential_model_hyperparameters \
        (x_train_scaled_nparray, \
         x_test_scaled_nparray, \
         y_train_nparray, \
         y_test_nparray):

    current_keras_tuner = return_keras_tuner()

    earlystopping_callback \
        = tf.keras.callbacks.EarlyStopping  \
            (monitor = HYPERPARAMETERS_DICTIONARY['objective'],
             mode = HYPERPARAMETERS_DICTIONARY['objective_direction'],
             patience = HYPERPARAMETERS_DICTIONARY['patience'],
             restore_best_weights = HYPERPARAMETERS_DICTIONARY['restore_best_weights'])
            
    current_keras_tuner.search \
            (x_train_scaled_nparray,
             y_train_nparray,
             epochs = HYPERPARAMETERS_DICTIONARY['max_epochs'],
             validation_data = (x_test_scaled_nparray, y_test_nparray),
             callbacks = [earlystopping_callback])

            
    best_model = current_keras_tuner.get_best_models(HYPERPARAMETERS_DICTIONARY['best_model_count'])
            
    best_hyperparameters = current_keras_tuner.get_best_hyperparameters(HYPERPARAMETERS_DICTIONARY['best_model_count'])

    best_models_dictionary_list = []

            
    for i in range(HYPERPARAMETERS_DICTIONARY['best_model_count']):

        try:
        
            best_model_loss_float, best_model_objective_float \
                = best_model[i].evaluate(x_test_scaled_nparray, y_test_nparray, verbose = 2)

            print(best_model_loss_float, best_model_objective_float, best_hyperparameters)

            best_model_dictionary \
                = {'objective': best_model_objective_float,
                   'loss': best_model_loss_float,
                   'hyperparameters': best_hyperparameters[i].values}

            best_models_dictionary_list.append(best_model_dictionary)

        except:

            pass

            
    return best_models_dictionary_list


# In[ ]:




