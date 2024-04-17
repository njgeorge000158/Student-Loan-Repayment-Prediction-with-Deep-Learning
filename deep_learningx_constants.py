#!/usr/bin/env python
# coding: utf-8

# In[1]:


#*******************************************************************************************
 #
 #  File Name:  deep_learningx_constants.py
 #
 #  File Description:
 #      This Python script, deep_learningx_constants, contains generic Python constants
 #      for deep learning models.
 #
 #
 #  Date            Description                             Programmer
 #  ----------      ------------------------------------    ------------------
 #  04/17/2024      Initial Development                     Nicholas J. George
 #
 #******************************************************************************************/


# In[2]:


CONSTANT_LOCAL_FILE_NAME = 'deep_learningx_constants.py'


# In[3]:


parameters_dictionary \
    = {'activation_choice_list': ['relu'],
       'input_features': 10,
       'objective': 'val_accuracy',
       'objective_direction': 'max',
       'first_node_min_value': 1,
       'first_node_max_value': 100,
       'first_step': 1,
       'node_min_value': 1,
       'node_max_value': 99,
       'step': 1,
       'min_layers': 0,
       'max_layers': 5,
       'min_learning_rate': 1e-3,
       'max_learning_rate': 1e-3,
       'learning_sampling': 'linear',
       'output_activation': 'sigmoid',
       'loss': 'binary_crossentropy',
       'optimizer': 'adam',
       'metrics': 'accuracy'}


# In[ ]:




