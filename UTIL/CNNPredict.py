#!/usr/bin/env python
# coding: utf-8

# In[98]:


from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score
import base64
import re
import h5py
import torch
import os
import sys
sys.path.append("/teamspace/studios/this_studio/")
from cnn_models import CNNModel3

# In[99]:


def get_feature_groups(df):
    feature_groups = {}

    for col in df.columns:
        if col.startswith("DNN_"):  # Filtra apenas colunas que começam com "DNN_"
            parts = col.split("_")
            if len(parts) > 2:  # Garante que há pelo menos um conjunto intermediário
                base_feature = "_".join(parts[1:-1])  # Pega todos os conjuntos entre o primeiro e o último
                
                if base_feature not in feature_groups:
                    feature_groups[base_feature] = []
                
                feature_groups[base_feature].append(col)
        elif col.startswith("CNN_"):
            feature_groups[col].append(col)

    return feature_groups


# In[100]:


def get_features_by_prefixes(group_combination, features):

    #group_combination = eval(group_combination.replace("/", ","))
    result = []
    for group in group_combination:
        result.extend(features[group])
    return result


# In[101]:


def select_features_from_hdf5(hdf5_path, feature_list):
    """
    Given an HDF5 file and a list of selected features, extracts and returns the data 
    with only the selected features in a format suitable for model prediction.

    Parameters:
        hdf5_path (str): Path to the HDF5 file.
        feature_list (list): List of feature names to select.

    Returns:
        np.ndarray: A 4D NumPy array (batch_size, channels, height, width) suitable for CNN input.
    """
    selected_data = []
    filenames = []
    
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        for file_name in hdf5_file.keys():
            if file_name.endswith(".wav"):  # Ensure we only process valid entries
                feature_arrays = []
                
                for feature in feature_list:
                    if feature in hdf5_file[file_name]:
                        dataset = hdf5_file[file_name][feature][:]  # Directly load as NumPy array
                        feature_arrays.append(dataset)
                    else:
                        raise ValueError(f"Feature '{feature}' not found in '{file_name}'")
                
                filenames.append(file_name)
                selected_data.append(np.stack(feature_arrays, axis=0))  # Stack along the channel dimension
    
    return np.array(selected_data)  # Return as 4D NumPy array (batch_size, channels, 167, 167)

def select_scalar_from_hdf5(hdf5_path, scalar_name):
    """
    Extracts scalar values (such as 'emotion') from an HDF5 file.
    
    Parameters:
        hdf5_path (str): Path to the HDF5 file.
        scalar_name (str): Name of the scalar feature to extract.

    Returns:
        np.ndarray: A NumPy array containing scalar values for each file.
    """
    scalar_values = []
    
    with h5py.File(hdf5_path, 'r') as hdf5_file:
        for file_name in hdf5_file.keys():
            if file_name.endswith(".wav"):  # Process only valid entries
                if scalar_name in hdf5_file[file_name]:
                    scalar_values.append(hdf5_file[file_name][scalar_name][()])  # Extract scalar value
                else:
                    raise ValueError(f"Scalar '{scalar_name}' not found in '{file_name}'")
    
    return np.array(scalar_values)  # Convert to NumPy array


# In[102]:


def predict_with_torch(model_path, test_dataset, features, target, return_y_proba):    
    model = CNNModel3((len(features), 167, 167), num_classes=8, learning_rate=0.0005)   
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval() 
    
    x_test = select_features_from_hdf5(test_dataset, features)
    y_test = select_scalar_from_hdf5(test_dataset, target)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)  
    with torch.no_grad():
        y_pred_proba = model(x_test_tensor).numpy()
        
    y_pred = y_pred_proba.argmax(axis=1)
    y_true = y_test
    
    if return_y_proba:
        return {"y_proba": y_pred_proba, "y_pred": y_pred, "y_true": y_true, "accuracy": accuracy_score(y_true, y_pred), "features": features }
    else:
        val2_accuracy = accuracy_score(y_true, y_pred)      
    


# In[103]:


def decode_base64(encoded_str: str) -> str:
    """
    Decodifica uma string codificada em Base64.
    :param encoded_str: A string codificada em Base64.
    :return: A string decodificada.
    """
    try:
        decoded_bytes = base64.b64decode(encoded_str)
        return decoded_bytes.decode('utf-8')
    except Exception as e:
        return f"Erro ao decodificar: {e}"

def format_string_for_eval(s: str) -> str:
    # Adiciona aspas em torno de cada palavra
    s = s.replace("/", ",")
    formatted = re.sub(r'([a-zA-Z0-9_]+)', r'"\1"', s)
    return eval("[" + formatted + "]")   


# In[104]:


def CNNPredict(model_path, test_dataset, target, return_y_proba=True):    
    features = format_string_for_eval(decode_base64(os.path.splitext(os.path.basename(model_path))[0]))
    return predict_with_torch(model_path, test_dataset, features, target, return_y_proba)


# In[106]:


#model_path = "/teamspace/studios/this_studio/CNN_MODEL_TRAINING/models/Q05OX0NRVF9TcGVjdHJvZ3JhbS9DTk5fTUZDQ193aXRoX0RlbHRhcw==.pth"
#test_dataset = "/teamspace/studios/this_studio/CNN_val2.h5"
#features = format_string_for_eval(decode_base64(os.path.splitext(os.path.basename(model_path))[0]))
#target = "emotion"
#CNNPredict(model_path, test_dataset, target)


# In[ ]:





# In[ ]:




