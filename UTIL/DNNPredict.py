#!/usr/bin/env python
# coding: utf-8

# In[10]:


from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score


# In[11]:


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


# In[12]:


def get_features_by_prefixes(group_combination, features):

    #group_combination = eval(group_combination.replace("/", ","))
    result = []
    for group in group_combination:
        result.extend(features[group])
    return result


# In[13]:


def select_features_from_joblib(joblib_path, field_list):
    """
    Given a joblib file and a list of selected fields, extracts and returns the data 
    with only the selected fields in a format suitable for model prediction.

    Parameters:
        joblib_path (str): Path to the joblib file.
        field_list (list): List of field names to select.

    Returns:
        pandas.DataFrame: A DataFrame containing only the selected fields.
    """
    data = joblib.load(joblib_path)
    
    if not isinstance(data, (list, pd.DataFrame)):
        raise ValueError("Expected joblib data to be a list of dictionaries or a pandas DataFrame.")
    
    if isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        df = data
    
    selected_data = df[field_list] if all(field in df.columns for field in field_list) else df[df.columns.intersection(field_list)]
    
    return selected_data


# In[14]:


def predict_with_keras(keras_model_path, test_dataset, features, target, return_y_proba):

    df = joblib.load(test_dataset)
    feature_groups = get_feature_groups(df)

    features_list = get_features_by_prefixes(features, feature_groups)

    model = load_model(keras_model_path)

    x_test = select_features_from_joblib(test_dataset, features_list)
    y_test = select_features_from_joblib(test_dataset, [target])
    y_pred_proba = model.predict(x_test)

    y_pred = y_pred_proba.argmax(axis=1)
    y_true = y_test
    
    if return_y_proba:
        return {"y_proba": y_pred_proba, "y_pred": y_pred, "y_true": y_true, "accuracy": accuracy_score(y_true, y_pred), "features": features }
    else:
        return accuracy_score(y_true, y_pred)
      
    


# In[15]:


import base64
import re
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
    
def extract_and_decode_model_path(model_path: str) -> str:
    """
    Extrai a parte codificada em Base64 do caminho do modelo e a decodifica.
    :param model_path: O caminho completo do modelo.
    :return: A string decodificada ou uma mensagem de erro se não encontrar.
    """
    match = re.search(r'/(DNN|CNN)_([a-zA-Z0-9+/=]+)\.keras$', model_path)
    if match:
        base64_part = match.group(2)
        return format_string_for_eval(decode_base64(base64_part))
    return "Nenhuma string Base64 encontrada no caminho."


# In[16]:


def DNNPredict(model_path, test_dataset, target, return_y_proba=True):  
    features = extract_and_decode_model_path(model_path)
    return predict_with_keras(model_path, test_dataset, features,target, return_y_proba)
    


# In[18]:


#keras_model_path = '/teamspace/studios/this_studio/DNN_MODEL_TRAINING/models/DNN_cm1zL21mY2MyMC9tZmNjNDAvdG90YWxfZW5lcmd5.keras'
#test_dataset = "/teamspace/studios/this_studio/DNN_val2.joblib"
#target = "emotion"
#DNNPredict(keras_model_path, test_dataset, target, False)


# In[ ]:





# In[ ]:





# In[ ]:




