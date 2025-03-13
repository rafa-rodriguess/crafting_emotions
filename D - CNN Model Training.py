import os
import torch
import base64
import json
import pandas as pd
import pytorch_lightning as pl
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, cohen_kappa_score
from cnn_models import CNNModel3  # Adicione outros modelos conforme necessário
from itertools import combinations

# Função para gerar combinações de features
def generate_combinations(lista, tamanho):
    resultado = []
    for i in range(1, tamanho + 1):
        resultado.extend(map(list, combinations(lista, i)))
    return resultado

# Classe para carregar o dataset com múltiplas features combinadas
class EmotionDataset(Dataset):
    def __init__(self, file_path, feature_names):
        self.data = []
        self.labels = []

        with h5py.File(file_path, 'r') as hdf:
            for group in hdf.values():
                # Verifica se todas as features da combinação estão presentes
                if all(feature in group for feature in feature_names):
                    # Extrai e empilha as features para formar os canais da CNN
                    features = [group[feature][()] for feature in feature_names]
                    combined_feature = np.stack(features, axis=0)  # (canais, altura, largura)
                    label = group["emotion"][()]

                    self.data.append(combined_feature)
                    self.labels.append(label)

        # Converter para arrays numpy
        self.data = np.array(self.data, dtype=np.float32)  # (N, canais, altura, largura)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = torch.tensor(self.data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label


# DataModule para PyTorch Lightning
class EmotionDataModule(pl.LightningDataModule):
    def __init__(self, train_file, val_file, test_file, feature_group, batch_size=64):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.feature_group = feature_group
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = EmotionDataset(self.train_file, self.feature_group)
        self.val_dataset = EmotionDataset(self.val_file, self.feature_group)
        self.test_dataset = EmotionDataset(self.test_file, self.feature_group)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

# Função principal para treinar os modelos CNN
def train_models(train_file, val_file, test_file, feature_groups, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics_dir = os.path.join(output_dir, "metrics")
    models_dir = os.path.join(output_dir, "models")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    feature_combinations = generate_combinations(feature_groups, len(feature_groups))
    
    # Hiperparâmetros
    batch_sizes = [64]
    epochs_list = [50]
    patience_list = [10]
    learning_rates = [0.0005]

    models = {
        "CNN-3": CNNModel3
    }
    
    for model_name, model_class in models.items():
        for feature in feature_combinations:
            print(f"\nFeature Combination: {feature}")
            # Gerar nome base codificado
            base_name = "/".join(feature)
            base_name_encoded = base64.urlsafe_b64encode(base_name.encode("utf-8")).decode("utf-8")

            for batch_size in batch_sizes:
                for epochs in epochs_list:
                    for patience in patience_list:
                        for learning_rate in learning_rates:
                            file_name = base_name_encoded
                            model_path = os.path.join(models_dir, file_name + ".pth")
                            metrics_path = os.path.join(metrics_dir, file_name + ".csv")

                            if os.path.exists(model_path) and os.path.exists(metrics_path):
                                print(f"Skipping {file_name}, already trained.")
                                continue
                            
                            print(f"Training {file_name}")
                            data_module = EmotionDataModule(train_file, val_file, test_file, feature, batch_size)
                            model = model_class((len(feature), 167, 167), num_classes=8, learning_rate=learning_rate)
                            
                            trainer = pl.Trainer(
                                max_epochs=epochs, 
                                accelerator='auto',
                                callbacks=[pl.callbacks.EarlyStopping(monitor='val_loss', patience=patience)]
                            )
                            
                            trainer.fit(model, datamodule=data_module)
                            torch.save(model.state_dict(), model_path)
                            print(f"Model saved to {model_path}")
                            
                            # Avaliação do modelo
                            trainer.test(model, datamodule=data_module)
                            test_results = trainer.callback_metrics
                            test_accuracy = test_results.get("test_acc", None)

                            # Coletar métricas detalhadas
                            y_pred, y_true, y_proba = [], [], []
                            for batch in data_module.test_dataloader():
                                x_batch, y_batch = batch
                                y_batch_pred = model(x_batch).softmax(dim=1)
                                y_pred.extend(y_batch_pred.argmax(dim=1).tolist())
                                y_true.extend(y_batch.tolist())
                                y_proba.extend(y_batch_pred.tolist())

                            val2_accuracy = accuracy_score(y_true, y_pred)
                            val2_recall = recall_score(y_true, y_pred, average='weighted')
                            val2_precision = precision_score(y_true, y_pred, average='weighted')
                            val2_f1 = f1_score(y_true, y_pred, average='weighted')
                            val2_confusion_matrix = confusion_matrix(y_true, y_pred).tolist()
                            val2_cohen_kappa = cohen_kappa_score(y_true, y_pred)
                            val2_error_indices = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
                            val2_accuracy_vector = [1 if t == p else 0 for t, p in zip(y_true, y_pred)]
                            gap = test_accuracy.item() - val2_accuracy if test_accuracy is not None else "N/A"

                            # Salvar métricas
                            metrics = {
                                "Model": model_name,
                                "Feature Group": feature,
                                "train_accuracy": test_results.get("train_acc", "N/A"),
                                "val_accuracy": test_results.get("val_acc", "N/A"),
                                "val2_accuracy": val2_accuracy,
                                "gap": gap,
                                "val2_recall": val2_recall,
                                "val2_precision": val2_precision,
                                "val2_f1": val2_f1,
                                "val2_model_path": model_path,
                                "val2_Confusion_Matrix": json.dumps(val2_confusion_matrix, separators=(',', ':')),
                                "val2_Cohen_Kappa_Score": val2_cohen_kappa,
                                "val2_error_indices": json.dumps(val2_error_indices, separators=(',', ':')),
                                "val2_accuracy_vector": json.dumps(val2_accuracy_vector, separators=(',', ':')),
                                "val2_y_pred": json.dumps(y_pred, separators=(',', ':')),
                                "val2_y_true": json.dumps(y_true, separators=(',', ':')),
                                "val2_y_proba": json.dumps(y_proba, separators=(',', ':')),  # Aqui está o problema
                            }

                            pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
                            print(f"Metrics saved to {metrics_path}")

# Chamando a função
feature_group =["CNN_CQT_Spectrogram",
                "CNN_Chromagram",
                "CNN_MFCC_with_Deltas",
                "CNN_Mel_Spectrogram",
                "CNN_Spectral_Contrast"]
train_models("CNN_train.h5", "CNN_val.h5", "CNN_val2.h5", feature_group, "CNN_MODEL_TRAINING")

import os
import pandas as pd

def consolidate_csv(directory: str, output_file: str):
    """
    Traverses a directory and its subdirectories to find all CSV files,
    consolidating them into a single destination file.
    
    Parameters:
    directory (str): Path to the root directory where CSV files are located.
    output_file (str): Path to the destination file where consolidated data will be saved.
    """
    all_dfs = []  # List to store temporary DataFrames
    
    # Walk through all directories and subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".csv"):  # Check if the file is a CSV
                file_path = os.path.join(root, file)
                try:
                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(file_path)
                    all_dfs.append(df)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    # Consolidate all DataFrames into a single one and save to the destination file
    if all_dfs:
        consolidated_df = pd.concat(all_dfs, ignore_index=True)
        consolidated_df.to_csv(output_file, index=False)
        print(f"Consolidation completed. File saved at: {output_file}")
    else:
        print("No CSV files found for consolidation.")

consolidate_csv("CNN_MODEL_TRAINING/metrics","CNN_MODEL_TRAINING/CNN_models_combination_metrics.csv")
