# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}

import pandas as pd
import numpy as np
import os
import gzip
import pickle
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score, confusion_matrix

INPUT_DIR = 'files/input/'
MODELS_DIR = 'files/models/'
OUTPUT_DIR = 'files/output/'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_data(df):
    """
    Realiza la limpieza de los datos según el Paso 1.
    """
    # Renombrar columna objetivo
    df = df.rename(columns={'default payment next month': 'default'})
    
    # Remover columna ID
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
        
    # Eliminar registros con información no disponible
    df = df.dropna()
    
    # Agrupar valores de EDUCATION > 4 en 4 ('others')
    df.loc[df['EDUCATION'] > 4, 'EDUCATION'] = 4
    
    return df

def main():
  
    try:
        train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train_data.csv.zip'), compression='zip')
        test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test_data.csv.zip'), compression='zip')
    except FileNotFoundError:
        # Fallback por si los nombres son genéricos csv
        train_df = pd.read_csv(os.path.join(INPUT_DIR, 'train.csv'))
        test_df = pd.read_csv(os.path.join(INPUT_DIR, 'test.csv'))

  
    train_df = clean_data(train_df)
    test_df = clean_data(test_df)


    x_train = train_df.drop(columns=['default'])
    y_train = train_df['default']
    
    x_test = test_df.drop(columns=['default'])
    y_test = test_df['default']


    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' 
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(random_state=42))
    ])

  
    param_grid = {
        'clf__n_estimators': [100, 200, 300],
        'clf__max_depth': [15, 25, None],
        'clf__min_samples_split': [2, 5, 10],
        'clf__min_samples_leaf': [1, 3, 5]
    }

    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=10, 
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(x_train, y_train)
    
    best_model = grid_search.best_estimator_


    model_path = os.path.join(MODELS_DIR, 'model.pkl.gz')
    with gzip.open(model_path, 'wb') as f:
        pickle.dump(grid_search, f)

    
    metrics_list = []

    def get_metrics(dataset_name, X, y, model):
        """
        Calcula y devuelve los diccionarios de métricas de resumen y matriz de confusión.
        """
        y_pred = model.predict(X)
        
        
        prec = precision_score(y, y_pred, zero_division=0)
        bal_acc = balanced_accuracy_score(y, y_pred)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
    
        metrics_dict = {
            'type': 'metrics',
            'dataset': dataset_name,
            'precision': float(round(prec, 4)),
            'balanced_accuracy': float(round(bal_acc, 4)),
            'recall': float(round(rec, 4)),
            'f1_score': float(round(f1, 4))
        }
        
  
        cm_dict = {
            'type': 'cm_matrix',
            'dataset': dataset_name,
            'true_0': {"predicted_0": int(tn), "predicted_1": int(fp)},
            'true_1': {"predicted_0": int(fn), "predicted_1": int(tp)}
        }
      
        return metrics_dict, cm_dict


    
    
    train_metrics, train_cm = get_metrics('train', x_train, y_train, grid_search)
    test_metrics, test_cm = get_metrics('test', x_test, y_test, grid_search)
    

    metrics_list = [
        train_metrics, 
        test_metrics, 
        train_cm, 
        test_cm
    ]

   
    output_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    with open(output_path, 'w') as f:
        for metric in metrics_list:
            f.write(json.dumps(metric) + '\n')

    

if __name__ == "__main__":
    main()
