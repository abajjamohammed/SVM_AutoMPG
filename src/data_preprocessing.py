import numpy as np
import pandas as pd
import seaborn as sns


def load_and_preprocess_data():
    """
    Charge et nettoie le dataset Auto MPG :
    - Supprime les lignes avec NaN (arrive exactement à 392 échantillons)
    - Supprime la colonne 'name' (identifiant non généralisable)
    - Encodage One-Hot de 'origin' (variable nominale)
    """
    df = sns.load_dataset('mpg')
    df_clean = df.dropna().reset_index(drop=True)
    df_clean = df_clean.drop(columns=['name'])
    df_encoded = pd.get_dummies(df_clean, columns=['origin'], drop_first=True, dtype=float)
    return df_encoded


def discretize_mpg(df, strategy='qcut'):
    """
    Discrétise la variable continue mpg en 3 classes de consommation :
    Attention : MPG élevé = Consommation Faible
    - 0 : Consommation Élevée (MPG faible)
    - 1 : Consommation Moyenne
    - 2 : Consommation Faible (MPG élevé)
    """
    labels = [0, 1, 2]
    if strategy == 'qcut':
        # Équi-fréquence / Quantiles
        y_class, bins = pd.qcut(df['mpg'], q=3, labels=labels, retbins=True)
    elif strategy == 'cut':
        # Largeurs égales
        y_class, bins = pd.cut(df['mpg'], bins=3, labels=labels, retbins=True)
    else:
        raise ValueError("La stratégie doit être 'cut' ou 'qcut'.")
    
    return y_class.astype(int).values, bins