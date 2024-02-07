import numpy as np
import pandas as pd

def get_data_transformed_by_pipe(pipe, X, y):
    return pipe[:-1].fit_transform(X, y)

def feature_importance(X, y, model, pipe):
    X_train_transf = get_data_transformed_by_pipe(pipe, X, y)
    feat_labels = X_train_transf.columns
    feature_importances = np.round((model.feature_importances_ / sum(model.feature_importances_)) * 100, 2)

    f = pd.DataFrame({'features': np.array(list(feat_labels)), 'score %': feature_importances})
    f.sort_values(by=['score %'], ignore_index=True, ascending=False, inplace=True)
    f = f.reset_index(drop=True)
    
    return f, X_train_transf

def transform_curracy(df, columns):
    for col in columns:
        df[col] = df[col].replace(r'[\$ ,]', '', regex=True).astype(float)

def transform_date(df, columns, format='%d-%b-%y'):
    for col in columns:
        try:
            df[col] = pd.to_datetime(df[col], format=format, errors='raise')
        except ValueError:
            print('Erreur sur la date col:', col)
            pass