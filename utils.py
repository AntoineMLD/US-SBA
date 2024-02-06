import numpy as np
import pandas as pd

def get_data_transformed_by_pipe(pipe, X, y):
    return pipe[:-1].fit_transform(X, y)

def feature_importance(X, y, model, pipe):
    X_train_transf = get_data_transformed_by_pipe(pipe, X, y)
    feat_labels = X_train_transf.columns
    feature_importances = np.round((model.feature_importances_ / sum(model.feature_importances_)) * 100, 2)

    f = pd.DataFrame({'features': np.array(list(feat_labels)), 'score %': feature_importances})
    f.sort_values(by=['score %'], ascending=False, inplace=True)
    
    return f, X_train_transf