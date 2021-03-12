def test_data(features, target):
    import numpy as np 
    import pandas as pd
    import seaborn as sns
    from datetime import datetime
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn import linear_model
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn import metrics
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.preprocessing import StandardScaler
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(features, target, random_state = 15, test_size=0.2)
    #instantiate a linear regression object
    lm = LinearRegression()
    #fit the linear regression to the data
    lm = lm.fit(X_train, y_train)

    y_train_pred = lm.predict(X_train)

    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    
    #use fitted model to predict on the test examples
    y_test_pred = lm.predict(X_test)

    #evaluate the predictions on the test examples
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    
    return {'train_rmse': train_rmse, 'test_rmse' : test_rmse}


def test_data_log_target(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, random_state = 15, test_size=0.2)
    
    y_log = np.log(y_train)
    #instantiate a linear regression object
    lm_log = LinearRegression()
    #fit the linear regression to the data
    lm_log = lm_log.fit(X_train, y_log)

    log_train_pred  = lm_log.predict(X_train)
    #log_train_pred the predictions to get them on the same original scale 
    y_train_pred = np.exp(log_train_pred)

    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    
    #use fitted model to predict on the test examples
    
    log_test_pred = lm_log.predict(X_test)
    #get test predictions back to original scale
    y_test_pred = np.exp(log_test_pred)
    
    #evaluate the predictions on the test examples
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))
    
    return {'train_rmse': train_rmse, 'test_rmse' : test_rmse}

def residual_plot(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, random_state = 15, test_size=0.2)
    #instantiate a linear regression object
    lm = LinearRegression()
    #fit the linear regression to the data
    lm = lm.fit(X_train, y_train)

    y_train_pred = lm.predict(X_train)

    
    #use fitted model to predict on the test examples
    y_test_pred = lm.predict(X_test)

    sns.residplot( y_test, y_test_pred,lowess=True, color="g")
    
    plt.show()
    return None

def residual_plot_log(features, target):
    X_train, X_test, y_train, y_test = train_test_split(features, target, random_state = 15, test_size=0.2)
    
    y_log = np.log(y_train)
    #instantiate a linear regression object
    lm_log = LinearRegression()
    #fit the linear regression to the data
    lm_log = lm_log.fit(X_train, y_log)

    log_train_pred  = lm_log.predict(X_train)
    #log_train_pred the predictions to get them on the same original scale 
    y_train_pred = np.exp(log_train_pred)

    train_rmse = np.sqrt(metrics.mean_squared_error(y_train, y_train_pred))
    
    #use fitted model to predict on the test examples
    
    log_test_pred = lm_log.predict(X_test)
    #get test predictions back to original scale
    y_test_pred = np.exp(log_test_pred)
    
    #evaluate the predictions on the test examples
    test_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

    sns.residplot( y_test, y_test_pred,lowess=True, color="g")
    
    plt.show()
    return None
