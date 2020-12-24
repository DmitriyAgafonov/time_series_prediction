import numpy as np
from statsmodels.tsa.api import Holt, ExponentialSmoothing


def moving_avarage(df, n):
    df_av = df.rolling(window=n).mean()  # .dropna()

    return df_av


def exp_smoothing(df, alpha):
    df_ex = df.ewm(alpha=alpha).mean().dropna()
    
    return df_ex


def holt_smoothing(df, alpha, beta):
    start_df = df[['TIME'] + ['PITCH'] + ['YAW'] + ['ROLL']]

    result_df = df[['TIME']]

    for i in start_df.columns[1:]:
        fit_model = Holt(start_df[i]).fit(smoothing_level=alpha, smoothing_trend=beta)
        fit_series = fit_model.fittedvalues
        result_df[i] = fit_series

    return result_df


def holt_predict(df_train, df_test, feature_drop2):
    hyperparams = {'TMIN': np.array([0.05, 0.04]),
                   'TMAX': np.array([0.05, 0.05]),
                   'ELECTRIC_PROD': np.array([0.55, 0.25]),
                   'BEER_PROD': np.array([0.05, 0.4])}

    model = Holt(df_train[feature_drop2]).fit(smoothing_level=hyperparams[feature_drop2][0],
                                              smoothing_trend=hyperparams[feature_drop2][1])
    pred = model.predict(start=df_test.index[0], end=df_test.index[-1])

    return pred


def holt_winters_predict(df_train, df_test, feature_drop2):
    model = ExponentialSmoothing(df_train[feature_drop2],
                                 seasonal_periods=12, trend='mul', seasonal='add',
                                 use_boxcox=True,
                                 initialization_method="estimated").fit(optimized=True, use_brute=True)

    pred = model.predict(start=df_test.index[0], end=df_test.index[-1])


    return pred