import pickle


def sarimax_prediction(df2_train_, df2_test_):
    with open('learned_model.pkl', 'rb') as f:
        model = pickle.load(f)

    df2_train_['sarima_fitted'] = model.fittedvalues

    pred = model.predict(start =df2_test_.index[0], end = df2_test_.index[-1])

    return pred, df2_train_


