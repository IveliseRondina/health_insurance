import pandas as pd
import numpy as np


class Project:
    def __init__(self):
        pass
        
    def metrics_at_k(self, y_true, y_pred, percent): 
        data = pd.DataFrame(y_true)

        k = int(len(data) * percent)

        data['score'] = y_pred[:,1].tolist()
        data = data.sort_values('score', ascending=False)

        data['ranking'] = np.arange(1, (len(data)+1),1)

        data['prec_at_k'] = data.response.cumsum() / data.ranking
        data['recall_at_k'] = data.response.cumsum() / data.response.sum()

        data = data.reset_index(drop=True)

        prec_at_k = data.loc[k, 'prec_at_k']
        recall_at_k = data.loc[k, 'recall_at_k']

        return pd.DataFrame({'precision': [prec_at_k], 'recall': [recall_at_k]})


    def metrics_cv(self, model, train, cross_validation, percent, pipetransf):
        score_train = pd.DataFrame()
        score_test = pd.DataFrame()

        for train_index, test_index in cross_validation.split(train, train.response):
            Xtrain = train.iloc[train_index, 1:-1]
            Xval = train.iloc[test_index, 1:-1]
            ytrain = train.iloc[train_index, -1]
            yval = train.iloc[test_index, -1]

            Xtrain = pipetransf.fit_transform(Xtrain)
            Xval = pipetransf.transform(Xval)

            model.fit(Xtrain, ytrain)
            ytrain_pred = model.predict_proba(Xtrain)
            yval_pred = model.predict_proba(Xval)

            score_train = pd.concat([score_train, self.metrics_at_k(ytrain, ytrain_pred, percent)], ignore_index=True)
            score_test = pd.concat([score_test, self.metrics_at_k(yval, yval_pred, percent)], ignore_index=True)

        score_train['fold'] = score_train.index+1
        score_test['fold'] = score_test.index+1
        metrics = score_train.merge(score_test, on='fold', suffixes=('_train', '_test'))
        metrics = metrics[['fold', 'precision_train', 'precision_test', 'recall_train', 'recall_test']]

        media = pd.DataFrame(metrics.mean()).T
        media.fold = 'mean'
        mean_metrics = pd.concat([metrics, media], ignore_index=True)

        return mean_metrics


    def final_metrics_at_k(self, y_true, y_pred, percent):
        data = pd.DataFrame(y_true)

        k = int(len(data) * percent)

        data['score'] = y_pred
        data = data.sort_values('score', ascending=False)

        data['ranking'] = np.arange(1, (len(data)+1),1)

        data['prec_at_k'] = data.response.cumsum() / data.ranking
        data['recall_at_k'] = data.response.cumsum() / data.response.sum()

        data = data.reset_index(drop=True)

        prec_at_k = data.loc[k-1, 'prec_at_k']
        recall_at_k = data.loc[k-1, 'recall_at_k']

        return pd.DataFrame({'precision': [prec_at_k], 'recall': [recall_at_k]})


    def recall_at_k(self, data, k=2000 ): #PROJECT
        # reset index
        data = data.reset_index( drop=True )

        # create ranking order
        data['ranking'] = data.index + 1

        data['recall_at_k'] = data['response'].cumsum() / data['response'].sum()

        return data.loc[k, 'recall_at_k']


    def make_score(self, y_true, yhat_model): #PROJECT
        # copy data
        data = pd.DataFrame(y_true)

        # propensity score
        data['score'] = yhat_model[:,1].tolist()

        # sorted clients by propensity score
        data = data.sort_values('score', ascending=False)
        data = data.reset_index( drop=True )

        data['ranking'] = np.arange(1, len(data)+1, 1)
        data['recall_at_k'] = data['response'].cumsum() / data['response'].sum()
        data['precision_at_k'] = data['response'].cumsum() / data['ranking']

        percents = np.arange(0.05, 1.05, 0.05)
        k_list = []
        for i in percents:
            k = int(len(data)*i) - 1
            k_list.append( k )

        data = data.loc[k_list, ['score', 'precision_at_k', 'recall_at_k']]
        data['k'] = percents
        data = data[['k', 'precision_at_k', 'recall_at_k', 'score']]
        data = data.reset_index(drop=True)

        return data