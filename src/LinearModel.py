from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 


class LinearModel:
        
    def plot_true_vs_preds(self, y, preds, ax=None):
        if not ax:
            fig, ax = plt.subplot()
        ax.scatter(y, preds, edgecolor='black', s=80, label='Prediction')
        ax.plot(y, y, color='red', lw=3, label='True Values')
        ax.set_title('Predictions vs True Values', fontsize=20)
        ax.set_xlabel('True Value')
        ax.set_ylabel('Prediction')
        ax.legend()
        
    def plot_residuals(self, y, preds, ax=None):
        if not ax:
            fig, ax = plt.subplot()
        sns.regplot(preds, preds - y,
                    line_kws={'color':'red'}, 
                    scatter_kws={'edgecolor':'black', 's':80},
                    ax=ax)
        ax.set_title('Residuals', fontsize=20)
        ax.set_xlabel('Prediction')
        ax.set_ylabel('Prediction - True Value')
        
    def construct_equation(self, model, columns, scaler=None):
        if scaler:
            stds = scaler.__dict__['scale_']
            equation = '({:.2e})'.format(model.intercept_) + ' + ' + ' + '.join(['({:.2e}{})'.format(round(coef/scale, 5),x) for coef, x, scale in zip(model.coef_, columns, stds)])
        else:
            equation = '({:.2e})'.format(model.intercept_) + ' + ' + ' + '.join(['({:.2e}{})'.format(round(coef, 5),x) for coef, x in zip(model.coef_, columns)])

        return equation
        
    def linear_model(self, X, y, scaler=None):
        model = LinearRegression()
        model.fit(X, y)
        
        r2 = round(model.score(X, y), 3)
        preds = model.predict(X)
        
        fig, ax = plt.subplots(1,2, figsize=(20,6))
        self.plot_true_vs_preds(y, preds, ax=ax[0])
        self.plot_residuals(y, preds, ax=ax[1])
        
        equation = self.construct_equation(model, X.columns, scaler=scaler)

        fig.suptitle(f'R2= {r2}\n{equation}', fontsize=20, y=1.1)
        return model
    
    def pca_coef(self, feature_names, model, pca, scaler, class_index=1):
        """

        Returns the coefficients for a model that has been reduced
        with sklearn's PCA.

        """

        coeffs = {}
        # For multi class classification problems, model.coef_
        # returns a matrix of coefficients for each class
        # If model.coef_.shape[1] exists and is not 0
        # then the coefficients are collected for the desired
        # class

        if len(model.coef_.shape) > 1:
            weights = model.coef_[class_index]
        else:
            weights = model.coef_

        notate = np.vectorize(lambda x: '{:.2e}'.format(x))
        for idx in range(len(feature_names)):
            coeffs[feature_names[idx]] = str(notate(weights @ pca.components_[:,idx]))


        keys = list(coeffs)
        for idx in range(len(coeffs)):
            key = keys[idx]
            scale = scaler.__dict__['scale_'][idx]
            coeffs[key] = str(notate(float(coeffs[key])/scale))

        return coeffs