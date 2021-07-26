import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix

class ModelHarness:
    
    def __init__(self, X_train, X_test, y_train, y_test, scorer):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scorer = scorer
        self.history = pd.DataFrame()
        
    def run(self, model):
        model.fit(self.X_train, self.y_train)
        name = self.get_name(model)
        train, test = self.print_scores(model)
        self.plot_cm(model)
        frame = pd.DataFrame([[train, test, name]], columns=['train', 'test', 'name'])
        self.history = self.history\
        .append(frame)\
        .reset_index(drop=True)\
        .sort_values(by='test', ascending=False)
        
        
    def print_scores(self, model):
        train = self.scorer(model, self.X_train, self.y_train)
        test = self.scorer(model, self.X_test, self.y_test)
        
        print('Train:', train)
        print('Test:', test)
        
        return train, test
        
    def plot_cm(self, model):
        fig, ax = plt.subplots(1,2, figsize=(20,6))
        
        plot_confusion_matrix(model, self.X_train, self.y_train, ax=ax[0])
        plot_confusion_matrix(model, self.X_test, self.y_test, ax=ax[1])
        ax[0].set_title('Train', fontsize=15)
        ax[1].set_title('Test', fontsize=15)
        
    def get_name(self, model):
        name = type(model).__name__
        if name == 'Pipeline':
            name = type(model.steps[-1][-1]).__name__
        return name

