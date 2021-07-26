import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

lk={'color':'red'}
sk={'edgecolor':'black', 's':80}


def print_data(df):
    print('============================================')
    print('              Head of DataFrame')
    display(df.head(2))
    print('============================================')
    print('             Correlation Matrix:')
    display(df.corr())
    print('============================================')

def plot_relationships(frame, transformed = False):

    fig, axes = plt.subplots(1,5, figsize=(20,6))

    sns.regplot(frame.x1, frame.y, ax=axes[0], line_kws=lk, scatter_kws=sk)
    sns.regplot(x='x2', y='y', data=frame, ax=axes[1],line_kws=lk, scatter_kws=sk)
    sns.regplot(x='x3', y='y', data=frame, ax=axes[2],line_kws=lk, scatter_kws=sk)
    sns.regplot(x='x4', y='y', data=frame, ax=axes[3],line_kws=lk, scatter_kws=sk)
    sns.regplot(x='x5', y='y', data=frame, ax=axes[4],line_kws=lk, scatter_kws=sk)
    axes[0].set_title('y~x1', fontsize=20)
    axes[1].set_title('y~x2', fontsize=20)
    axes[2].set_title('y~x3', fontsize=20)
    axes[3].set_title('y~x4', fontsize=20)
    axes[4].set_title('y~x5', fontsize=20)
    if transformed:
        fig.suptitle('Dependent~Independent\nTransformed Relationships', fontsize=30, y=1.1);
    else:
        fig.suptitle('Dependent~Independent\nUntransformed Relationships', fontsize=30, y=1.1);
        
def plot_lda_clusters(X, target_encoder, category = 'Referred for Prosecution'):
    plt.figure(figsize=(15,6))
    for label in X.target.unique():
        
        frame = X.query(f'target=={label}')
        descr = target_encoder.inverse_transform([label])[0]
        if descr!=category:
            plt.scatter(frame.C1, frame.C2, label=descr, 
                        edgecolor='black', alpha=.6)

    frame = X.query(f'target=={target_encoder.transform([category])[0]}')
    plt.scatter(frame.C1, frame.C2, label=category, 
                edgecolor='black', alpha=1,s=50)
    plt.legend(loc='upper right');