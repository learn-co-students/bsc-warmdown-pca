from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.mixture import GaussianMixture
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def get_clusters(encoder, X_train, y_train, X_test):
    
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    
    mixture_preprocess = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(n_components=2))

    mixture_preprocess.fit(X_train_encoded, y_train)

    X_train_lda = mixture_preprocess.transform(X_train_encoded)
    X_train_lda = pd.DataFrame(X_train_lda, columns=[f'C{i}' for i in range(1,X_train_lda.shape[1]+1)])
    X_train_lda['target'] = y_train


    cluster = GaussianMixture(n_components=3, means_init=[[-1.25, -1 ],
                                                          [1,1],
                                                          [2.25, -1]],
                             random_state=2021)

    cluster.fit(X_train_lda.iloc[:,:2])
    y_train_clusters = cluster.predict(X_train_lda.iloc[:,:2])

    X_test_lda = mixture_preprocess.transform(X_test_encoded)
    y_test_clusters = cluster.predict(X_test_lda)

    return y_train_clusters, y_test_clusters