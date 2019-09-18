import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split

dataset="banana"
#dataset="phoneme"

def split_data():
    dataset_data = pd.read_csv("../data/"+dataset+".csv")
    data_train, data_test = train_test_split(dataset_data, test_size=0.10)

    data_train.to_csv("../data/"+dataset+"data_train.csv", sep='\t')
    data_test.to_csv("../data/"+dataset+"data_test.csv", sep='\t')


if __name__ == '__main__':
    split_data()
else:
    data_train = pd.read_csv("../data/"+dataset+"data_train.csv", sep='\t', index_col=0)
    print(data_train.shape)
    X_train = data_train.drop('Class', axis=1)
    y_train = data_train['Class']

    data_test = pd.read_csv("../data/"+dataset+"data_test.csv", sep='\t', index_col=0)
    print(data_test.shape)
    X_test = data_test.drop('Class', axis=1)
    y_test = data_test['Class']

    dataset_data = pd.read_csv("../data/"+dataset+".csv")
    X = dataset_data.drop('Class', axis=1)
    y = dataset_data['Class']

    data = dict(
        bananadata=dataset_data,
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
