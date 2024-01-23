# In[]: Import required Python built-in libraries
import sys  # Added import for sys module
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report

# In[]: Constants
FIGURE_SIZE = (12, 6)
SUBPLOT_ROWS = 2
SUBPLOT_COLS = 6
HISTOGRAM_BINS = 30
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5

# In[]: Defining all the functions needed for this project

class FileReader:
    def __init__(self, file_path):
        self.file_path = file_path

    def read_file(self, file):
        """
        Read txt files.
        """
        data = pd.read_csv(self.file_path + file + '.txt', sep=' ', header=None, skip_blank_lines=False)
        return data   


def explore_data(data):
    """
    Explore data by generating descriptive statistics and visualizations.
    """
    print(data.head())
    print(data.describe())
    print(data.info())
    print(data.skew())
    print(data.kurtosis())
    
    '''
    plot the data
    '''
    plt.figure()
    plt.plot(data)
    plt.ylabel('Values in each column')
    plt.xlabel('Data samples/rows')
    
    '''
    Box plot
    '''
    plt.figure()
    data.boxplot()
    plt.ylabel('Values in each column')
    plt.xlabel('Data columns')
    
    '''
    Histogram
    '''
    plt.figure(figsize=FIGURE_SIZE)
    for col in data.columns:
        plt.subplot(SUBPLOT_ROWS, SUBPLOT_COLS, col + 1)
        sns.histplot(data[col], kde=True, bins=HISTOGRAM_BINS)
        plt.title(f'Column {col}')


def preprocess_data(data):
    """
    Preprocess data by dropping duplicates, dropping empty rows, and resetting the index.
    """
    data = data.drop_duplicates()
    data = data.dropna().reset_index(drop=True)
    return data


def split_train_test(data, n_features):
    """
    Split data into training and testing sets.
    """
    X = data.iloc[:, :n_features]
    y = data.iloc[:, -1]

    X_trn, X_tst, y_trn, y_tst = train_test_split(X, y, test_size=TEST_SIZE, shuffle=True, random_state=RANDOM_STATE, stratify=y)
      
    return pd.DataFrame(X_trn), pd.DataFrame(X_tst), y_trn, y_tst


def train_model(model, parameters, X_trn, y_trn):
    """
    Train a machine learning model.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    grid_search = GridSearchCV(model, parameters, cv=cv, scoring='f1_weighted')
    grid_search.fit(X_trn, y_trn)
    best_estimator = grid_search.best_estimator_
    return best_estimator


def get_estimator(X_trn, y_trn):
    """
    Get the best estimator for the model.
    """
    model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced')
    parameters = {'n_estimators': [50, 100],
                  'max_depth': [None, 10],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_split': [2, 5]
                 }
    trained_model = train_model(model, parameters, X_trn, y_trn)    
    return trained_model


def validate_model(model, X_trn, y_trn):
    """
    Validate the model using cross-validation and print classification report.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(model, X_trn, y_trn, cv=cv)
    report = classification_report(y_trn, y_pred)
    print(report)


def visualize_feature_importance(model, data, n_features):
    """
    Visualize feature importance.
    """
    feature_importance = model.feature_importances_
    feature_names = data.iloc[:, :n_features].columns 
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    print(feature_importance_df)

    '''
    Plot feature importance
    '''
    plt.figure()
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance from RandomForestClassifier')
    plt.show()


def get_results(data, n_features, file_path):
    """
    Get results of model training, validation, and predictions.
    """
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_train_test(data, n_features)
    best_estimator_rf = get_estimator(X_train.iloc[:, :n_features], y_train)
    validate_model(best_estimator_rf, X_train.iloc[:, :n_features], y_train)
    validate_model(best_estimator_rf, X_test.iloc[:, :n_features], y_test)
    visualize_feature_importance(best_estimator_rf, data, n_features)

# In[]: Main Code

if __name__ == "__main__":
    
    # Specify the file path
    file_path = 'C:/Users/data/'
    number_of_features = 12

    # Read and process data
    file_reader = FileReader(file_path)
    signal_data = file_reader.read_file('signal_data') 

    # Explore data
    explore_data(signal_data)

    # Get results
    get_results(signal_data, number_of_features, file_path)
