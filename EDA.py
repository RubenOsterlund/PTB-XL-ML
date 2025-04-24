import pandas as pd
import numpy as np
import ast
import wfdb
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# Stället med filen i datorn
path = r"/path/to/your/data"

# Laddar ner och konverterar annotation datan
Z_df = pd.read_csv(path+'\\ptbxl_database.csv', index_col='ecg_id')
Z_df.scp_codes = Z_df.scp_codes.apply(lambda x: ast.literal_eval(x))

# Det finns mer recordings än annutation data, den här koden indentifierar vilka recordings som saknas
expected_index = set(range(1, 21838))
actual_index = set(Z_df.index)
missing_indices = expected_index - actual_index
print("Missing indices:", missing_indices)
print(f"Number of missing indices: {len(missing_indices)}")

print(Z_df.head())

dc_df = pd.read_csv(path+'\\scp_statements.csv', index_col=0) # Laddar ner diagnos datan
dc_df = dc_df[dc_df.diagnostic == 1] # Filtrerar och tar bara med diagnoser

print(dc_df.shape)
print(dc_df.head())

# Funktion för att addera kolomnen med superclass diagnostiken
def add_superclass (Z_df_dic):
    res=set()  # Skapar ett tomt set
    for key in Z_df_dic.keys(): # Söker igenom nyckeln finns i df_dic och lägger då till diagnostic_class från dc_df
            if key in dc_df.index:
                res.add(dc_df.loc[key].diagnostic_class)
    return list(res)

Z_df['diagnostic_superclass'] = Z_df.scp_codes.apply(add_superclass) # Lägger till superclass diagnos i Z_df med funktiinen add_superclass

print(Z_df.head())

def load_data(df, path): # Funktion för att ladda ner signal datan från filerna
    if os.path.exists('.venv/signaldata.npy'):
        data = np.load('.venv/signaldata.npy')
    else:
        data = [wfdb.rdsamp(os.path.join(path, f)) for f in tqdm(df.filename_lr)] # Laddar ner datan
        data = np.array([signal for signal, meta in data]) # Lägger den i en np array
        np.save('signaldata', data) # Datan blir en fil som kan kallas på
    return data

X = load_data(Z_df, path)
print(X.shape)

print(Z_df.isnull().sum())

Z_df.reset_index(drop=True, inplace=True)

# Visar första raden av signaldatan i en figur
sample = X[0]

bar, axes = plt.subplots(sample.shape[1], 1, figsize=(20,10))
for i in range(sample.shape[1]):
    sns.lineplot(x=np.arange(sample.shape[0]), y=sample[:, i], ax=axes[i])

plt.show()

# Visar tillhörande annotation
print(Z_df.iloc[0])

# Visar hur många mätningar som tillhör varje superclass
superclass_counts = Z_df.explode('diagnostic_superclass')['diagnostic_superclass'].value_counts()
print(superclass_counts)

# Tar bort nullvärden
Z_df.fillna({col: 0 for col in Z_df.columns}, inplace=True)

# Skapar en ny Y data frame som börjar med värden 0 har kolumnerna med diagnoser, samma index som Z_df och type int
Y = pd.DataFrame(0, index=Z_df.index, columns=['NORM', 'MI', 'STTC', 'CD', 'HYP'], dtype='int')
for i in Y.index: # Går igenom varje kolumn i varje rad och tilldelar värde 1 om diagnos finns
    for k in Z_df.loc[i].diagnostic_superclass:
        Y.loc[i, k] = 1

print(Y.head())
print(Y.shape)

def balanced_split(X, Y):
    """
    Split dataset into train, validation, and test sets
    - Ensure equal number of samples in training sets for each class
    - Maintain original class distribution in validation and test sets
    - Use ALL samples that belong to a single class

    Parameters:
    X (numpy.ndarray): Input features with shape (n_samples, time, channels)
    Y (pandas.DataFrame): Binary labels with columns: 'NORM', 'MI', 'STTC', 'CD'

    Returns:
    tuple of X and Y training, validation, and test sets for each diagnose
    """
    # Remove HYP class if present
    diagnose_columns = [col for col in Y.columns if col != 'HYP']

    # Prepare lists to store sets for each diagnose
    X_train_sets, Y_train_sets = [], []
    X_val_sets, Y_val_sets = [], []
    X_test_sets, Y_test_sets = [], []

    # Set random seed for reproducibility
    np.random.seed(42)

    # Iterate through non-NORM classes
    for other_class in [col for col in diagnose_columns if col != 'NORM']:
        # Find samples for the specific other class
        other_mask = Y[other_class] == 1
        other_X = X[other_mask]
        other_Y = Y[other_mask]

        # Find NORM samples
        norm_mask = Y['NORM'] == 1
        norm_X = X[norm_mask]
        norm_Y = Y[norm_mask]

        # Calculate total samples
        other_total_samples = len(other_X)
        norm_total_samples = len(norm_X)

        # Determine the original ratio of NORM to other class
        norm_to_other_ratio = norm_total_samples / other_total_samples

        # Split sizes (70% train, 15% val, 15% test)
        train_size_other = int(other_total_samples * 0.7)
        val_size_other = test_size_other = (other_total_samples - train_size_other) // 2

        # Calculate corresponding NORM samples based on original ratio
        val_size_norm = int(val_size_other * norm_to_other_ratio)
        test_size_norm = int(test_size_other * norm_to_other_ratio)

        # Randomly select samples for each set for other class
        train_other_indices = np.random.choice(len(other_X), train_size_other, replace=False)
        remaining_other_indices = np.setdiff1d(np.arange(len(other_X)), train_other_indices)
        val_other_indices = np.random.choice(remaining_other_indices, val_size_other, replace=False)
        test_other_indices = np.setdiff1d(remaining_other_indices, val_other_indices)

        # Select matching NORM samples
        # Training set: exactly match other class samples
        train_norm_indices = np.random.choice(len(norm_X), train_size_other, replace=False)

        # Validation and test: use proportional number of NORM samples
        remaining_norm_indices = np.setdiff1d(np.arange(len(norm_X)), train_norm_indices)
        val_norm_indices = np.random.choice(remaining_norm_indices, val_size_norm, replace=False)
        test_norm_indices = np.random.choice(
            np.setdiff1d(remaining_norm_indices, val_norm_indices),
            test_size_norm,
            replace=False
        )

        # Create training set
        X_train = np.concatenate([
            other_X[train_other_indices],
            norm_X[train_norm_indices]
        ])
        Y_train = pd.DataFrame({
            'NORM': [0] * train_size_other + [1] * train_size_other,
            other_class: [1] * train_size_other + [0] * train_size_other
        })

        # Create validation set
        X_val = np.concatenate([
            other_X[val_other_indices],
            norm_X[val_norm_indices]
        ])
        Y_val = pd.DataFrame({
            'NORM': [0] * val_size_other + [1] * val_size_norm,
            other_class: [1] * val_size_other + [0] * val_size_norm
        })

        # Create test set
        X_test = np.concatenate([
            other_X[test_other_indices],
            norm_X[test_norm_indices]
        ])
        Y_test = pd.DataFrame({
            'NORM': [0] * len(test_other_indices) + [1] * len(test_norm_indices),
            other_class: [1] * len(test_other_indices) + [0] * len(test_norm_indices)
        })

        # Shuffle sets
        for dataset_X, dataset_Y in [(X_train, Y_train), (X_val, Y_val), (X_test, Y_test)]:
            shuffle_indices = np.random.permutation(len(dataset_X))
            dataset_X[:] = dataset_X[shuffle_indices]
            dataset_Y.iloc[:] = dataset_Y.iloc[shuffle_indices]

        # Collect datasets
        X_train_sets.append(X_train)
        Y_train_sets.append(Y_train)
        X_val_sets.append(X_val)
        Y_val_sets.append(Y_val)
        X_test_sets.append(X_test)
        Y_test_sets.append(Y_test)

        # Print detailed distributions
        print(f"\nDataset split for {other_class}:")
        print("\nTraining set:")
        print(f"Total samples: {len(X_train)}")
        for cls in Y_train.columns:
            print(f"Class {cls}: {(Y_train[cls] == 1).sum()} samples")

        print("\nValidation set:")
        print(f"Total samples: {len(X_val)}")
        for cls in Y_val.columns:
            print(f"Class {cls}: {(Y_val[cls] == 1).sum()} samples")

        print("\nTest set:")
        print(f"Total samples: {len(X_test)}")
        for cls in Y_test.columns:
            print(f"Class {cls}: {(Y_test[cls] == 1).sum()} samples")

    return (X_train_sets, X_val_sets, X_test_sets,
            Y_train_sets, Y_val_sets, Y_test_sets)

X_train_sets, X_val_sets, X_test_sets, Y_train_sets, Y_val_sets, Y_test_sets = balanced_split(X, Y)

X_train_1, X_train_2, X_train_3 = X_train_sets
X_valid_1, X_valid_2, X_valid_3 = X_val_sets
X_test_1, X_test_2, X_test_3 = X_test_sets

Y_train_1, Y_train_2, Y_train_3 = Y_train_sets
Y_val_1, Y_val_2, Y_val_3 = Y_val_sets
Y_test_1, Y_test_2, Y_test_3 = Y_test_sets

np.save('X_train_1.npy', X_train_1)
np.save('X_valid_1.npy', X_valid_1)
np.save('X_test_1.npy',  X_test_1)
Y_test_1.to_csv('Y_test_1.csv')
Y_train_1.to_csv('Y_train_1.csv')
Y_val_1.to_csv('Y_valid_1.csv')

np.save('X_train_2.npy', X_train_2)
np.save('X_valid_2.npy', X_valid_2)
np.save('X_test_2.npy',  X_test_2)
Y_test_2.to_csv('Y_test_2.csv')
Y_train_2.to_csv('Y_train_2.csv')
Y_val_2.to_csv('Y_valid_2.csv')

np.save('X_train_3.npy', X_train_3)
np.save('X_valid_3.npy', X_valid_3)
np.save('X_test_3.npy',  X_test_3)
Y_test_3.to_csv('Y_test_3.csv')
Y_train_3.to_csv('Y_train_3.csv')
Y_val_3.to_csv('Y_valid_3.csv')

print(X_train_1.shape)
print(Y_train_1.shape)