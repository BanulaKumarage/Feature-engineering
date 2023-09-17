# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
import math
import seaborn as sn
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
train = pd.read_csv('train.csv')
valid = pd.read_csv('valid.csv')
test = pd.read_csv('test.csv')

# %%
test.isnull().sum()

# %%
train.head()

# %%
# Separate features and labels
X_train = train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_train = train[['label_1', 'label_2', 'label_3', 'label_4']]
X_val = valid.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_val = valid[['label_1', 'label_2', 'label_3', 'label_4']]
X_test = test.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_test = test[['label_1', 'label_2', 'label_3', 'label_4']]

# %%
X_train.head()

# %%
knn = KNeighborsClassifier(n_neighbors=1)
def knn_classifier(X_train, Y_train, X_val, Y_val):
    knn.fit(np.array(X_train), Y_train)

    y_pred = knn.predict(np.array(X_val))

    accuracy = accuracy_score(Y_val, y_pred)
    return accuracy

# %% [markdown]
# ### Label_1

# %%
plt.figure(figsize=(18, 6))
sn.countplot(data=y_train, x='label_1', color='teal')
plt.xlabel('Speaker ID', fontsize=12)

# %%
accuracy = knn_classifier(X_train, y_train['label_1'], X_val, y_val['label_1'] )
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
label_1_pred_before = knn.predict(np.array(X_test))

# %%
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range (len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr        

# %%
def correlation_with_label(dataset, label, thereshold = 0.01):

    corr_matrix = dataset.corr()
    label_col_id = corr_matrix.columns.get_loc(label)

    corr_with_label = corr_matrix.iloc[:, label_col_id]

    corr_fearures = corr_with_label[corr_with_label.index != label]
    corr_fearures = corr_fearures[corr_fearures.abs() < thereshold]

    return corr_fearures.index.tolist()

# %%
train.corr()

# %%
corr_features = correlation_with_label(train, 'label_1', 0.01)
len(set(corr_features))

# %%
X_train_filtered = X_train.drop(columns=list(corr_features))
X_val_filtered = X_val.drop(columns=list(corr_features))
X_test_filtered = X_test.drop(columns=list(corr_features))

# %%
corr_features = correlation(X_train_filtered, 0.5)
len(set(corr_features))

# %%
X_train_filtered = X_train_filtered.drop(columns=list(corr_features))
X_val_filtered = X_val_filtered.drop(columns=list(corr_features))
X_test_filtered = X_test_filtered.drop(columns=list(corr_features))

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_val_scaled = scaler.transform(X_val_filtered)
X_test_scaled = scaler.transform(X_test_filtered)

# %%
accuracy = knn_classifier(X_train_scaled, y_train['label_1'], X_val_scaled, y_val['label_1'] )
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
pca = PCA(n_components=0.95, svd_solver = 'full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
X_train_pca.shape

# %%
min_shape = 256
while True:
    pca = PCA(n_components=0.95, svd_solver='full')
    min_shape = X_train_pca.shape[1]
    X_train_pca = pca.fit_transform(X_train_pca)
    X_val_pca = pca.transform(X_val_pca)
    X_test_pca = pca.transform(X_test_pca)
    accuracy_val = knn_classifier(X_train_pca, y_train['label_1'], X_val_pca, y_val['label_1'] )
    if accuracy_val < 0.98:
        break
 
print (min_shape)

# %%
pca = PCA(n_components=49, svd_solver = 'full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
X_train_pca.shape

# %%
accuracy = knn_classifier(X_train_pca, y_train['label_1'], X_val_pca, y_val['label_1'] )
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
label_1_pred_after = knn.predict(np.array(X_test_pca))

# %%
label1_features = pd.DataFrame(data=X_test_pca, columns=[f'new_feature_{i+1}' for i in range(X_test_pca.shape[1])])
label1_features.insert(0,'Predicted labels before feature engineering',label_1_pred_before)
label1_features.insert(1,'Predicted labels after feature engineering', label_1_pred_after)
label1_features.insert(2,'No of new features', X_test_pca.shape[1])

# %%
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )
 
plt.plot(
    range(1,len(pca.explained_variance_ )+1),
    np.cumsum(pca.explained_variance_),
    c='red',
    label='Speaker ID')
 
plt.legend(loc='upper left')
plt.xlabel('Number of Features')
plt.ylabel('Explained variance (eignenvalues)')
 
plt.show()

# %% [markdown]
# ### Label_2

# %%
valid['label_2'].isnull().sum()

# %%
label2_train = train.copy()
label2_valid = valid.copy()
label2_test = test.copy()

# %%
label2_train = label2_train.dropna(subset=['label_2'])
label2_valid = label2_valid.dropna(subset=['label_2'])

# %%
label2_train['label_2'].head()

# %%
X_train = label2_train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_train = label2_train[['label_1', 'label_2', 'label_3', 'label_4']]
X_val = label2_valid.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_val = label2_valid[['label_1', 'label_2', 'label_3', 'label_4']]
X_test = label2_test.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_test = label2_test[['label_1', 'label_2', 'label_3', 'label_4']]

# %%
plt.figure(figsize=(18, 6))
ax = sn.histplot(data=y_train, x='label_2', bins=20, kde=False)
plt.xlabel('Speaker Age')

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

plt.show()

# %%
model = xgb.XGBRegressor()
model.fit(X_train, y_train['label_2'])
y_pred = model.predict(X_val)
testScore = math.sqrt(mean_squared_error(y_val['label_2'], y_pred))
print('Test Score: %.2f RMSE' % (testScore))

# %%
label_2_pred_before = model.predict(np.array(X_test))

# %%
label2_train_filtered = label2_train[label2_train['label_2'] < 45]
X_train_filtered = label2_train_filtered.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_train_filtered = label2_train_filtered[['label_1', 'label_2', 'label_3', 'label_4']]

model = xgb.XGBRegressor()
model.fit(X_train_filtered, y_train_filtered['label_2'])
y_pred = model.predict(X_val)
testScore = math.sqrt(mean_squared_error(y_val['label_2'], y_pred))
print('Test Score: %.2f RMSE' % (testScore))

# %%
corr_features = correlation_with_label(label2_train, 'label_2')
len(set(corr_features))

# %%
X_train_filtered = X_train.drop(columns=list(corr_features))
X_val_filtered = X_val.drop(columns=list(corr_features))
X_test_filtered = X_test.drop(columns=list(corr_features))

# %%
corr_features = correlation(X_train_filtered, 0.5)
len(set(corr_features))

# %%
X_train_filtered = X_train_filtered.drop(columns=list(corr_features))
X_val_filtered = X_val_filtered.drop(columns=list(corr_features))
X_test_filtered = X_test_filtered.drop(columns=list(corr_features))

# %%
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_val_scaled = scaler.transform(X_val_filtered)
X_test_scaled = scaler.transform(X_test_filtered)

# %%
pca = PCA(n_components=0.95, svd_solver = 'full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
min_shape = 256
while True:
    pca = PCA(n_components=0.95, svd_solver='full')
    min_shape = X_train_pca.shape[1]
    X_train_pca = pca.fit_transform(X_train_pca)
    X_val_pca = pca.transform(X_val_pca)
    X_test_pca = pca.transform(X_test_pca)
    model = xgb.XGBRegressor()
    model.fit(X_train_pca, y_train['label_2'])
    y_pred = model.predict(X_val_pca)
    testScore_val = math.sqrt(mean_squared_error(y_val['label_2'], y_pred))
    if testScore_val > 3.8:
        break
 
print (min_shape)   
    

# %%
pca = PCA(n_components=25, svd_solver = 'full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
model = xgb.XGBRegressor()
model.fit(X_train_pca, y_train['label_2'])
y_pred = model.predict(X_val_pca)
testScore = math.sqrt(mean_squared_error(y_val['label_2'], y_pred))
print('Test Score: %.2f RMSE' % (testScore))

# %%
label_2_pred_after = model.predict(np.array(X_test_pca))

# %%
label2_features = pd.DataFrame(data=X_test_pca, columns=[f'new_feature_{i+1}' for i in range(X_test_pca.shape[1])])
label2_features.insert(0,'Predicted labels before feature engineering',label_2_pred_before)
label2_features.insert(1,'Predicted labels after feature engineering', label_2_pred_after)
label2_features.insert(2,'No of new features', X_test_pca.shape[1])

# %%
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )
 
plt.plot(
    range(1,len(pca.explained_variance_ )+1),
    np.cumsum(pca.explained_variance_),
    c='red',
    label='Speaker Age')
 
plt.legend(loc='upper left')
plt.xlabel('Number of Features')
plt.ylabel('Explained variance (eignenvalues)')
 
plt.show()

# %% [markdown]
# ### Label_3

# %%
# Separate features and labels
X_train = train.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_train = train[['label_1', 'label_2', 'label_3', 'label_4']]
X_val = valid.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_val = valid[['label_1', 'label_2', 'label_3', 'label_4']]
X_test = test.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
y_test = test[['label_1', 'label_2', 'label_3', 'label_4']]

# %%
X_train.head()

# %%
tr_df = train.copy()
tr_df = tr_df.drop(['label_1', 'label_2', 'label_4'], axis=1)
val_df = valid.copy()
val_df = val_df.drop(['label_1', 'label_2', 'label_4'], axis=1)
tr_df = tr_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# %%
tr_df.merge(val_df, on=list(tr_df.columns))

# %%
ax = sn.countplot(x=y_train['label_3'])

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=9, color='black')
    
plt.xlabel('Speaker Gender')


# %%
accuracy = knn_classifier(X_train, y_train['label_3'], X_val, y_val['label_3'] )
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
label_3_pred_before = knn.predict(np.array(X_test))

# %%
ros = RandomOverSampler(random_state=0, sampling_strategy=0.75)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train['label_3'])

# %%
ax = sn.countplot(x=y_train_resampled)

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=9)
    
plt.xlabel('Speaker Gender')

# %%
accuracy = knn_classifier(X_train_resampled, y_train_resampled, X_val, y_val['label_3'] )
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
corr_features = correlation_with_label(train, 'label_3', 0.01)
len(set(corr_features))

# %%
X_train_filtered = X_train_resampled.drop(columns=list(corr_features))
X_val_filtered = X_val.drop(columns=list(corr_features))
X_test_filtered = X_test.drop(columns=list(corr_features))

# %%
corr_features = correlation(X_train_filtered, 0.5)
len(set(corr_features))

# %%
X_train_filtered = X_train_filtered.drop(columns=list(corr_features))
X_val_filtered = X_val_filtered.drop(columns=list(corr_features))
X_test_filtered = X_test_filtered.drop(columns=list(corr_features))

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_val_scaled = scaler.transform(X_val_filtered)
X_test_scaled = scaler.transform(X_test_filtered)

# %%
accuracy = knn_classifier(X_train_scaled, y_train_resampled, X_val_scaled, y_val['label_3'] )
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
pca = PCA(n_components=0.95, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
X_train_pca.shape[1]

# %%
min_shape = 256
while True:
    pca = PCA(n_components=0.95, svd_solver='full')
    min_shape = X_train_pca.shape[1]
    X_train_pca = pca.fit_transform(X_train_pca)
    X_val_pca = pca.transform(X_val_pca)
    X_test_pca = pca.transform(X_test_pca)
    accuracy_val = knn_classifier(X_train_pca, y_train_resampled, X_val_pca, y_val['label_3'] )
    if accuracy_val < 1:
        break
 
print (min_shape)
    

# %%
pca = PCA(n_components=35, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
accuracy = knn_classifier(X_train_pca, y_train_resampled, X_val_pca, y_val['label_3'] )
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
label_3_pred_after = knn.predict(np.array(X_test_pca))

# %%
label3_features = pd.DataFrame(data=X_test_pca, columns=[f'new_feature_{i+1}' for i in range(X_test_pca.shape[1])])
label3_features.insert(0,'Predicted labels before feature engineering',label_3_pred_before)
label3_features.insert(1,'Predicted labels after feature engineering', label_3_pred_after)
label3_features.insert(2,'No of new features', X_test_pca.shape[1])

# %%
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )
 
plt.plot(
    range(1,len(pca.explained_variance_ )+1),
    np.cumsum(pca.explained_variance_),
    c='red',
    label='Speaker Gender')
 
plt.legend(loc='upper left')
plt.xlabel('Number of Features')
plt.ylabel('Explained variance (eignenvalues)')
plt.title('Scree plot')
 
plt.show()

# %% [markdown]
# ### Label_4

# %%
plt.figure(figsize=(18, 6))
ax = sn.countplot(x=y_train['label_4'], color='teal')

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=9, color='black')
    
plt.xlabel('Speaker Accent')

# %%
accuracy = knn_classifier(X_train, y_train['label_4'], X_val, y_val['label_4'] )
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
label_4_pred_before = knn.predict(np.array(X_test))

# %%
ros = RandomOverSampler(random_state=0)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train['label_4'])

# %%
plt.figure(figsize=(18, 6))
ax = sn.countplot(x=y_train_resampled, color='teal')

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=9, color='black')

# %%
accuracy = knn_classifier(X_train_resampled, y_train_resampled, X_val, y_val['label_4'] )
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
corr_features = correlation_with_label(train, 'label_4', 0.01)
len(set(corr_features))

# %%
X_train_filtered = X_train_resampled.drop(columns=list(corr_features))
X_val_filtered = X_val.drop(columns=list(corr_features))
X_test_filtered = X_test.drop(columns=list(corr_features))

# %%
corr_features = correlation(X_train_filtered, 0.5)
len(set(corr_features))

# %%
X_train_filtered = X_train_filtered.drop(columns=list(corr_features))
X_val_filtered = X_val_filtered.drop(columns=list(corr_features))
X_test_filtered = X_test_filtered.drop(columns=list(corr_features))

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_filtered)
X_val_scaled = scaler.transform(X_val_filtered)
X_test_scaled = scaler.transform(X_test_filtered)

# %%
accuracy = knn_classifier(X_train_scaled, y_train_resampled, X_val_scaled, y_val['label_4'] )
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
pca = PCA(n_components=0.95, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
X_train_pca.shape[1]

# %%
min_shape = 256
while True:
    pca = PCA(n_components=0.95, svd_solver='full')
    min_shape = X_train_pca.shape[1]
    X_train_pca = pca.fit_transform(X_train_pca)
    X_val_pca = pca.transform(X_val_pca)
    X_test_pca = pca.transform(X_test_pca)
    accuracy_val = knn_classifier(X_train_pca, y_train_resampled, X_val_pca, y_val['label_3'] )
    if accuracy_val < 0.985:
        break
 
print (min_shape)


# %%
pca = PCA(n_components=34, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
X_train_pca

# %%
accuracy = knn_classifier(X_train_pca, y_train_resampled, X_val_pca, y_val['label_4'] )
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
label_4_pred_after = knn.predict(np.array(X_test_pca))

# %%
label4_features = pd.DataFrame(data=X_test_pca, columns=[f'new_feature_{i+1}' for i in range(X_test_pca.shape[1])])
label4_features.insert(0,'Predicted labels before feature engineering',label_4_pred_before)
label4_features.insert(1,'Predicted labels after feature engineering', label_4_pred_after)
label4_features.insert(2,'No of new features', X_test_pca.shape[1])

# %%
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )
 
plt.plot(
    range(1,len(pca.explained_variance_ )+1),
    np.cumsum(pca.explained_variance_),
    c='red',
    label='Speaker Accent')
 
plt.legend(loc='upper left')
plt.xlabel('Number of Features')
plt.ylabel('Explained variance (eignenvalues)')
plt.title('Scree plot')
 
plt.show()

# %% [markdown]
# ### Generating Output

# %%
label4_features.shape[0]

# %%
def write_csv(feature_df, label):
  for i in range(feature_df['No of new features'][0], 256):
        feature_df[f'new_feature_{i+1}'] = pd.NA
  filename = f'output/190331A_label_{label}.csv'
  feature_df.to_csv(filename, index=False)

# %%
write_csv(label1_features.copy(), 1)
write_csv(label2_features.copy(), 2)
write_csv(label3_features.copy(), 3)
write_csv(label4_features.copy(), 4)

# %% [markdown]
# ### Summary

# %%
print (f"Features Count after apply feature engineering for Speaker ID: {label1_features['No of new features'][0]}")
print (f"Features Count after apply feature engineering for Speaker Age: {label2_features['No of new features'][0]}")
print (f"Features Count after apply feature engineering for Speaker Gender: {label3_features['No of new features'][0]}")
print (f"Features Count after apply feature engineering for Speaker Accent: {label4_features['No of new features'][0]}")


