# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import joblib

transactions = pd.read_csv("/content/drive/MyDrive/Dtasets/transactions_TP2.csv")


print("=========== dem de dataset============")
print(transactions.shape)
print("=========== les 5 pr lignes============")
print(transactions.head())
print("=========== les types  de variables============")
print(transactions.dtypes)
print("le nombre, le porcentage trée de valeurs manquantes ==========================")
N_V_manq= transactions.isnull().sum()
P_V_manq =  transactions.isnull().mean()*100
tabl_manq = pd.DataFrame(
    {"Nombre manquants": N_V_manq, "Pourcentage": P_V_manq}
).sort_values(by="Pourcentage", ascending=False)
print(tabl_manq)
print("le nombre de doublons avec  id_operation=========================")
N_doubl = transactions.duplicated(subset=['id_operation']).sum()
print(N_doubl)
delet_doubl = transactions.drop_duplicates(subset=['id_operation'], keep='first')
print("le nombre de lignes restantes.=========================")
print(len(delet_doubl))


plt.figure(figsize=(15, 5))
#Histogramme
plt.subplot(1, 2, 1)
sns.histplot(transactions['montant_dh'], kde=True, bins=30, color='blue')
plt.title("Hist montant")
plt.xlabel("DH")
plt.ylabel("Fréquence")
#boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x=transactions['montant_dh'], color='red')
plt.title("Boxplot")
plt.xlabel("DH")
plt.tight_layout()
plt.show()

nbr_fraud = transactions['fraude'].value_counts()
porcentage_prop = transactions['fraude'].value_counts(normalize=True) * 100

print("Nombre de transactions par fraude :")
print(nbr_fraud)
print("\nPourcentage de chaque classe (%) :")
print(porcentage_prop)
print("==========diagrame=============")
plt.figure(figsize=(6, 4))
sns.countplot(x='fraude', data=transactions,hue='fraude', palette=['red', 'yellow'], legend=False)

plt.title("(0 : Non-fraude) - (1 : Fraude)")
plt.xlabel("fraude")
plt.ylabel("Nbr de transactions")
plt.show()

y = transactions['resultat_enquete']
x = transactions.drop(columns=['resultat_enquete'])
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size= 0.2 , random_state=42 )
print("y_train :", y_train.value_counts(normalize=True) * 100)

print("y_test :", y_test.value_counts(normalize=True) * 100)


num_cols = x_train.select_dtypes(include=['int64', 'float64']).columns
categor_cols = x_train.select_dtypes(include=['object', 'category']).columns
print("numériques :", categor_cols)
print("catégorielles :", categor_cols)

num_trans = Pipeline(
    steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ]
)

categ_trans = Pipeline(
    steps=[
        (
            'imputer', SimpleImputer(strategy='most_frequent'),
        ),
        (
            'onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
        ),
    ]
)

preproc = ColumnTransformer(
    transformers=[
        ('num', num_trans, num_cols),
        ('cat', categ_trans, categor_cols)
        ]
)

X_train_processed = preproc.fit_transform(x_train)
X_test_processed = preproc.transform(x_test)

print('Dim de X_train après transformation :', X_train_processed.shape)
print('Dim de X_test après transformation :', X_test_processed.shape)

Q1 = x_train['montant_dh'].quantile(0.25)
Q3 = x_train['montant_dh'].quantile(0.75)
IQR = Q3 - Q1
print(f"Q1 = {Q1}")
print(f"Q3 = {Q3}")
print(f"IQR = {IQR}")

v_bas = Q1 - 1.5 * IQR
v_haut = Q3 + 1.5 * IQR

concernees = x_train[
    (x_train['montant_dh'] < v_bas) | (x_train['montant_dh'] > v_haut)
]

print(f"Nombre de lignes concernées : {len(concernees)}")

montant_extreme = x_train['montant_dh']

montant_limit = np.clip(x_train['montant_dh'], v_bas, v_haut)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.boxplot(y=montant_extreme, color='blue')
plt.title('extrêmes')

plt.subplot(1, 2, 2)
sns.boxplot(y=montant_limit, color='green')
plt.title('Limités')

plt.tight_layout()
plt.show()

scaler_minmax = MinMaxScaler()
scaler_standard = StandardScaler()
scaler_robust = RobustScaler()
# appliquer sur une seule colonne
montant_train = x_train[['montant_dh']]
montant_test = x_test[['montant_dh']]
# 1.fit sur train et transformer le test
scaler_minmax.fit(montant_train)
montant_minmax_test = scaler_minmax.transform(montant_test)
#_____minmax_______
scaler_minmax.fit(montant_train)
m_minmax = scaler_minmax.transform(montant_test)
#_____standard_______
scaler_standard.fit(montant_train)
m_std = scaler_standard.transform(montant_test)
#______robust______
scaler_robust.fit(montant_train)
m_rob = scaler_robust.transform(montant_test)

"""====Q19: examples===="""

exemples = pd.DataFrame(
    {'montant_dh': [100, 500, 1000, 25000]}
)

Q19 = pd.DataFrame()
Q19['Valeur_Brute'] = exemples['montant_dh']
Q19['MinMax'] = scaler_minmax.transform(exemples)
Q19['StandardScaler (Z-score)'] = scaler_standard.transform(exemples)
Q19['RobustScaler'] = scaler_robust.transform(exemples)

print(Q19)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.histplot(m_minmax, kde=True, legend=False)
plt.title('MinMaxScaler')

plt.subplot(1, 3, 2)
sns.histplot(m_std, kde=True, legend=False)
plt.title('Z-score')

plt.subplot(1, 3, 3)
sns.histplot(m_rob, kde=True, legend=False)
plt.title('RobustScaler')

plt.tight_layout()
plt.show()

nouveau_montant = pd.DataFrame({'montant_dh': [50000]})
print(scaler_minmax.transform(nouveau_montant))

counts = y_train.value_counts()
print(counts)

# le rapport entre les deux classes (majoritaire / minoritaire)
rapport = counts.iloc[0] / counts.iloc[1]
print(f"Rapport entre les deux classes : {rapport:.2f}")


r_o_s = RandomOverSampler(random_state=42)
x_train_over, y_train_over = r_o_s.fit_resample(x_train, y_train)

print("Avant oversampling :", y_train.value_counts().to_dict())
print("Après oversampling :", pd.Series(y_train_over).value_counts().to_dict())

r_u_s = RandomUnderSampler(random_state=42)
x_train_under, y_train_under = r_u_s.fit_resample(x_train, y_train)

print("Après undersampling :", pd.Series(y_train_under).value_counts().to_dict())

x_train_num = x_train.select_dtypes(include=['number'])

x_train_num = x_train_num.fillna(x_train_num.mean())

smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train_num, y_train)

print("Après SMOTE :", pd.Series(y_train_smote).value_counts().to_dict())

comparaison_classes = pd.DataFrame(
    {
        'Avant rééquilibrage': y_train.value_counts().sort_index(),
        'Après Oversampling ': pd.Series(y_train_over).value_counts().sort_index(),
        'Après Undersampling ': pd.Series(y_train_under).value_counts().sort_index(),
    }
)
print(comparaison_classes)

numeric_features = x_train.select_dtypes(include=['number']).columns
categorical_features = x_train.select_dtypes(
    include=['object', 'category']
).columns

preprocessor = ColumnTransformer(
    transformers=[
        (
            'num',
            Pipeline(
                steps=[
                    (
                        'imputer',
                        SimpleImputer(strategy='median'),
                    ),
                    ('scaler', RobustScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            'cat',
            Pipeline(
                steps=[
                    (
                        'imputer',
                        SimpleImputer(strategy='most_frequent'),
                    ),
                    (
                        'onehot',
                        OneHotEncoder(handle_unknown='ignore', sparse_output=False),
                    ),
                ]
            ),
            categorical_features,
        ),
    ]
)
print('ColumnTransformer succès ')

x_train_processed = preprocessor.fit_transform(x_train)
x_test_processed = preprocessor.transform(x_test)

print("Nombre de colonnes après transformation (Train) :", X_train_processed.shape[1])
print("Nombre de colonnes après transformation (Test) :", X_test_processed.shape[1])
print("Valeurs NaN dans Train :", np.isnan(x_train_processed).sum())
print("Valeurs NaN dans Test :", np.isnan(x_test_processed).sum())

exemple_inconnu = x_test.iloc[:5].copy()

colon_new = x_test.select_dtypes(
    include=['object', 'category']
).columns[0]
exemple_inconnu.iloc[0, exemple_inconnu.columns.get_loc(colon_new)] = (
    'valeur_inconnue_test'
)

# Vérifiez que OneHotEncoder peut transformer cette ligne sans erreur
x_test_transformed = preprocessor.transform(exemple_inconnu)

print('Simulation réussie sans erreur ')
print("Dim des données transformées :", x_test_transformed.shape)

joblib.dump(preprocessor, 'preprocessor_fraude.pkl')

loaded_preprocessor = joblib.load('preprocessor_fraude.pkl')

echantillon = x_test.iloc[:5]

res_original = preprocessor.transform(echantillon)
res_recharg = loaded_preprocessor.transform(echantillon)

# Comparé
sont_identiques = np.allclose(res_original, res_recharg)
print("Les valeurs rechargées sont-elles identiques ? :", sont_identiques)
