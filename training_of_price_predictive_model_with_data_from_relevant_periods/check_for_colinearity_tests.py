# ==========================================================
#  Housing Data Diagnostics Script
#  Includes:
#   1. Variance Inflation Factor (VIF) Test
#   2. Best Subset Feature Selection
#   3. Ramsey RESET Test
# ==========================================================

import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from mlxtend.feature_selection import ExhaustiveFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset

# ----------------------------------------------------------
# Load dataset
# ----------------------------------------------------------
df = pd.read_csv("merged_dataset_with_org_cat_values.csv")

# ----------------------------------------------------------
# Prepare features and target
# ----------------------------------------------------------
# Drop non-numeric or irrelevant columns
drop_cols = ['run id', 'city', 'type', 'date']  # adjust if your columns differ
num_df = df.drop(columns=drop_cols, errors='ignore')

# Separate target variable
y = num_df['price']
X = num_df.drop(columns=['price'], errors='ignore')

# ----------------------------------------------------------
# 1. COLLINEARITY TEST (VIF)
# ----------------------------------------------------------
print("\n=========================")
print("COLLINEARITY TEST (VIF)")
print("=========================")


X_numeric = X.apply(pd.to_numeric, errors='coerce')
# Drop any completely non-numeric or all-NaN columns
X_numeric = X_numeric.dropna(axis=1, how='all')

# Fill any remaining NaNs (optional but safe)
X_numeric = X_numeric.fillna(0)

# Add constant term
X_const = add_constant(X_numeric)

# Compute VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i)
                   for i in range(X_const.shape[1])]
print(vif_data)
vif_data = pd.DataFrame()
vif_data["Variable"] = X_const.columns
vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
print(vif_data)

# #==============================
# Variance Inflation Factor (VIF)
# ==============================
#                          Variable          VIF
# 0                           const  9191.175328
# 1                    squareMeters     3.387034
# 2                           rooms     3.258870
# 3                           floor     1.688144
# 4                      floorCount     2.332420
# 5                       buildYear     1.716020
# 6                        latitude     1.204056
# 7                       longitude     1.287396
# 8                  centreDistance     1.962540
# 9                        poiCount     1.609658
# 10                 schoolDistance     6.618750
# 11                 clinicDistance     1.971182
# 12             postOfficeDistance     3.546168
# 13           kindergartenDistance     4.283651
# 14             restaurantDistance     4.757454
# 15                collegeDistance     1.689112
# 16               pharmacyDistance     6.132029
# 17                      ownership     1.078915
# 18               buildingMaterial     1.175199
# 19                hasParkingSpace     1.125026
# 20                     hasBalcony     1.158707
# 21                    hasElevator     1.757968
# 22                    hasSecurity     1.097112
# 23                 hasStorageRoom     1.259265
# 24         distance_to_shopping_1    10.575654
# 25         distance_to_shopping_2    18.938349
# 26         distance_to_shopping_3    12.654714
# 27            distance_to_green_1   514.051785
# 28            distance_to_green_2   551.314790
# 29            distance_to_green_3    29.770215
# 30            distance_to_train_1     3.154642
# 31            distance_to_train_2     7.833438
# 32            distance_to_train_3     5.641327
# 33  distance_to_highschools_and_1    15.283919
# 34  distance_to_highschools_and_2    19.193293
# 35  distance_to_highschools_and_3     8.910225
# 36     distance_to_tram_and_bus_1    21.076786
# 37     distance_to_tram_and_bus_2    27.687840
# 38     distance_to_tram_and_bus_3    10.177595
# 39     distance_to_cultural_and_1     7.155947
# 40     distance_to_cultural_and_2    13.317268
# 41     distance_to_cultural_and_3    10.312304
# 42          distance_to_primary_1     7.045257
# 43          distance_to_primary_2     6.957483
# 44          distance_to_primary_3     4.725747
#                          Variable          VIF
# 0                           const  9191.175328
# 1                    squareMeters     3.387034
# 2                           rooms     3.258870
# 3                           floor     1.688144
# 4                      floorCount     2.332420
# 5                       buildYear     1.716020
# 6                        latitude     1.204056
# 7                       longitude     1.287396
# 8                  centreDistance     1.962540
# 9                        poiCount     1.609658
# 10                 schoolDistance     6.618750
# 11                 clinicDistance     1.971182
# 12             postOfficeDistance     3.546168
# 13           kindergartenDistance     4.283651
# 14             restaurantDistance     4.757454
# 15                collegeDistance     1.689112
# 16               pharmacyDistance     6.132029
# 17                      ownership     1.078915
# 18               buildingMaterial     1.175199
# 19                hasParkingSpace     1.125026
# 20                     hasBalcony     1.158707
# 21                    hasElevator     1.757968
# 22                    hasSecurity     1.097112
# 23                 hasStorageRoom     1.259265
# 24         distance_to_shopping_1    10.575654
# 25         distance_to_shopping_2    18.938349
# 26         distance_to_shopping_3    12.654714
# 27            distance_to_green_1   514.051785
# 28            distance_to_green_2   551.314790
# 29            distance_to_green_3    29.770215
# 30            distance_to_train_1     3.154642
# 31            distance_to_train_2     7.833438
# 32            distance_to_train_3     5.641327
# 33  distance_to_highschools_and_1    15.283919
# 34  distance_to_highschools_and_2    19.193293
# 35  distance_to_highschools_and_3     8.910225
# 36     distance_to_tram_and_bus_1    21.076786
# 37     distance_to_tram_and_bus_2    27.687840
# 38     distance_to_tram_and_bus_3    10.177595
# 39     distance_to_cultural_and_1     7.155947
# 40     distance_to_cultural_and_2    13.317268
# 41     distance_to_cultural_and_3    10.312304
# 42          distance_to_primary_1     7.045257
# 43          distance_to_primary_2     6.957483
# 44          distance_to_primary_3     4.725747

# ----------------------------------------------------------
# BEST SUBSET SELECTION (SEQUENTIAL FORWARD)
# ----------------------------------------------------------
print("\n==========================================")
print("BEST SUBSET SELECTION (SEQUENTIAL FORWARD)")
print("==========================================")

import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

# ---------------------------
# 1. Przygotowanie danych
# ---------------------------

# Zakładamy, że X i y są już przygotowane
# i że wszystkie kolumny kategoryczne są zakodowane liczbowo (label encoding)

# Kopia danych wejściowych (bez 'run id' jeśli jest)
X_encoded = X.copy()
X_encoded = X_encoded.drop(columns=['run id'], errors='ignore')

# Wypełniamy ewentualne brakujące wartości zerami
X_encoded = X_encoded.apply(pd.to_numeric, errors='coerce').fillna(0)

# ---------------------------
# 2. Definicja modelu
# ---------------------------
model = LinearRegression()

# ---------------------------
# 3. Sequential Forward Selection
# ---------------------------
sfs = SequentialFeatureSelector(
    model,
    k_features=10,      # liczba cech do wybrania
    forward=True,       # selekcja krok po kroku do przodu
    floating=False,     # jeśli True, pozwala na cofanie i zamianę cech
    scoring='r2',       # metryka oceny
    cv=5,               # liczba foldów w cross-validation
    n_jobs=-1           # użycie wszystkich rdzeni CPU
)

# Dopasowanie modelu do danych
sfs.fit(X_encoded, y)

# ---------------------------
# 4. Wyniki
# ---------------------------
print("Selected features:")
print(list(sfs.k_feature_names_))
print("Best R² score: {:.4f}".format(sfs.k_score_))



# ----------------------------------------------------------
# 3. RAMSEY RESET TEST (label encoding version)
# ----------------------------------------------------------
print("\n==========================================")
print("RAMSEY RESET TEST (label encoding version)")
print("==========================================")
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset

# Usuń kolumnę 'id', jeśli istnieje
X = X.drop(columns=['id'], errors='ignore')

# Wykryj kolumny kategoryczne
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print("Kolumny tekstowe:", categorical_cols)

# Tworzymy słowniki mapowania dla legend
encoding_maps = {}

# Zamiana wartości tekstowych na liczby
for col in categorical_cols:
    unique_values = X[col].dropna().unique()
    mapping = {val: i for i, val in enumerate(unique_values)}
    encoding_maps[col] = mapping
    X[col] = X[col].map(mapping)

# Wyświetlenie legend dla kolumn kategorycznych
for col, mapping in encoding_maps.items():
    print(f"Legend for '{col}': {mapping}")

# Konwersja wszystkich danych do typu float i obsługa braków danych
X_encoded = X.fillna(0).astype(float)
y_numeric = pd.to_numeric(y, errors='coerce').fillna(0)

# Dodaj kolumnę stałej (const)
X_const = sm.add_constant(X_encoded)

# Dopasuj model OLS
ols_model = sm.OLS(y_numeric, X_const).fit()

# Test Ramsey RESET
ramsey_result = linear_reset(ols_model, power=2, use_f=True)
print(ramsey_result)

print("\nInterpretation:")
print("If p-value < 0.05 → possible model misspecification.")
print("If p-value >= 0.05 → no evidence of misspecification.")
