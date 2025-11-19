#Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
#geliştirilmesi beklenmektedir.

#Veri Seti Hikayesi

# 21 Değişken 7043 Gözlem 977.5 KB
# CustomerId Müşteri İd’si
# Gender Cinsiyet
# SeniorCitizen Müşterinin yaşlı olup olmadığı (1, 0)
# Partner Müşterinin bir ortağı olup olmadığı (Evet, Hayır)
# Dependents Müşterinin bakmakla yükümlü olduğu kişiler olup olmadığı (Evet, Hayır
# tenure Müşterinin şirkette kaldığı ay sayısı
# PhoneService Müşterinin telefon hizmeti olup olmadığı (Evet, Hayır)
# MultipleLines Müşterinin birden fazla hattı olup olmadığı (Evet, Hayır, Telefon hizmeti yok)
# InternetService Müşterinin internet servis sağlayıcısı (DSL, Fiber optik, Hayır)
# OnlineSecurity Müşterinin çevrimiçi güvenliğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# OnlineBackup Müşterinin online yedeğinin olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# DeviceProtection Müşterinin cihaz korumasına sahip olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# TechSupport Müşterinin teknik destek alıp almadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingTV Müşterinin TV yayını olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# StreamingMovies Müşterinin film akışı olup olmadığı (Evet, Hayır, İnternet hizmeti yok)
# Contract Müşterinin sözleşme süresi (Aydan aya, Bir yıl, İki yıl)
# PaperlessBilling Müşterinin kağıtsız faturası olup olmadığı (Evet, Hayır)
# PaymentMethod Müşterinin ödeme yöntemi (Elektronik çek, Posta çeki, Banka havalesi (otomatik), Kredi kartı (otomatik))
# MonthlyCharges Müşteriden aylık olarak tahsil edilen tutar
# TotalCharges Müşteriden tahsil edilen toplam tutar
# Churn Müşterinin kullanıp kullanmadığı (Evet veya Hayır)

#### Görev 1 : Keşifçi Veri Analizi ####
# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
# Adım 5: Aykırı gözlem var mı inceleyiniz.
# Adım 6: Eksik gözlem var mı inceleyiniz.

#### Görev 2 : Feature Engineering ####
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
# Adım 2: Yeni değişkenler oluşturunuz.
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

#### Görev 3 : Modelleme ####
# Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli
# tekrar kurunuz.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import joblib
import graphviz
import pydotplus
import plotly.graph_objects as go

from scipy import stats
from datetime import date
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from skompiler import skompile

import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
df = pd.read_csv("Telco-Customer-Churn.csv")
df.head()

#### Görev 1 : Keşifçi Veri Analizi ####
# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.



def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    cat_cols: Kategorik değişken listesi
    num_cols: Numerik değişken listesi
    cat_but_car: Kategorik görünüp kardinal olan değişken listesi
    """

    # Kategorik kolonlar
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]

    # Numerik olup aslında kategorik olan kolonlar
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th
                   and dataframe[col].dtype != "O"]

    # Kategorik olup kardinal olan kolonlar (çok fazla sınıf)
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th
                   and dataframe[col].dtype == "O"]

    # Final kategorik kolon listesi
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # Final numerik kolon listesi
    num_cols = [col for col in dataframe.columns if dataframe[col].dtype != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
df["TotalCharges"].dtype # object olarak görünüyor veride bu nedenle unique değeri yüksek çıkmış.

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
cat_cols, num_cols, cat_but_car = grab_col_names(df)


#kategorik sutunlarda boşluk temizliği için ;
for col in df.columns:
    if df[col].dtype == "O":
        df[col] = df[col].str.strip()

df.dtypes

cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols
num_cols
cat_but_car

df.isnull().sum()


# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.

#Numerik değişkenler için histogram

def num_summary(dataframe, numerical_col, plot=False):
    print(dataframe[numerical_col].describe().T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                       "Ratio": dataframe[col_name].value_counts() / len(dataframe)}))
    print("####################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col, plot=True)



# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.


# 1) CHURN'Ü DOĞRU ENCODE ET
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
# 2) TARGET ANALİZ FONKSİYONU
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({
        "COUNT": dataframe[categorical_col].value_counts(),
        "RATIO": dataframe[categorical_col].value_counts() / len(dataframe),
        "TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()
    }))
    print("###########################################")
# 3) CHURN DIŞINDAKİ TÜM KATEGORİKLERİ ANALİZ ET
cat_cols = [col for col in cat_cols if col != "Churn"]

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


# 4) GÖRSEL ÖRNEK
plt.figure(figsize=(6, 4))
sns.barplot(x=df["Contract"], y=df["Churn"])
plt.title("Churn by Contract Type")
plt.show()

print(df["Churn"].unique())
print("Churn NA:", df["Churn"].isnull().sum())



# Adım 5: Aykırı gözlem var mı inceleyiniz.

#Telco datasetinde sayısal değişkenler:
#tenure
#MonthlyCharges
#TotalCharges
#(SeniorCitizen sayısal ama binari → aykırı değeri anlamlı değil)
#Bu adımda şunları yapacağız:
# IQR yöntemi ile alt–üst limitleri hesaplama
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    iqr = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * iqr
    up_limit = quartile3 + 1.5 * iqr
    return low_limit, up_limit

# Hangi değişkenlerde outlier var tespit etme
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col, ": ", check_outlier(df, col))



# Adım 6: Eksik gözlem var mı inceleyiniz.

df.isnull().sum()
df.isnull().mean() * 100
import missingno as msno
msno.matrix(df)
plt.show()

msno.bar(df)
plt.show()


# TotalCharges, Tenure=0 olan yeni müşterilerde boş gelir.  yani kişi daha yeni müşteri, henüz toplam ücret oluşmamıştır.


#### Görev 2 : Feature Engineering ####
# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

#Müşteri şirkette 0 ay durduysa → total charges = 0
#1 ay durduysa → total charges = monthly charge

df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
df.isnull().sum()


# Adım 2: Yeni değişkenler oluşturunuz.

#Tenure (sadakat seviyesi)
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

#Aylık ücret kategorileri
df["MonthlyChargeLevel"] = pd.cut(df["MonthlyCharges"],
                                  bins=[0, 35, 70, df["MonthlyCharges"].max()],
                                  labels=["Low", "Medium", "High"])
# Toplam hizmet sayısı

df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies']]== 'Yes').sum(axis=1)

#Streaming Sayısı (Müşterinin televizyon içeriklerini (dizi/film) internet üzerinden izleme hizmeti var mı)

df["NEW_FLAG_ANY_STREAMING"] = df.apply(lambda x: 1 if (x["StreamingTV"] == "Yes") or (x["StreamingMovies"] == "Yes") else 0, axis=1)

#Güvenlik hizmetleri sayısı

df["SecurityCount"] = df[["OnlineSecurity", "DeviceProtection", "TechSupport"]].apply(
    lambda x: sum(x == "Yes"), axis=1)

# Otomatik ödeme

df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

# Hem telefon hem internet var mı
df["HasPhoneAndInternet"] = np.where(
    (df["PhoneService"] == "Yes") & (df["InternetService"] != "No"), 1, 0
)

# Yeni müşteri mi (tenure < 6)

df["IsYoungCustomer"] = np.where(df["tenure"] < 6, 1, 0)

#Ortalama Aylık gelir

df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1) #tenure 0 dan başladığı için her yeni müşteri 0 olarak kayıtlı olacak, bu nedenle 1 ekledim.

#
df["NEW_Increase"] = df["NEW_AVG_Charges"] / df["MonthlyCharges"]

#Kontrat uzunluğu

df["LongContract"] = df["Contract"].apply(lambda x: 1 if x in ["One year", "Two year"] else 0)

#Aylık Servis ücret ortalaması

df["NEW_AVG_Service_Fee"] = df["MonthlyCharges"] / (df['NEW_TotalServices'] + 1) #müşteri 0 hizmet alabilir, bu nedenle  +1 ekledim.

df.head()

# Adım 3: Encoding işlemlerini gerçekleştiriniz.

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Returns the names of categorical, numeric and categorical but cardinal variables in the data set.
    Note Categorical variables include categorical variables with numeric appearance.

    Parameters
    ------
        dataframe: dataframe
                Variable names of the dataframe to be taken
        cat_th: int, optional
                class threshold for numeric but categorical variables
        car_th: int, optinal
                class threshold for categorical but cardinal variables

    Returns
    ------
        cat_cols: list
                Categorical variable list
        num_cols: list
                Numeric variable list
        cat_but_car: list
                List of cardinal variables with categorical appearance

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.
        The sum of the 3 return lists equals the total number of variables: cat_cols + num_cols + cat_but_car = number of variables

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat

    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]

    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")  # data frame in içerisindeki gözlem sayısına eriştik.
    print(f"Variables: {dataframe.shape[1]}")  # data frame in içerisindeki değişken sayısına eriştik.
    print(f'cat_cols: {len(cat_cols)}')  # kaçtane kategorik değişken olduğunu tespit ettik.
    print(f'num_cols: {len(num_cols)}')  # kaç tane nümerik değişken olduğunu tespit ettik.
    print(f'cat_but_car: {len(cat_but_car)}')  # kaç tane kardinal değişken olduğunu tespit ettik.
    print(
        f'num_but_cat: {len(num_but_cat)}')  # kaç tane numerik gibi görünüp kategorik olan değişken olduğunu belirledi

    return cat_cols, num_cols, cat_but_car, num_but_cat

cat_cols, num_cols, cat_but_car,  num_but_cat = grab_col_names(df)



############################
# ADIM 3: ENCODING
############################

# 1) Grab col names (güncel df üzerinden)
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)

# Churn hedef değişken, numerik listeden çıkaralım
num_cols = [col for col in num_cols if col != "Churn"]

# 2) Label Encoding (binary kategorikler)
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in cat_cols if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)
df.head()
# 3) One-Hot Encoding (3+ sınıflı kategorikler)
ohe_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, ohe_cols)
df.head()


# 6) Eksik veri kontrolü, gerekirse düşür
df = df.dropna()

# 7) customerID modelde işe yaramaz, düşürelim
if "customerID" in df.columns:
    df = df.drop("customerID", axis=1)
df.head()
############################
# ADIM 4: SCALING
############################
# Adım 4: Numerik değişkenler için standartlaştırma yapınız. x_scaled = (x - median) / IQR

scaler = RobustScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head(10)

############################
#### Görev 3 : Modelleme ####
# Adım1 Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
############################

y = df["Churn"]
X = df.drop("Churn", axis=1)

models = [
    ('LR', LogisticRegression(random_state=47, max_iter=1000)),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=47)),
    ('RF', RandomForestClassifier(random_state=47)),
    ('XGB', XGBClassifier(random_state=47, eval_metric="logloss")),
    ("LightGBM", LGBMClassifier(random_state=47)),
    ("CatBoost", CatBoostClassifier(verbose=False, random_state=47))
]

last_models_metrics = []

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=5,
                                scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])

    accuracy = round(cv_results['test_accuracy'].mean(), 4)
    auc = round(cv_results['test_roc_auc'].mean(), 4)
    recall = round(cv_results['test_recall'].mean(), 4)
    precision = round(cv_results['test_precision'].mean(), 4)
    f1 = round(cv_results['test_f1'].mean(), 4)

    last_models_metrics.append({
        "Model": name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Recall": recall,
        "Precision": precision,
        "F1": f1
    })

    print(f"########## {name} ##########")
    print(f"Accuracy: {accuracy}")
    print(f"AUC: {auc}")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"F1: {f1}")






# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli
# tekrar kurunuz.
#Accuracy skorlarına göre en iyi 4 model:
# Logistic Regression (0.8059), CatBoost(0.7975), LightGBM(0.7961), Random Forest(0.7890), XGB (0.7822), KNN (0,7686), CART(0.7291)


lr_params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}  #C, regularizasyon gücünü kontrol eder. logaritmik ölçekte çalışır.
knn_params = {"n_neighbors": [3, 5, 7]}  # Tek sayılar tercih edilir (bağlantı eşitliği olmasın diye) ve Küçük datasetlerde K=3,5,7 klasik olarak en iyi çalışan aralıklardır.
cart_params = {"max_depth": [3, 5, 7]}  #Derinlik arttıkça overfitting artar.  3–7 arası depth bir karar ağacı için “genel optimum banttır”
rf_params = {"n_estimators": [100, 300, 500]}   # Ağaç sayısı arttıkça performans artar ama sınırlı bir noktaya kadar. 100–500 arası pratikte en yaygın kullanılan banttır.
xgb_params = {"learning_rate": [0.01, 0.1], "max_depth": [3, 5, 7]}
lgbm_params = {"learning_rate": [0.01, 0.1], "n_estimators": [100, 300, 500]}
catboost_params = {"iterations": [200, 500], "learning_rate": [0.01, 0.1], "depth": [3, 6]}


classifiers = [
    ("LR", LogisticRegression(), lr_params),
    ("KNN", KNeighborsClassifier(), knn_params),
    ("CART", DecisionTreeClassifier(), cart_params),
    ("RF", RandomForestClassifier(), rf_params),
    ("XGB", XGBClassifier(), xgb_params),
    ("LGBM", LGBMClassifier(), lgbm_params),
    ("CatBoost", CatBoostClassifier(verbose=False), catboost_params)
]
def hyperparameter_optimization(X, y, cv=5, scoring="accuracy"):  # X bagımsız, y bagımlı değişkenler cv=5 5 katlı cross validation
    print("Hiperparametre Optimizasyonu")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} #########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X,y) #GridSearchCV = Menüdeki tüm seçenekleri tek tek denemek.
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After) : {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models
best_models = hyperparameter_optimization(X, y)


#not Örnek:
#LogisticRegression → C, penalty
#KNN → n_neighbors
#CART → max_depth, min_samples_split
#RandomForest → n_estimators, max_depth
#XGBoost → learning_rate, max_depth
#LightGBM → n_estimators, num_leaves
#CatBoost → iterations, depth
#Bu parametreler modelin özyinelemeli olarak kendi öğrendiği şeyler değil,
#eğitim sürecini kontrol eden dış ayarlardır.