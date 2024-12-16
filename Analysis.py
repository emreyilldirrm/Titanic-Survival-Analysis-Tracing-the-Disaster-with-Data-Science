import pandas as pd
import  seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from dask.array import block
from pandas import value_counts
from sympy.printing.pretty.pretty_symbology import line_width

matplotlib.use('TkAgg')  # veya 'QtAgg'
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier     #2
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option("display.max_rows", 300)
import pandas as pd

df = pd.read_csv("dataset/train.csv")
df_t = pd.read_csv("dataset/test.csv")


df.head()
df_t.head()
df.shape
df_t.shape

df_t.head()
df_t.shape

# Cabin değikeni base modelde önemli çıkarsa doldur çıkmazsa kaldır

# datayı check etme fotorafını almak
def check_df(dataframe,head=5):
    print("##################Shape###################")
    print(dataframe.shape)
    print("###################Types###################")
    print(dataframe.dtypes)
    print("###################Head####################")
    print(dataframe.head(head))
    print("####################Tail####################")
    print(dataframe.tail(head))
    print("#####################NA######################")
    print(dataframe.isnull().sum())
    print("######################Quantiles##############")
    print(dataframe.describe([0,0.05,0.50,0.95,0.99,1]).T)



check_df(df,5)
check_df(df_t,5)

# verilerin cat , num türü belirleme
def grab_col_names(dataframe,cat_th=10,car_th=20):
    """
      Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.

      Parameters
      ----------
      dataframe: dataframe
          değişken isimleri alınmak istenen dataframe'dir.
      cat_th: int, float
          numerik fakat kategorik olan değişkenler için sınıf eşik değeri
      car_th: int, float
          kategorik fakat kardinal değişkenler için sınıf eşik değeri

      Returns
      -------
      cat_cols: list
          Kategorik değişken listesi
      num_cols: list
          Numerik değişken listesi
      cat_but_car: list
          Kategorik görünümlü kardinal değişken listesi

      Notes
      ------
      cat_cols + num_cols + cat_but_car = toplam değişken sayısı
      num_but_cat cat_cols'un içerisinde.

      """
    cat_cols = [col for col in dataframe
                if str(dataframe[col].dtypes) in ["object","category"]]
    num_but_cat = [col for col in dataframe
                    if dataframe[col].nunique() < 10 and
                   dataframe[col].dtypes in ["int" ,"float"]]
    cat_but_car = [col for col in dataframe if
                   dataframe[col].nunique() > 20 and
                   str(dataframe[col].dtypes) in ["category","object"]]

    cat_cols = cat_cols+num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe if dataframe[col].dtypes in ["float","int"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols,num_cols,cat_but_car

cat_cols,num_cols,cat_but_car =grab_col_names(df)

df.head()


def cat_summary(dataframe,col_name,plot=False):
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)
                         }))
    print("####################################")
        if plot:
        sns.countplot(x=dataframe[col_name],data=dataframe)
        plt.show(block=True)


for col in cat_cols:
    cat_summary(df,col,plot=True)



def num_summary(dataframe,numerical_col,plot=False):
    quantiles = [0.05,0.10,0.20,.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


for col in num_cols:
    num_summary(df,col,plot=True)


# 1 eksik değer tahmini SimpleImputer ile Embarked değişkeni
# Basit bir imputer ile doldurma

check_df(df)
cat_summary(df,"Embarked",plot=True)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")
df["Embarked"] = imputer.fit_transform(df[["Embarked"]]).ravel()

# target değişkene göre göz gezdirme
def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

def target_summary_with_cat(dataframe, target, numerical_col):
    print(dataframe.groupby(numerical_col).agg({target: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Survived", col)

for col in cat_cols:
    target_summary_with_cat(df, "Survived", col)

for col in cat_cols:
    print("==========",col,"==========")
    display(pd.DataFrame({
                        "TARGET_MEAN": df.groupby(col)["Survived"].mean(),
                        "Count":df[col].value_counts(),
                        "Ratio": 100 * df[col].value_counts()/len(df)
                                    }))

df.groupby(["Pclass", "Embarked"]).agg({"Survived": "mean"})
df.groupby(["Sex", "Pclass", "Embarked"]).agg({
                                                "Survived": "mean",
                                                "Age": "mean"})

# ticket öyle bir baktım
df.head()
cat_summary(df,"Ticket",plot=True)
df["Ticket"].value_counts()
df["Ticket"].nunique()
df.groupby("Ticket").agg({"Survived":"mean"})

# Ag değişkeni median ile doldurma
df["Age"] = df["Age"].fillna(df["Age"].median())
df_t["Age"] = df_t["Age"].fillna(df_t["Age"].median())
df_t["Fare"] = df_t["Fare"].fillna(df_t["Fare"].median())
check_df(df)
check_df(df_t)


# Outlierlara bir göz gezdirme
for col in num_cols:
    plt.figure(figsize=(8,4))
    sns.boxplot(x=df[col])
    plt.xticks(rotation=0)
    plt.show(block=True)

# outlierlara alt detayda göz gedirmek
for col in num_cols:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df, x='Pclass',y=df[col],palette='Pastel1')
    plt.show(block=True)

for col in num_cols:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df, x='Pclass', y=df[col], hue='Sex', palette='Pastel1')
    plt.show(block=True)

for col in num_cols:
    plt.figure(figsize=(10,5))
    sns.boxplot(data=df, x='Pclass', y=df[col], hue='Survived', palette='Pastel1')
    plt.show(block=True)


# Correlation Analysis bir göz atma
df.head()
df[num_cols].corr()

f,ax = plt.subplots(figsize=[5,5])
sns.heatmap(df[num_cols].corr(),annot=True,fmt=".2f",ax=ax,cmap="YlGnBu",linewidths=5)
ax.set_title("Correlation Matrix",fontsize=20)
plt.show()

df[num_cols].corrwith(df["Survived"]).sort_values(ascending=False)

corr_data = df[num_cols].corrwith(df["Survived"]).to_frame()
fig , ax = plt.subplots(figsize=(10,5))

sns.heatmap(corr_data,annot=True,fmt=".2f", ax=ax,cmap="YlGnBu",linewidths=5)
plt.show()


# Base Model
dff = df.copy()
dff.drop("PassengerId",axis=1,inplace=True)
dff.head()

cat_cols = [col for col in cat_cols if col not in ["Survived"]]
#cat_col = [col for col in cat_col if col not in ["Churn"]]
cat_cols
dff[cat_cols].head()
dff = dff[[col for col in dff.columns if col not in cat_but_car]]

dff = pd.get_dummies(dff, columns=cat_cols, drop_first=True)

dff.head()

dff.astype(int).head()

y = dff["Survived"]
X = dff.drop(["Survived"],axis=1)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.30,random_state=17)


dff.shape
X_train.shape
y_train.shape

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
#print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")
print(classification_report(y_pred, y_test))

def plot_importance(model, features, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()


plot_importance(catboost_model, X)
##############################################################################################
# Model Improvement and Feature Extraction
##############################################################################################
df.head()
cat_summary(df,"Cabin",plot=True)
df["Cabin"].value_counts()

missing_ratio = df["Cabin"].isna().sum() / len(df) * 100
print(f"Missing Ratio: {missing_ratio:.2f}%")
df = df.drop(columns=["Cabin"])
df_t = df_t.drop(columns=["Cabin"])
df.head()
df_t.head()
# 3 age değişkeni kategorikleştirme (advance)
num_summary(df,"Age",plot=True)
df[df["Age"] > 15].shape[0]
df[df["Age"] < 15].shape[0]
check_df(df["Age"])

df["New_age"] = pd.qcut(df["Age"], 4)  #
df_t["New_age"] = pd.qcut(df["Age"], 4)  #
df["New_age"].value_counts()
df.head()
df_t.head()

# Pclass", "Embarked","new_age değişkenleri üzerinden hayatta kalma gözlemi (advance)
df.groupby(["Pclass","Embarked","New_age"]).agg({"Survived":"mean"})
# "Sex" , Pclass", "Embarked","new_age değişkenleri üzerinden hayatta kalma gözlemi (advance)
result_1 = df.groupby(["Sex","Pclass","Embarked","New_age"]).agg(
    survived_mean=("Survived", "mean"),
    survived_count=("Survived", "count")
    )

# Hayatta kalan 10 ve üzeri sınıflar bilgisi filtreleme (advance)
result_1.loc[result_1["survived_count"] >= 10 ]


##############################################################################################
# Parch değişkeni kategorikleştirme (advance)
##############################################################################################
check_df(df["Parch"])
df["Parch"].value_counts()

# Pclass", "Embarked","new_age","Parch" değişkenleri üzerinden hayatta kalma gözlemi (advance)
result_2 = (df.groupby(["Sex","Pclass","Embarked","New_age","Parch"]).agg(
    survived_mean=("Survived", "mean"),
    survived_count=("Survived", "count")
    ))
# Hayatta kalan 10 ve üzeri sınıflar bilgisi filtreleme (advance)
result_2.loc[result_2["survived_count"] >= 30 ]

df.groupby(["Parch"]).agg(
    survived_mean=("Survived", "mean"),
    survived_count=("Survived", "count")
    )

df["New_Parch"] = pd.qcut(df["Parch"], 5, duplicates='drop')
df_t["New_Parch"] = pd.qcut(df["Parch"], 5, duplicates='drop')
 #
df["New_Parch"].value_counts()
df.head()
df_t.head()

result_3 = (df.groupby(["Sex","Pclass","New_Parch","Embarked","New_age"]).agg(
    survived_mean=("Survived", "mean"),
    survived_count=("Survived", "count")
    ))

result_3.loc[result_3["survived_count"] >= 40 ]
##############################################################################################


#********************************************************************************************
#****                                                                                    ****
#********************************************************************************************


##############################################################################################
# SibSp değişkeni kategorikleştirme (advance)
##############################################################################################
df.head()

df["SibSp"].value_counts()
df["New_SibSp"] = pd.qcut(df["SibSp"], 4, duplicates='drop')  # 4 bin kullanarak
df_t["New_SibSp"] = pd.qcut(df["SibSp"], 4, duplicates='drop')  # 4 bin kullanarak
df["New_SibSp"].value_counts()

df.head()
df_t.head()
result_4 = df.groupby(["New_SibSp"],observed=True).agg(
    survived_mean=("Survived", "mean"),
    survived_count=("Survived", "count")
)

result_5 = (df.groupby(["Sex","Pclass","New_Parch","New_SibSp","Embarked","New_age"]).agg(
    survived_mean=("Survived", "mean"),
    survived_count=("Survived", "count")
    ))
result_5.loc[result_5["survived_count"] >= 10 ]
#********************************************************************************************


#********************************************************************************************
#****                                                                                    ****
#********************************************************************************************



##############################################################################################
#sibss ile parc birleştirilip family size oluşturulacak
df.head()
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
df["FamilySize"] = np.where(df["Family_Size"] > 1, 0, 1)
df_t["Family_Size"] = df_t["SibSp"] + df_t["Parch"] + 1
df_t["Family_Size"] = np.where(df_t["Family_Size"] > 1, 0, 1)

#bu formattada yapılabilir
#df.loc[df["Family_size"] > 1, "FamilySize"] = 0
#df.loc[df["Family_size"] <= 1, "FamilySize"] = 1
df.head()
df_t.head()
#############################################################################################
#********************************************************************************************


#********************************************************************************************
#****                                                                                    ****
#********************************************************************************************


##############################################################################################
df.head()
df["Ticket"].value_counts()
ticket_counts = df["Ticket"].value_counts()
frequent_tickets = ticket_counts[ticket_counts > 5].index
len(frequent_tickets)
df["Ticket"].nunique()
df["Ticket"].isnull().sum()
check_df(df["Ticket"])
cat_summary(df,"Ticket",plot=True)

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram


# Kategoriler için hedef değişken ortalamaları
target_means = df.groupby('Ticket')['Survived'].mean()

# "Condensed" mesafe matrisi oluştur
distance_matrix = pdist(target_means.values.reshape(-1, 1), metric='euclidean')

# Hiyerarşik kümeleme
linkage_matrix = linkage(distance_matrix, method='ward')

# Dendrogram görselleştirme
plt.figure(figsize=(10, 7))
dendrogram(linkage_matrix, labels=target_means.index, orientation='right')
plt.show()

result_6 = df.groupby("Ticket").agg(
    survived_mean=("Survived", "mean"),
    survived_count=("Survived", "count")
)
result_6.loc[result_6["survived_count"] >= 7 ]
result_7 = result_6.loc[result_6["survived_mean"] > 0 ]

len(result_7)
result_7 = result_7.reset_index()
# result_7["Ticket"] değerlerini bir listeye dönüştür
matching_tickets = result_7["Ticket"].unique()

# df'de yeni Ticket_Survived değişkenini oluştur
df["Ticket_Survived"] = df["Ticket"].apply(lambda x: 1 if x in matching_tickets else 0)
df.head()

# df'de yeni Ticket_Survived değişkenini oluştur
df_t["Ticket_Survived"] = df_t["Ticket"].apply(lambda x: 1 if x in matching_tickets else 0)
df_t.head()



df.drop("Ticket",axis=1,inplace=True)
df_t.drop("Ticket",axis=1,inplace=True)
df.head()
df_t.head()

##############################################################################################


#********************************************************************************************
#****                                                                                    ****
#********************************************************************************************


##############################################################################################
df["Age"].value_counts()
df["Age"].nunique()

df["Fare"].value_counts()
df["Fare"].nunique()

df.head()
##############################################################################################


#********************************************************************************************
#****                                                                                    ****
#********************************************************************************************
# Name değişkeni ünvanları ayırma
#********************************************************************************************
df['title'] = df['Name'].str.extract(r',\s*(\w+)\.')
df_t['title'] = df_t['Name'].str.extract(r',\s*(\w+)\.')
df["title"].value_counts()
df.head()
df_t.head()
df.groupby("title")["Survived"].sum()
##############################################################################################


#********************************************************************************************
#****                                                                                    ****
#********************************************************************************************


##############################################################################################
# değişken düzenleme silme (advance)
# sibs parch ham değişkenlerini sil kategorik hale getirdim ana bilgilerini

df.head()
#df.drop("SibSp",axis=1,inplace=True)
#df.drop("Parch",axis=1,inplace=True)
df.drop("Name",axis=1,inplace=True)
df_t.drop("Name",axis=1,inplace=True)
df.head()
df_t.head()

check_df(df_t)
##############################################################################################


#********************************************************************************************
#****                                                                                    ****
#********************************************************************************************


##############################################################################################
# Outlier Analysis
dff = df.copy()
dff_t = df_t.copy()
check_df(dff)
check_df(dff_t)

def outlier_threshold(data, col_name, w1=0.05, w2=0.95):
    q1 = data[col_name].quantile(w1)
    q3 = data[col_name].quantile(w2)
    IQR = q3 - q1
    up = q3 + 1.5 * IQR
    low = q1 - 1.5 * IQR
    return up,low
####################################
def check_outlier(data, col_name, w1=0.05, w2=0.95):
    up, low = outlier_threshold(data, col_name, w1, w2)
    if data[(data[col_name]<low) | (data[col_name]>up)][col_name].any(axis=None):
        return True
    else:
        return False
####################################
def grab_outliers(data, col_name, index=False, w1=0.05, w2=0.95):
    up, low = outlier_threshold(data, col_name, w1, w2)
    if data[(data[col_name] < low) | (data[col_name] > up)][col_name].shape[0]>10:
        print(data[(data[col_name] < low) | (data[col_name] > up)][col_name].shape[0])
    else:
        print(data[(data[col_name] < low) | (data[col_name] > up)][col_name])
    if index:
        outlier_index = data[(data[col_name] < low) | (data[col_name] > up)][col_name].index
        return outlier_index
#####################################
def remove_outlier(data, col_name, w1=0.05, w2=0.95):
    up, low = outlier_threshold(data, col_name, w1, w2)
    df_without_outliers = data[~(data[col_name]<low)|(data[col_name]>up)]
    return df_without_outliers
######################################
def replace_with_thresholds(data, col_name, w1=0.05, w2=0.95):
    up, low = outlier_threshold(data, col_name, w1, w2)
    data.loc[data[col_name] > up, col_name] = up
    data.loc[data[col_name] < low, col_name] = low

cat_cols, num_cols, cat_but_car = grab_col_names(dff)
cat_cols_t, num_cols_t, cat_but_car_t = grab_col_names(dff_t)

for col in num_cols:
    print(col, check_outlier(dff, col))

for col in num_cols_t:
    print(col, check_outlier(dff_t, col))


for col in num_cols:
    print(col,check_outlier(dff,col))
    if check_outlier(dff,col):
        replace_with_thresholds(dff, col)


for col in num_cols_t:
    print(col,check_outlier(dff_t,col))
    if check_outlier(dff_t,col):
        replace_with_thresholds(dff_t, col)


##############################################################################################
# Advance Model
##############################################################################################


dff.index = dff["PassengerId"]
dff_t.index = dff_t["PassengerId"]
dff.drop("PassengerId",axis=1,inplace=True)
dff_t.drop("PassengerId",axis=1,inplace=True)

cat_cols,num_cols,cat_but_car =grab_col_names(dff)
cat_cols_t, num_cols_t, cat_but_car_t = grab_col_names(dff_t)
cat_cols = [col for col in cat_cols if col not in ["Survived"]]
dff = pd.get_dummies(dff, columns=cat_cols, drop_first=True)
dff_t = pd.get_dummies(dff_t, columns=cat_cols_t, drop_first=True)

dff.head()
dff_t.head()

dff.astype(int)
dff_t.astype(int)


X = dff.drop(["Survived"], axis=1) #bu
y = dff["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=17)

X_train.shape
X_test.shape

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
#print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")
print(classification_report(y_pred, y_test))

plot_importance(catboost_model, X)



# Random Forest
rf_model = RandomForestClassifier(random_state=17).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")
print(classification_report(y_pred, y_test))
plot_importance(rf_model, X_train)

#Cart
cart_model = DecisionTreeClassifier(random_state=17).fit(X_train, y_train)
y_pred = cart_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 4)}")
print(f"Recall: {round(recall_score(y_pred,y_test),4)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 4)}")
print(f"F1: {round(f1_score(y_pred,y_test), 4)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 4)}")
print(classification_report(y_pred, y_test))


def plot_stem(model, features, num=None):
    if num is None:
        num = len(features.columns)

    feature_imp = pd.DataFrame({
        'Value': model.feature_importances_,
        'Feature': features.columns
    })

    feature_imp = feature_imp.sort_values(by="Value", ascending=False)[:num]

    plt.figure(figsize=(12, 8))
    sns.set(style="whitegrid", font_scale=1.1)

    # Sütunlu çizgi grafiği (stem plot)
    markerline, stemlines, baseline = plt.stem(
        feature_imp['Feature'],
        feature_imp['Value'],
        basefmt=" ",
        linefmt='b-',
        markerfmt='bo'
    )
    plt.setp(stemlines, 'linewidth', 2)
    plt.setp(markerline, markersize=6)

    plt.title('Feature Importance ', fontsize=16, weight='bold', color='darkblue')
    plt.xlabel('Features', fontsize=14, weight='bold')
    plt.ylabel('Importance Score', fontsize=14, weight='bold')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


plot_stem(cart_model, X_train)

# Linear Reg
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error


from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet


num_cols = [col for col in num_cols if col != "Survived"]  # Hedef değişkeni çıkarıyoruz

# Scaler işlemi
scaler = StandardScaler()
dff[num_cols] = scaler.fit_transform(dff[num_cols])

# Hedef ve özellik setleri
X = dff.drop(["Survived"], axis=1)
y = dff["Survived"]

# Eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Modeli eğitme
lr_model = LinearRegression().fit(X_train, y_train)
y_lr_pred = lr_model.predict(X_test)

# Hata metriği
lr_model_base_error = np.sqrt(mean_squared_error(y_test, y_lr_pred))
lr_model_base_error

def plot_importance_linear_reg(model, X_train):
    # Katsayıları al
    importance = model.coef_.flatten()  # .flatten() ile katsayıları tek boyutlu diziye çeviriyoruz

    # Özelliklerin isimlerini al (varsa)
    feature_names = X_train.columns if hasattr(X_train, 'columns') else np.arange(X_train.shape[1])

    # Önem sırasına göre sıralama
    sorted_idx = np.argsort(np.abs(importance))[::-1]

    # Grafiği oluştur
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), np.abs(importance[sorted_idx]), align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Katsayı Değeri')
    plt.title('Linear Regression')
    plt.show()


plot_importance_linear_reg(lr_model, X_train)



# LGBM
# Özellik adlarındaki özel karakterleri temizleme
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import numpy as np

# Hedef değişken ve özellik setini ayırma
X = dff.drop(["Survived"], axis=1)
y = dff["Survived"]

# Eğitim ve test setine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

# Kolon adlarını düzenleme
X_train.columns = X_train.columns.str.replace(r'[^\w\s]', '', regex=True)
X_test.columns = X_test.columns.str.replace(r'[^\w\s]', '', regex=True)

# 1. Scalersız LightGBM modeli
lgbm_model_raw = LGBMRegressor(random_state=42).fit(X_train, y_train)
y_lgbm_pred_raw = lgbm_model_raw.predict(X_test)
lgbm_raw_error = np.sqrt(mean_squared_error(y_test, y_lgbm_pred_raw))
print(f"LightGBM (scalersız) RMSE: {lgbm_raw_error}")

# 2. StandardScaler ile ölçeklendirme normalde uygulanmaz ağaç tabanlı bir model olduğu için ben denemek istedim
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Scalerlı LightGBM modeli
lgbm_model_scaled = LGBMRegressor(random_state=42).fit(X_train_scaled, y_train)
y_lgbm_pred_scaled = lgbm_model_scaled.predict(X_test_scaled)
lgbm_scaled_error = np.sqrt(mean_squared_error(y_test, y_lgbm_pred_scaled))
print(f"LightGBM (StandardScaler ile) RMSE: {lgbm_scaled_error}")




def plot_bar_importance2(model, X, orientation='horizontal'):
    # Modelin özellik önemlerini al
    feature_importances = model.feature_importances_

    # Özellik isimlerini al
    features = X.columns

    # Özellikler ve önem derecelerini birleştirip bir DataFrame oluştur
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    })

    # Özellikleri önem derecelerine göre sıralama
    importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)

    # Sıralanmış özellikleri ve önem derecelerini al
    sorted_features = importance_df_sorted['Feature']
    sorted_importances = importance_df_sorted['Importance']

    # Grafik yönü kontrolü: 'horizontal' veya 'vertical'
    if orientation == 'horizontal':
        # Yatay bar grafiği çizme
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_importances, y=sorted_features, palette='viridis')
        plt.xlabel('Importance')
        plt.ylabel('Features')
    else:
        # Dikey bar grafiği çizme
        plt.figure(figsize=(10, 6))
        sns.barplot(x=sorted_features, y=sorted_importances, palette='viridis')
        plt.ylabel('Importance')
        plt.xlabel('Features')

    # Başlık ve etiketler
    plt.title('Feature Importance Based on LightGBM Model')
    plt.xticks(rotation=45)

    plt.show()


plot_bar_importance2(lgbm_model, X_train, orientation='horizontal')


###############################Final##############################################

# 2. Eğitim ve hedef değişkenlerini ayıralım
y = dff["Survived"]  # Hedef değişken (Survived)
X = dff.drop(["Survived"], axis=1)  # Eğitim verisinin geri kalan özellikler

# 3. Eğitim ve test verisi arasındaki ortak kolonları bulalım
common_columns = list(set(X.columns).intersection(dff_t.columns))

# 4. Eğitim ve test verilerini ortak kolonlarla düzenleyelim
X_train_common = X[common_columns]
dff_t_common = dff_t[common_columns]

# 5. Modeli eğitelim
rf_model = RandomForestClassifier(random_state=17).fit(X_train_common, y)

# 6. Test verisi üzerinde tahmin yapalım
y_pred_test = rf_model.predict(dff_t_common)

# 7. Hyperparameter Optimization
from sklearn.model_selection import GridSearchCV, cross_validate
rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [8, 10, None],
             "max_features": [10, 15, 20],
             "min_samples_split": [2,10],
             "n_estimators": [100]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
rf_best_grid.best_params_
rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)
y_pred = rf_final.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
#print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")
print(classification_report(y_pred, y_test))

# 8. Tahmin sonuçlarını kaydedelim
submission = pd.DataFrame({
    "PassengerId": dff_t.index,  # Test verisindeki PassengerId'yi ekliyoruz
    "Survived": y_pred_test  # Tahmin edilen Survived değerini ekliyoruz
})

submission.to_csv("submission.csv", index=False)  # CSV'ye kaydet
print("Test verisi üzerinde tahmin yapıldı ve submission.csv kaydedildi!")