#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import packages
import os
import joblib
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler

# Train Test Split
from sklearn.model_selection import train_test_split

# Model0
from sklearn.linear_model import LogisticRegression,LinearRegression

# Métricas
from sklearn.metrics import accuracy_score, classification_report, roc_curve

# Cross Validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import warnings as wr
wr.filterwarnings("ignore") #to ignore the warnings


# ### 1. Limpieza y tratamiento de Datos

# In[2]:


# Leemos el csv Heart
df=pd.read_csv("heart.csv")

# Buscamos las 5 primeras filas y vemos que datos tenemos
df.head()


# In[3]:


# Contabilizamos el número de filas y columnas que tenemos

print('Número de filas =',df.shape[0], 'y número de columnas =',df.shape[1])


# In[4]:


# Vamos a eliminar una serie de columnas que no nos interesan para nuestro análisis futuro.

df.drop(columns=['caa','restecg','slp','cp','oldpeak'],inplace = True)


# In[5]:


df


# In[6]:


# Ahora vamos a ver que información tenemos de nuestros datos

df.info()


# - Como podemos obervar, no tenemos ningun dato nulo en nuestras columnas. Además, vemos como todos los datos son de tipo enter a excepción de oldpeak que es float.

# In[7]:


# Verificamos que no tengamos valores nulos.

df.isnull().sum()


# - Podemos concluir que tenemos un 0% de valores nulos en nuestros datos, por lo que a priori no deberemos utilizar ninguna fórmula de sunstitución de nulos. 

# In[8]:


# Una vez analizados los valores nulos, vamos a ver si tenemos valores duplicados en nuestro dataset.

df.duplicated().sum()


# In[9]:


# Eliminamos el valor duplicado obtenido y imprimimos por pantalla el número de filas resultantes.

df.drop_duplicates(inplace=True)
print('Number of rows are =',df.shape[0], ',and number of columns are =',df.shape[1])


# In[10]:


# Hecho esto, vamos a ver como nos queda el análisis descriptivo de nuestros datos resultantes.

df.describe().T


# - La presión arterial promedio de un individuo es 131.6 mientras que el valor máximo sube a 200.
# - La frecuencia cardíaca media del grupo es de 149.5, con un máximo de 202. Tambíen vemos que el 75% de los datos tienen valores inferiores a 166.
# - La edad del grupo varía de 29 a 77 años y la media de edad es de 54.42.
# - También podemos observar como tenemos más sex = 1 que igual a 0. Esto lo veamos porque el promedio es superior a 0.5, lo que nos indica que hay más valores igual a 1.

# In[11]:


# Dividimos por el tipo de columnas que tenemos.

cat_cols = ['sex','exng','fbs','thall']
con_cols = ["age","trtbps","chol","thalachh"]
target_col = ["output"]
print("The categorial cols are : ", cat_cols)
print("The continuous cols are : ", con_cols)
print("The target variable is :  ", target_col)


# In[12]:


fig = plt.figure(figsize=(10,8))
gs = fig.add_gridspec(3,3)
gs.update(wspace=0.5, hspace=0.25)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0])


color_palette = ["#800000","#8000ff","#6aac90","#5833ff","#da8829"]

# Sex count
ax0.text(0.3, 220, 'Sex', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax0.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax0,data=df,x='sex',palette=color_palette)
ax0.set_xlabel("")
ax0.set_ylabel("")

# Exng count
ax1.text(0.3, 220, 'Exng', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax1,data=df,x='exng',palette=color_palette)
ax1.set_xlabel("")
ax1.set_ylabel("")

# Fbs count
ax2.text(0.5, 290, 'Fbs', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax2,data=df,x='fbs',palette=color_palette)
ax2.set_xlabel("")
ax2.set_ylabel("")

# Thall count
ax3.text(1.2, 180, 'Thall', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.countplot(ax=ax3,data=df,x='thall',palette=color_palette)
ax3.set_xlabel("")
ax3.set_ylabel("")



# In[13]:


# Ahora vamos a ver como se distribuyen los valores en un boxplot. De este modo, podremos ver si tenemos algún outlier que podamos eliminar.

fig = plt.figure(figsize=(10,8))
gs = fig.add_gridspec(2,3)
gs.update(wspace=0.3, hspace=0.15)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[0,2])
ax3 = fig.add_subplot(gs[1,0])

# Age 
ax0.text(-0.05, 81, 'Age', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax0.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxplot(ax=ax0,y=df['age'],palette=["#800000"],width=0.6)
ax0.set_xlabel("")
ax0.set_ylabel("")

# Trtbps 
ax1.text(-0.05, 208, 'Trtbps', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxplot(ax=ax1,y=df['trtbps'],palette=["#8000ff"],width=0.6)
ax1.set_xlabel("")
ax1.set_ylabel("")

# Chol 
ax2.text(-0.05, 600, 'Chol', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax2.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxplot(ax=ax2,y=df['chol'],palette=["#6aac90"],width=0.6)
ax2.set_xlabel("")
ax2.set_ylabel("")

# Thalachh 
ax3.text(-0.09, 210, 'Thalachh', fontsize=14, fontweight='bold', fontfamily='serif', color="#000000")
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.boxplot(ax=ax3,y=df['thalachh'],palette=["#da8829"],width=0.6)
ax3.set_xlabel("")
ax3.set_ylabel("")


# In[14]:


# Ahora vemos la correlación de núestrass varíables continuas

df_corr = df.corr().transpose()
df_corr


# In[15]:


df.to_csv('heart_final.csv',decimal=",",encoding='utf-16')


# ### 2. Análisis Estadístico de Variables

# In[16]:


from scipy.stats import chi2_contingency

# Creamos una tabla de contingencia para sex y output
ct = pd.crosstab(df["sex"], df["output"])

# Realizamos el chi-cuadrado
chi2, p, dof, expected = chi2_contingency(ct)

# Print
print(f"Chi-square test statistic: {chi2}")
print(f"p-value: {p}")

# Indicamos nuestro nivel de significancia
alpha = 0.05
if p < alpha:
    print("Existe relación entre el sexo y tener o no un infarto")
else:
    print("No existe relación entre el sexo y tener o no un infarto")


# In[17]:


# Creamos un gráfico de barras por sexo.
sns.countplot(x="sex", hue="output", data=df,palette=["#56B4E9","#F0E442"],edgecolor=".6")

# Añadimos labels y título
plt.xlabel("Sex")
plt.ylabel("Count")
plt.title("Stacked Bar Plot of Output by Sex")

plt.show()


# In[18]:


# Hacemos algo parecido pero para ver si hay diferencias entre valores altos y bajos de chol y thalachh

from scipy.stats import norm

success_rate_low = df[(df["chol"] <= 200) & (df["thalachh"] <= 150)]["output"].mean()

success_rate_high = df[(df["chol"] > 200) & (df["thalachh"] > 150)]["output"].mean()

# Calculamos nuestro valor z de contraste
z =(success_rate_high - success_rate_low)/np.sqrt((success_rate_low * (1 - success_rate_low)) / 
                                                            len(df[(df["chol"] > 200) & (df["thalachh"] > 150)]) + 
                                                            (success_rate_high * (1 - success_rate_high)) / 
                                                            len(df[(df["chol"] > 200) & (df["thalachh"] > 150)]))

# Calculamos el p-value gracias a nuestra z.
p = 2 * (1 - norm.cdf(abs(z)))

# Añadimo el valor de significancia.
alpha = 0.05
if p < alpha:
    print(p, "Existe diferencia entre clientes con menos de 200 de colesterol y pacientes con más de 200 en colesterol")
else:
    print(p, "No existe diferencia entre clientes con menos de 200 de colesterol y pacientes con más de 200 en colesterol")


# In[19]:


fig = plt.figure(figsize=(18,18))
gs = fig.add_gridspec(5,2)
gs.update(wspace=0.5, hspace=0.5)
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])

background_color = "#ffe6e6"
color_palette = ["#800000","#8000ff","#6aac90","#5833ff","#da8829"]
fig.patch.set_facecolor(background_color) 
ax0.set_facecolor(background_color) 
ax1.set_facecolor(background_color) 
ax2.set_facecolor(background_color)
ax3.set_facecolor(background_color)


# título Chol
ax0.text(0.5,0.5,"Distribución de chol\nde acuerdo con\n la variable objetivo\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax0.spines["bottom"].set_visible(False)
ax0.set_xticklabels([])
ax0.set_yticklabels([])
ax0.tick_params(left=False, bottom=False)

# Chol
ax1.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax1, data=df, x='chol',hue="output", fill=True,palette=["#8000ff","#da8829"], alpha=.5, linewidth=0)
ax1.set_xlabel("")
ax1.set_ylabel("")


# Título Thalachh
ax2.text(0.5,0.5,"Distribución de thalachh\nde acuerdo con\n la variable objetivo\n___________",
        horizontalalignment = 'center',
        verticalalignment = 'center',
        fontsize = 18,
        fontweight='bold',
        fontfamily='serif',
        color='#000000')
ax2.spines["bottom"].set_visible(False)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.tick_params(left=False, bottom=False)

# Thalachh
ax3.grid(color='#000000', linestyle=':', axis='y', zorder=0,  dashes=(1,5))
sns.kdeplot(ax=ax3, data=df, x='thalachh',hue="output", fill=True,palette=["#8000ff","#da8829"], alpha=.5, linewidth=0)
ax3.set_xlabel("")
ax3.set_ylabel("")

for i in ["top","left","right"]:
    ax0.spines[i].set_visible(False)
    ax1.spines[i].set_visible(False)
    ax2.spines[i].set_visible(False)
    ax3.spines[i].set_visible(False)


# ### 3. Predicción

# In[20]:


# Creamos una copia de df
df1 = df

# DEfinimos las columnas para realizar el encoding y el escalado
cat_cols = ['sex','exng','fbs','thall']
con_cols = ["age","trtbps","chol","thalachh"]

# Realizamos un encoding de las variables categóricas
df1 = pd.get_dummies(df1, columns = cat_cols, drop_first = True)

# Defiimos la variable X e Y
X = df1.drop(['output'],axis=1)
y = df1[['output']]

# Escalamos nuestros datos
scaler = RobustScaler()

X[con_cols] = scaler.fit_transform(X[con_cols])
print("The first 5 rows of X are")
X.head()


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)
print("El tamaño de X_train es  ", X_train.shape)
print("El tamaño de X_test es   ",X_test.shape)
print("El tamaño de y_train es  ",y_train.shape)
print("El tamaño de y_test es   ",y_test.shape)


# In[22]:


# Instanciamos el modelo
model = LogisticRegression()

# Hacemos un fit sobre los datos de entrenamiento
model.fit(X_train, y_train)

# Realizamos el predict sobre los datos de test.
y_pred = model.predict(X_test)

# Vemos la precisión del modelo
print("La accuracy es ", accuracy_score(y_test, y_pred))


# In[23]:


model.intercept_


# In[24]:


model.coef_


# In[25]:


# Realizamos la curva roc.
fpr,tpr,threshols=roc_curve(y_test,y_pred)

plt.plot([0,1],[0,1],"k--",'r+')
plt.plot(fpr,tpr,label='Logistic Regression')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




