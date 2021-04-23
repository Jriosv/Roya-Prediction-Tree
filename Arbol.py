from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import pandas as pd
from sklearn import tree
import graphviz
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if __name__==("__main__"):
    archivo = pd.read_csv('datos.csv')

paraX = ['ph','soil_temperature', 'soil_moisture', 'illuminance', 'env_temperature','env_humidity']
paraY = ['label_yes']
x_train = archivo[paraX].values
y_train = archivo[paraY].values

numeros = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']

profunCent = True

while profunCent:
    auxiliar = input('Profundidad del arbol: ')
    if auxiliar not in numeros:
        print("Error: Ingrese un numero entero del 1 al 20")
    else:
        profundidad = int(auxiliar)
        profunCent = False

arbol = tree.DecisionTreeClassifier(criterion= 'gini', max_depth=profundidad)
arbol = arbol.fit(x_train,y_train)
#tree.plot_tree(arbol.fit(x_train,y_train))

phCentinela = True
while phCentinela:
    auxiliar = input('ph: ')
    try:
        if "." in auxiliar :
            ph = float(auxiliar)
            phCentinela = False
        else:
            ph = int(auxiliar)
            phCentinela = False
    except ValueError:
        print ("Error: ingrese un valor flotante")
stemCentinela = True
while stemCentinela:
    auxiliar = input('Temperatura del suelo: ')
    try:
        if "." in auxiliar :
            stemp = float(auxiliar)
            stemCentinela = False
        else:
            stemp = int(auxiliar)
            stemCentinela = False
    except ValueError:
        print ("Error: ingrese un valor flotante")
shumCentinela = True
while shumCentinela:
    auxiliar = input('Humedad del suelo: ')
    try:
        if "." in auxiliar :
            shum = float(auxiliar)
            shumCentinela = False
        else:
            shum = int(auxiliar)
            shumCentinela = False
    except ValueError:
        print ("Error: ingrese un valor flotante")


luzCentinela = True
while luzCentinela:
    auxiliar = input('Iluminacion: ')
    try:
        if "." in auxiliar :
            luz = float(auxiliar)
            luzCentinela = False
        else:
            luz = int(auxiliar)
            luzCentinela = False
    except ValueError:
        print ("Error: ingrese un valor flotante")


etempCentinela = True
while etempCentinela:
    auxiliar = input('Temperatura ambiente: ')
    try:
        if "." in auxiliar :
            etemp = float(auxiliar)
            etempCentinela = False
        else:
            etemp = int(auxiliar)
            etempCentinela = False
    except ValueError:
        print ("Error: ingrese un valor flotante")

ehumCentinela = True
while ehumCentinela:
    auxiliar = input('Humedad ambiente: ')
    try:
        if "." in auxiliar :
            ehum = float(auxiliar)
            ehumCentinela = False
        else:
            ehum = int(auxiliar)
            ehumCentinela = False
    except ValueError:
        print ("Error: ingrese un valor flotante")


probabilidad = arbol.predict([[ph, stemp,shum,luz,etemp,ehum]])

if probabilidad == 1:
    print("La planta tiene roya")
else:
    print("La planta no tiene roya")

certeza = arbol.score(x_train,y_train)*100
certeza = round(certeza,2)
print("La certeza de la prediccion es de: " + str(certeza)+"%")

dot_data = tree.export_graphviz(arbol, out_file=None,filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("iris")

lprobabilidad = ['ph','stemp','shum','luz','etemp','ehum'] 
y_pos=np.arange(len(lprobabilidad))
scores = arbol.feature_importances_
plt.barh(y_pos,scores,align='center',alpha=0.4)
plt.title('Razones para tomar la decision')
plt.yticks(y_pos,lprobabilidad)
plt.xlabel('Nivel de importancia')
plt.show()
