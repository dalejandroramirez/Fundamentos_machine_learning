Machine Learning
Capacidad de un algoritmo de adquirir conocimiento a partir de los datos recolectados para mejorar, 
describir y predecir resultados

Estrategias de Aprendizaje:

	-Aprendizaje Supervisado: Permite al algoritmo aprender a partir de datos previamente etiquetados.

	-Aprendizaje no Supervisado: El algoritmo aprende de datos sin etiquetas, es decir encuentra similitudes 
		y relaciones, agrupando y clasificando los datos.

	-Aprendizaje Profundo (Deep Learning): Está basado en redes Neuronales

Importancia del ML
	Permite a los algoritmos aprender a partir de datos históricos recolectados por las empresas permitiendo
	 así tomar mejores decisiones.

Pasos a Seguir para Desarrollar un modelo en ML

Definición del Problema: Es necesario definir previamente el problema que va a resolver nuestro algoritmo
Construcción de un modelo y Evaluación: Una vez definido el problema procedemos a tratar los datos y a entrenar
nuestro modelo que debe tener una evaluación cercana al 100% 

Deploy y mejoras: El algoritmo es llevado a producción (aplicación o entorno para el que fue creado), 
en este entorno podemos realizar las mejoras pertinentes de acuerdo al comportamiento con los usuario e 
incluso aprovechando los datos generados en esta interacción.
_________________________________________________________________________________________
		Pandas
_________________________________________________________________________________________
- Para crear una serie temporal se usa el modulo Series de pandas

		series = pd.Series([1,2,3,4,5,6,6,7])

-¿Como escoger columnas particulares?
		Por ejemplo para extrar dos columnas con nomres col1 y col2
			df[["col1","col2"]]
		Para extrar una columna en particular
			colum=df.nombre_columna

-Describe una columna:
data[‘columna’].describe()

- ¿Como llamar un archivo csv?
		data=pd.read_csv("path")
	
- Para tomar un individuo con sus atributos usamos data.iloc[i] donde i e s el iesima fila 


- ¿Como obtener las columans del df?
		data.columns

- Ordenar los elementos
		para ordenar los indices  se usa data.sort_index(axis=0,ascending=False)


-Crear arreglo de NxM:
	np.full( (n, m), x )
		Ej.
			np.full( (3, 5), 10)
			Resultado:
				array([
				[10, 10, 10, 10, 10],
				[10, 10, 10, 10, 10],
				[10, 10, 10, 10, 10]
				])
_________________________________________________________________________________________
  									Scikit Learn
_________________________________________________________________________________________

Scikit Learn es una biblioteca de Python que está conformada por algoritmos de clasificación,
 regresión, reducción de la dimensionalidad y clustering. Es una biblioteca clave en la 
 aplicación de algoritmos de Machine Learning, tiene los métodos básicos para llamar un algoritmo,
  dividir los datos en entrenamiento y prueba, entrenarlo, predecir y ponerlo a prueba.


-División del conjunto de datos para entrenamiento y pruebas:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

-Entrenar modelo:
[modelo].fit(X_train, y_train)

-Predicción del modelo:
Y_pred = [modelo].predict(X_test)

-Matriz de confusión:
metrics.confusion_matrix(y_test, y_pred)

-Calcular la exactitud:
metrics.accuracy_score(y_test, y_pred)

_________________________________________________________________________________________

Overfiting and underfiting
_________________________________________________________________________________________

Sobreajunte (overfiting): 
	Es cuando intentamos obligar a nuestro algoritmo a que se ajuste 
	demasiado  a todos los datos posibles. Es muy importante proveer con información abundante 
	a nuestro modelo pero también esta debe ser lo suficientemente variada para que nuestro 
	algoritmo pueda generalizar lo aprendido.

Subajuste (underfiting): 
	Es cuando le suministramo a nuestro modelo un conjunto de datos es
	muy pequeño, en este caso nuestro modelo no sera capas de aprender lo suficiente ya que tiene
	muy poca infomación. La recomendación cuando se tienen muy pocos datos es usar el 70% de los 
	datos para que el algoritmo aprenda y usar el resto para entrenamiento.

_________________________________________________________________________________________

Los modelos de clasificación son capaces de predecir cuál es la etiqueta correspondiente a
cada ejemplo o instancia basado en aquello que ha aprendido del conjunto de datos de entrenamiento.
Estos modelos necesitan ser evaluados de alguna manera y posteriormente comparar los resultados 
obtenidos con aquellos que fueron entrenados.

Una manera de hacerlo es mediante la matriz de confusión la cual nos permite evaluar el desempeño
de un algoritmo de clasificación a partir del conteo de los aciertos y errores en cada una de las
clases del algoritmo.

Como su nombre lo dice tenemos una matriz que nos ayuda a evaluar la predicción mediante positivos
y negativos


Los verdaderos positivos (VP) son aquellos que fueron clasificados correctamente como positivos
como el modelo.

Los verdaderos negativos (VN) corresponden a la cantidad de negativos que fueron clasificados 
correctamente como negativos por el modelo.

Los falsos negativos (FN) es la cantidad de positivos que fueron clasificados incorrectamente como 
negativos.

Los falsos positivos (FP) indican la cantidad de negativos que fueron clasificados incorrectamente 
como positivos.

la entrada 1-1 es Verdadero Positivo
la entrada 2-1 es False Positive
la entrada 2-1 es False Negative
la entrada 2-2 es True Negative 

_________________________________________________________________________________________
Arbol de decisiones
_________________________________________________________________________________________

Árboles de decisión:

**Ventajas: **

	-Claridad en los datos
	-Tolerantes al ruido y datos faltantes
	-Las reglas extraídas permiten hacer extracciones
	-Desventajas:
		-Criterio de división es deficiente
		-Sobreajuste-overfitting
		-Ramas poco significativas
	-Optimización del modelo:
		-Evitar sobreajuste
		-Selección efectiva de los atributos
		-Campos nulos
_________________________________________________________________________________________
Notas	
_________________________________________________________________________________________

El comando para poder ver si no existen nulls .info()
algunas cosas que comentar que pueden ser utiles

cuando se hace el drop intentando eliminar algunas variables que no serán de interés, axis = 1 
indica que estas variables son "columnas ", axis = 0 , indicaría que son filas.

las dummy variables que se mencionan ligeramente convierten las variables categóricas en
indicadoras como 0,1,2,…etc

cuando se completaron los valores faltantes en las variables edad y la clase del pasajero (embarked),
faltó mencionar un comando muy util para saber en que variables se tienen valores faltantes. 
Se puede usar train_df.isnull.any().


-Cuando se llenan los espacios con vacíos (fillna), para el caso de datos numéricos se utiliza 
la mediana porque es una de las medidas de tendencia central que menos se afecta por los datos atípicos.

Para el caso de los datos categóricos relacionados con el embarque, se utiliza la letra S porque 
representa el embarque en la ciudad de South Hampton, en donde más personas se unieron al viaje.

Estos datos se obtienen de un análisis previo a los datos trabajados.
_________________________________________________________________________________________
K-mean
_________________________________________________________________________________________
Algoritmo no supervisado.
	- Crea K grupos a partir de observaciones de un set de datos.
	- Trata información que no tiene etiquetas asignadas.
	- Agrupa información basada en sus características.
	- K = centroides
	- Aproximación a K: método del codo
	
	Aplicaciones:
	- 	Segmentación por comportamiento: 
		-	por historial de compras
		- 	actividad en una aplicación móvil, web
		- 	Definir personas basadas en sus intereses.
		- 	Crear perfiles basado en el monitoreo de actividad.

	- 	Ordenando medidas de sensores:
		- 	Detecta tipos de actividades en sensores de movimiento.
		- 	Grupos de imágenes.
		- 	Separar audio.
		- 	Identificar grupos en monitoreo de salud
_________________________________________________________________________________________
Notas
_________________________________________________________________________________________
Inicialmente iris es un array de datos
y si se le quiere colocar columnas con nombres
basta con usar columns=[nombre_columnas]

Ejemplo:
	x=pd.DataFrame(iris.data,columns=["Sepal Length","Sepal width","Petal length","Petal width"])
	y=pd.DataFrame(iris.target,columns=["Target"])

-
En el dataset de vinos, la razón por la que debemos normalizar los datos antes de aplicar el modelo
 es porque estamos trabajando con datos que miden diferentes cosas, como la intensidad del color, 
 la alcalinidad, el grado de alcohol, etc, estos son valores que no pueden ser comparados entre sí 
 (es como sumar peras con manzanas) y por eso se debe estandarizar, es decir, aplicar un proceso
  estadístico que hace que todos los datos se ajusten a una curva de distribución normal y se situen
   en una misma escala que permita su comparacion, por eso es que el accuracy aumenta desde un 36-37%, 
   hasta un 89%

