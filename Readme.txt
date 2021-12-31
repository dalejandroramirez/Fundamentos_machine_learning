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

- ¿Como llamar un archivo csv?
		data=pd.read_csv("path")
	
- Para tomar un individuo con sus atributos usamos data.iloc[i] donde i e s el iesima fila 


- ¿Como obtener las columans del df?
		data.columns

- Ordenar los elementos
		para ordenar los indices  se usa data.sort_index(axis=0,ascending=False)
