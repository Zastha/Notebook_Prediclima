# Notebook_Prediclima
Notebook de Imputacion de datos meteorologicos faltantes
<img width="3999" height="1777" alt="image" src="https://github.com/user-attachments/assets/3dacabe2-89b8-45c9-9803-74f0170449df" />


## Objetivo
El objetivo de investigacion es explorar diferentes metodologias de Machine learning para la imputacion de datos faltantes de regisros meteorologicos.
Para este estudio se utilizaron los datos de estaciones en sinaloa que se encuentran en la pagina publica de la conagua.

Los modelos a evaluar son:
-FBProphet
-XGboost
-Tensorflow
-Arimax

Los resultados seran medidos apartir de las medidas de Error estandar: MAE, MSE, MAPE y RMSE
Se utilizara los datos de la estacion con mayor porcentaje de registros completos
Se dividira dicho registro en una distribucion 70% Entrenamiento/20% Pruebas/ 10% validacion para evitar el sesgo.
Utilizaremos para el analisis de resultados solamente los datos generados para la Region Seco ya que tiene el mayor porcentaje de presencia en la region.
Se puede considerar analizar si ciertos modelos son mas compatibles con ciertas regiones.


# 0 -. Extraccion

# Flujo de trabajo

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/5a5eab75-ea99-4ca4-a036-e68768dbaa2a" />


## 0.1 -. Webscrapeo de datos crudos
- Se obtienen datos historicos diarios de estaciones meteorologicas desde el servidor de Conagua
- Los datos son guardados como .txt

## 0.2 -. Transformacion de datos crudos
- Se extraen los metadatos importantes (Latitud, Longitud, Altitud, Estacion)
- Se limpian los caracteres especiales y unidades
- Se convierte la informacion en un dataframe convirtiendo las columnas numericas y formateando la fecha.
- Los datos son guardados como csv

## 0.3 -. Analisis de calidad de datos. (Analisis de huecos)
- Identifica la continuedad de los datos relevantes (Precip, Evap,Tmax,Tmin)
- Crea un Reporte de huecos para uso futuro
- Se clasifican los huecos como cortos y largos para saber el tipo de imputacion necesaria

## 0.4 -. Evaluacion de integridad y Filtrado
- Se crea un rango de fecha para detectar dias que no aparecian en el archivo original
- Se calcula el porcentaje real de datos faltantes acorde a los dias omitidos.
- Se utiliza un umbral para el filtrado de registros considerados como aceptables para ser utilizados en el entrenamiento.

## 0.5 -. Clasificacion Climatica
- A partir de los datos que cruzan el umbral son clasificados a partir de un archivo creado manualmente
- Las posibles categorizaciones meteorologicas son Seco,Tropical,Templado acorde a la clasificacion Koppen-Geiger, modificada por Garcia (2004)
- Los archivos son guardados acorde a su clasificacion para comparar resultados acorde a las clasificaciones


# 1-.FBProphet

## 1.1 Introduccion

Creado por Taylor y Letham (2017) con el objetivo de facilitar el proceso de creacion de modelos FBProphet ofrece herramientas para la creacion de modelos predictivos sin necesidad de expertos en el area.
Es un modelo con funcionalidades de tendencia en el tiempo para sectores economicos, pero puede ser utilizado para predicciones meteorologicas con la manipulacion suficiente.
FB Prophet se puede resumir como un Modelo Aditivo Generalizado (GAM). Lo cual le da resistencia a datos faltantes al predecir datos futuros.
Este se basa en el ajuste de curvas y no en la autocorrelacion de datos.
Prophet fue nuestro unico modelo que utilizo predicion univariable por que se queria probar las herramientas que ofrecia la plataforma, siendo este el tipo de uso con el que fue creado.


## 1.2 Funcionamiento

<img width="174" height="31" alt="image" src="https://github.com/user-attachments/assets/7b11cd62-63b4-45e0-bf4a-c5f759e002bc" />

El modelo se puede resumir en esta ecuacion
g(t) = Tendencia 
s(t) = Cambios periodicos
h(t) = Cambios por dias festivos

Tendencia = El modelo utiliza crecimiento lineal por tramos, permitiendo la deteccion de puntos de cambio, adaptandose a cambios climaticos o variaciones locales.

Estacionalidad = Como otros modelos, utiliza la serie de Fourier para aproximarse a ciclos periodicos mediante sumas de seno y coseno.
<img width="281" height="58" alt="image" src="https://github.com/user-attachments/assets/ce9b9211-1b35-4553-8f67-d98d848384b6" />


Cambios por dias festivos = Esto le permite flexibilidad al modelo para tomar en cuenta irregularidades en fechas especificas.
<img width="194" height="30" alt="image" src="https://github.com/user-attachments/assets/7ce51b31-a06e-45b9-a13d-3dd2d2cf479e" />
<img width="94" height="29" alt="image" src="https://github.com/user-attachments/assets/44c23338-692e-4440-9b1c-3744a6f7a672" />



## 1.3 Imputacion de datos faltantes
FBProphet tiene una ventaja en contraste con otros modelos, a diferencia de SARIMAX o LSTM, Prophet no necesita una secuencia continua evitando la necesidad de interpolacion o parches parrecidos.
Durante un entrenamiento, si Prophet detecta valores Nulos los omitira al ajustar la curva.
Al imputar datos el modelo puede estimar los valores


## 1.3 Implementacion

### Datos a Ajustar
- changepoint_prior_scale: El parametro mas importante a ajustar. Ajusta la flexibilidad de la tendencia y cuando puede cambiar en los puntos de cambio.
- seasonality_prior_scale: Parametro que presta flexibilidad a la Estacionalidad, un numero alto permite una estacionalidad a tener fluctiaciones altas
- holidays_prior_scale: Permite flexibilidad en el ajuste de efectos de Dias festivos. Para nuestro caso es irrelevante debido a ser datos meteorologicos.
- seasonality_mode: Se puede seleccionar si la estacionalidad sera aditiva o multiplicativa, viendo las series de tiempo y los cambios de magnitud.
- changepoint_range: Proporcion de la historia que la tendencia puede cambiar.

### Otros parametros
Estos otros parametros no son recomendables para cambiar, pero es igualmente importante tener un conocimiento de como afectan el modelo.
- growth: Las opciones son 'linear' y 'logistica'. Es usualmente designado como lineal, a menos que existea un punto de saturacion y crecimiento.
- changepoints: Utilizado para especificar los changepoints manualmente, pero es automatico por defecto.
- n_changepoints: El defecto es 25, y deberian ser suficientes, es mas probable que el cambio al changepoint_prior_scale sea mas relevante.
- yearly_seasonality: Siempre en Auto, para estacionalidad por año. Si es el unico año apagarlo es mas efectivo.
- weekly_seasonality: Igual a yearly_seasonality.
- daily_seasonality: Igual a yearly_seasonality.
- holidays: Sirve para indicar al dataframe dias festivos. Este efecto es mas efectivo por holidays_prior_scale.
- mcmc_samples: pides al modelo que no busque solo "el mejor valor", sino que explore la distribución completa de probabilidad de cada parámetro.
- interval_width: Cambia los intervalos de inseguridad.
- uncertainty_samples: Determina el numero de samples para el Monte Carlo Sampling.


## 1.4 Explicacion de Flujo de codigo:


<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/ec935ada-49a7-46e8-82ab-4b6a0fbbeae8" />



#### 1.4.1-. Configuracion Dinamica:
- Carga de rutas y parámetros específicos por estación (Seco, Templado, Tropical).
- Inyección de hiperparámetros optimizados (changepoint_prior_scale, seasonality_prior_scale) definidos previamente para cada variable.

#### 1.4.2. Preprocesamiento de Series de Tiempo:
- Transformación al formato estricto de Prophet: ds (fecha) y y (valor numérico).
- Limpieza de valores nulos y ordenamiento cronológico.

### 1.4.3 Entrenamiento y Persistencia:
- Ajuste del modelo (fit) con los parámetros configurados.
- Serialización y guardado del modelo entrenado en formato .joblib para uso futuro.

### 1.4.4 Validacion Cruzada (Rolling Origin Cross-Validation):
- Se utiliza cross_validation nativo de Prophet para simular pronósticos históricos.
- Configuración: Entrenamiento inicial de 730 días, con reentrenamientos cada 180 días y predicciones a 1 año (365 días).

### 1.4.5 Analisis de Metrica por Horizonte Temporal:
- El script no solo calcula el error global, sino que segmenta el desempeño en Bins de Tiempo (e.g., 1-3 Días, 1-2 Años, >2 Años).
- Métricas calculadas: MSE, RMSE, MAE, MAPE.

### 1.4.6 Generacion de Reportes Visuales:
- Se generan y guardan automáticamente tres tipos de artefactos en la carpeta Performance_metrics_plots:
  - Tabla de Metricas: Resumen numerico del desempeño por horizonte.
  - Grafico de Barras: Visualización de la degradación del RMSE a lo largo del tiempo.
  - Grafico de Línea: Comparativa visual directa entre los Valores Reales vs. Prediccion (Cross-Validation).
 


# 2-. SARIMAX

## 2.1 Introduccion
El modelo SARIMAX puede ser explicado por el significado de sus letras: Seasonal AutoRegressive  Integrated Moving Average with eXegenous Regressors.
Es un modelo de extension del modelo Arima, el cual permite el modelaje de series temporales con correlaciones entre variables exogenas y componentes estacionales.
La formula de ARIMA seria:
<img width="383" height="63" alt="image" src="https://github.com/user-attachments/assets/df133aa9-362e-4b4e-ba79-0f2539e627c2" />

$\Delta y_t$ = Rerpresenta la serie diferenciada en el tiempo
c = constante
$\phi_1 \Delta y_{t-1}$ (Componente Autoregresivo) = Captura la inercia de la serie.
$\theta_1 \epsilon_{t-1}$ (Componente de Media Movil) = Captura choque o error de prediccion pasada.
$\epsilon_t$(Ruido blanco) = Representa el termino de error aleatorio actuaal.

La presencia de las variables exogenas agregan un termino agregado, lo cual en nuestro caso son las otras columnas aparte de la que se esta imputando, y las series de fourier para estacionalidad.

## 2.2 Parametros:
- Changepoint_prior_scale: Representa la flexibilidad de la tendencia. Valores bajos hacen mas rigido, mientras que altos permiten flexibilidad. Valores como TMAX se prefiere parametros bajos, mientras la precipitacion mejora con valores bajos.
- Seasonality_prior_scale: Es el control de fluctuaciones estacionales. Particularmente util en conjunto con ondas de fourier. Un valor bajo no da tanta importancia a la estacionalidad, uno alto crea una relevancia alta a las estaciones. En nuestro caso la estacionalidad es clave.
- seasonality_mode: Define la interaccion entre estacionalidad y tendencia. No es relevante en nuestro trabajo, la aditiva es la que viene por defecto, la usaremos por que el ciclo anual es constante en el tiempo, incluso si las temporadas de lluvia pueden afectar.
- changepoint_range: Viene por defecto 0.8 e indica los cambios de tendencia que buscara para evitar que el modelo se ajuste a los datos recientes.

## 2.3 Flujo de Trabajo:
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/215c7a91-0607-47a3-9098-9f51952059d3" />

### 2.3.1 Division de Datos.
- Se utilizo una distribucion de 70/20/10. 70$ Entrenamiento 20% Prueba 10% Validacion
- No se utilizo train_test_split debido a la naturaleza de las dependencias temporales de Arima, ya que este necesita una serie de datos sin falta de datos.
- Se utiliza el set de validacion para poder jugar con los hiperparametros sin sesgar la evaluacion final.

### 2.3.2 Preprocesaimento y manejo de huecos:
- Debido a las limitaciones de ARIMA se debia tener dos consideraciones. Los datos debian estar ordenados por fechas diarias. Se debia rellenar los datos faltantes artificialmente
  - .asfreq('D') nos permite asegurar que los datos siguen una frecuencia diaria, ademas de introducir NaN donde hay registros faltantes.
  - Para los datos ficticios utilizamos interpolacion temporal. 

## 2.3.3 Verificacion de Estacionariedad:
-  Para que funcione correctamente SARIMAX se necesita comprobar que el archivo para entrenar es estacionario, es por esto que utilizamos el analisis de estacionariedad de Dickey-Fuller
-  Tambien este analisis nos informa sobre los terminos autoregresivos y de media movil, los cuales funcionan como hiperparametros durante el entrenamiento.
## 2.3.4 Modelado de Estacionalidad:
- Habiendo comprobado la estacionalidad de los datos es fundamental el incorporar las columnas de series de fourier.
- La series de fourier consiste en ondas senoidales y cosenoidales que nos permite representar ciclos anuales sin necesidad de utilizar recursos computacionales para las revisiones de años por medio de s=365 (revisa 365 registros en el pasado).
- Fourier captura la onda anual con pocos terminos.

## 2.3.5 Seleccion de hifdsperparametros (AutoArima):
- En esta seccion se utilizo la herramienta auto_arima para determinar los hiperparametros optimos que mencionamos anteriormente= (p,d,q)
- Los hiperparametros son encontrados midiendo las posibles combinaciones exploradas. Optamos por limitarlo al rango de 0 a 5 cuando vimos que los datos ya no llegaban a 5.

## 2.3.6 Evaluacion por Horizontes temporales y grafica de imputacion vs realidad.
- Como en FB Prophet se utilizaron metricas de medicion de error MSE, MAE, RMSE y MAPE para determinar la efectividad del modelo
- Tambien se determino el error en diferentes huecos para ver su efectividad en cada uno.
- Se creo una grafica para ver la cercania entre los datos imputados y los reales con los datos de validacion para ver la precision del modelo.

# 3-. Tensorflow - Autoencoder BiLSTM
## 3.1 Introduccion
El estado del arte indica que la redes neuronales con mejor rendimiento en imputacion de datos son las Recurrent Neural Networks.
Este tipo de redes permiten que, contrario a modelos como Arima, no aprende apartir de la prediccion del siguiente valor, sino a partir de comprimir la dinamica completa.
<img width="720" height="189" alt="image" src="https://github.com/user-attachments/assets/57066cbb-efef-4088-9ef1-3172594f6249" />

En nuestro caso seleccionamos un LSTM bidireccional, el cual a diferencia de un LSTM estandar (que es igual un modelo solido para series de tiempo), el Bidireccional significa que no solo mira al pasado.
Un BiLSTM utiliza tanto el pasado como el futuro para la prediccion, el cual es ideal para la imputacion de datos faltantes en huecos.
<img width="720" height="254" alt="image" src="https://github.com/user-attachments/assets/706d70a3-ee89-430f-abee-d1ef6b80427a" />

### 3.2 Hiperparametros
-INPUT_SHAPE = Define entrada a la red neuronal. 30 representa los time steps, o ventana de tiempo. Esto ayuda al modelo a entender ciclos lunares y patrones mensuales. Mientras que 4 son nuestras caracteristicas multivariadas.
-LATENT_DIM =Numero de neuronas en las capas ocultas del LSTM y el cuello de botella del autoencoder. Estamos forzando a que el modelo resuma la informacion de un dia en 64 numeros. Si hubiera un numero mas alto probablemente habria overfitting
-BATCH_SIZE = Esto muestra la cantidad de muestras que procesa una red antes de la actualizacion de pesos internos (backpropagation) Se utilizo  este numero por su estabilidad para memorias RAM
-EPOCHS = Numero de repeticiones de vistas hacia los datos. Se utilizo 100 pero tambien se uso Early stopping si el modelo dejaba de mejorar significativamete su error:
-MASKING_RATE = Porcentaje de datos que se destruyen artificialmente durante el entrenamiento. Eso se utiliza junto con Denoising (se explicara despues) para evitar que aprenda a copiar la entrada.
-LEARNING_RATE = Se inicio con una taza baja de estandar de la industria para que el modelo no sea ni muy lento, ni sea imposible en converger.

## 3.3 Flujo de Trabajo
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/35596e53-5fbe-48b8-958c-ee48631fe3f2" />


### 3.3.1 Preprocesamiento de datos.
- Previo a la entrada a una red neuronal, los datos crudos deben ser transformados
  - Escalado: Para que una neurona sea capaz de trabajar con los datos, los datos deben trabajar en un rango de 0 a 1 (Datos escalados por medio de Scaler)
  - Tensores: Los datos tabulares son tranformados en sliding windows para que vea las tendencias en ciertos espacios de datos.
 
### 3.3.2 Arquitectura del modelo BiLSTM Autoencoder
- La estructura de nuestra red neuronal se compone de 3 tensores:
  - Encoder: Comprime los datos procesando una secuencia temporal y extrayendo un vector.
  - Latent space: Representa la dinamica climatica
  - Decoder: Reconstruye los datos regenerando la secuencia original de los 30 dias a partir del vector generado por el encoder.
### 3.3.3 Denoising
- El Denoising consta en la reconstruccion de una señal desde una version corrupta, por lo que hacemos corrupcion intencional por medio de masking.
- Cada iteracion del entrenamiento creamos una matriz de ruido aleatoria del 20% de los datos faltantes, convirtiendolos a 0.
- Entrenamiento: El modelo recibe la entrada rota y se optimiza con backpropagration para minimizar el error a la hora de reconstruir la entrada.

### Validacion y Pruebas
- Prediccion: El modelo imputa los huecos en los tensores de prueba.
- Se aplanan los tensores 3D para que vuelvan a ser de segunda dimension.
- Se retornan los datos a sus unidades reales.
- Se calcula el error con las metricas MAE, MAPE, MSE y RMSE
- Se Genera una comparacion de Datos generados vs. datos reales para validar la precision
- Se imputan los datos faltantes del set de validacion.

# 4-.XGBoost MICE

## 4.1 - Introduccion
Con el fin de la imputacion de variables meteorologicas, se implemento el MICE (Multiple Imputation Chained equations) una metodologia estadistica que es impulsada por el poder algoritmico de XGBOOST.
XGboost cumple el rol de ser un algoritmo de aprendizaje supervisado. Esta basado en arboles de decision y utiliza el Gradient Boosting, la cual es una tecnica de creacion de modelos de forma secuencial aprendiendo de los errores de modelos pasados.
<img width="903" height="245" alt="image" src="https://github.com/user-attachments/assets/16fa9ef2-22bb-44db-b8f7-838f6707636f" />
Lo que XGboost hace con este descenso de gradiente es su regularizacion, preveniendo overfitting.
La ventaja que ofrece xgboost es que las relaciones entre datos meteorologicos no son siempre lineales, y XGboost es capaz de capturar estas relaciones no lineales en comparacion con otros modelos de regresion lineal estandar.
El MICE por su parte es una metodologia iterativa en la cual se rellenan huecos faltantes con un valor aleatorio incial, y utiliza las otra variables para entrenar un modelo que estime el valor faltante incial, pasando a la siguiente columna.

Cabe mencionar que no se utilizo regresion lineal simple para estas iteraciones, sino que usamos IterativeImputer con XGBRegressor, esto debido a que por defecto IterativeInputer utiliza regresion lineal, pero eso es contraintuitivo con el fuerte de XGBoost en comparacion con otros modelos, ademas que los datos climaticos no siguen patrones lineales.

## 4.2- Hiperparametros
- n_estimators = Numero de Arboles, representan las iteraciones de boosting. Como en nuestro caso estamos utilizandolo para 4 columnas, utilizar un numero mas alto resulta inviable. 
 - Segun Chen y Guestrin (2016), 100 arboles capturan la mayoria de patrones sin llegar a rendimientos decrecientes.
- Learning_rate = Controla la magnitud de contribucion de cada arbol de prediccion final. El valor que se eligio es bajo por que queremos una convergencia suave, reduciendo el error. 
- max_depth = Profundidad maxima de un arbol. Las divisiones que puede haber desde la raiz hasta la hoja mas lejana.
  - Dependiuendo de la profundidad es que tan especificas seran las interacciones entre variables, un numero alto puede llevar a sobreajuste.
  - 6 es el valor por defecto, se eligio 5 para que evite memorizar peculiaridades que podrian afectar la imputacion.
- tree_method = Este es el metodo de construccion del arbol, se eligio agrupar los valores continuos en lugar de hacer una division por cada valor posible.
  - Se eligi opor que en series de tiempo larga esto acelera altamente el entrenamiento sin perder precision.
- max_iter = estas son las iteraciones del MICE, y es cuantas veces este mismo recorre las variables.
  - Acorde al estado del arte (Van Buuren, 2011), la convergencia en distribuciones imputadas ocurre rapidamente, por lo que 5 a 10 iteraciones son suficientes para estabilizar la media.

## 4.3 - Flujo de Trabajo
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/ac22267e-4689-42f1-baba-7df25718e004" />

### 4.3.1 Preprocesamiento y Feature Engineering
- Division de datos: Los datos originales son cargados y se segmentan en la distribucion que ya fue vista en SARIMAX de 70% entrenamiento /20% Pruebas /10% Validacion
- Variables temporales: Como xgboost no es un modelo secuencial se deben crear variables explicitas para que el modelo capte ciclicidad en tiempo.
  - Para esto utilizamos las variables dayofyear, month y las transformadas de fourier, creando la contuidad estacional
    
### 4.3.2 Entrenamiento de MICE
- Se configura el imputador iterativo con los parametros optimizados que fueron explorados anteriormente
- El modelo hace el ajuste creando relaciones multivariadas entre las columnas relevantes (Precipitacion, evaporacion, temperatura maxima, temperatura minima)
- Se guarda nuestro imputer para su uso en la siguiente fase

### 4.3.3 Validacion y pruebas
- Por medio de los conjuntos de test y validacion se crea enmascaramiento artificial con horizontes especificos.
- El modelo Imputa estos huecos y se calcula el error comparando el valor artificial con el real (Ground Truth)
- Se crean tablas de visualizacion para ver la precision de los datos artificiales por medio de graficas de comparacion y graficas de los errores calculados.


# 5. Resultados 
## Region Seca
### Evaporacion
#### FB Prophet
<img width="1485" height="510" alt="image" src="https://github.com/user-attachments/assets/61931af8-bf77-4cc2-9f32-f253abd57771" />
<img width="996" height="547" alt="image" src="https://github.com/user-attachments/assets/e244d8bc-f6db-46d9-b91c-7bee4e7df4fa" />

#### SARIMAX
<img width="1335" height="508" alt="image" src="https://github.com/user-attachments/assets/6bff315f-2002-4609-a82f-855d98f89d30" />
<img width="1005" height="470" alt="image" src="https://github.com/user-attachments/assets/0bd938ef-2fe7-4f73-94ff-dd9d4df2f3cd" />

#### Tensorflow
<img width="794" height="328" alt="table_metrics_EVAP" src="https://github.com/user-attachments/assets/91a419a0-ee06-46a9-9210-354a24f9264b" />
<img width="1000" height="400" alt="plot_test_EVAP" src="https://github.com/user-attachments/assets/570d26f5-b2bf-4f15-92db-1fbcf13fe723" />

#### XGBOOST
<img width="794" height="273" alt="tabla_metricas" src="https://github.com/user-attachments/assets/c88daa51-0df2-489e-824d-4fdcc49f9e02" />
<img width="1200" height="500" alt="plot_validacion_EVAP" src="https://github.com/user-attachments/assets/5c9c60fc-9900-4b17-a137-34a60a72e269" />

### Analisis
- En las imagenes se puede visualizar que tensorflow tuvo los mejores resultados con un error absoluto menor.
- Tiene resistencia a huecos largos.
- La graafica de datos ficticios contra reales tienen un seguimiento muy cercano.
- Lo mas probable es que eso sea un resultado a que Evaporacion tenga mas correlaciones entre datos anteriores y posteriores

### Precipitacion
#### FB Prophet
<img width="1485" height="510" alt="25037_PRECIP_grouped_metrics" src="https://github.com/user-attachments/assets/84230e0b-7f88-42ef-b3f2-c754a80a8813" />
<img width="996" height="547" alt="25037_PRECIP_real_vs_pred_plot" src="https://github.com/user-attachments/assets/db331f39-09a3-4798-a8ce-ec55e338c096" />

#### SARIMAX
<img width="1335" height="508" alt="metrics_table_VAL" src="https://github.com/user-attachments/assets/ce4e1664-ebad-4951-8264-e13adae1205f" />
<img width="1026" height="470" alt="line_plot_comparison_VAL" src="https://github.com/user-attachments/assets/82242455-55bf-46c3-90e6-fb8bc88e82ad" />


#### Tensorflow
<img width="794" height="328" alt="table_metrics_PRECIP" src="https://github.com/user-attachments/assets/c8fca9f6-25ae-40cd-b7d1-7464ac45b085" />
<img width="1000" height="400" alt="plot_test_PRECIP" src="https://github.com/user-attachments/assets/55beb3e1-f268-4787-8d55-fe8a7fd7eeb7" />


#### XGBOOST
<img width="794" height="273" alt="tabla_metricas" src="https://github.com/user-attachments/assets/703162dd-c354-41a5-91f3-84c07bbeb7ed" />
<img width="1200" height="500" alt="plot_validacion_PRECIP" src="https://github.com/user-attachments/assets/8dff6c23-14b8-45b4-acc0-86902544361a" />


### Analisis
- El que tuvo mejores resultados fue XGBoost, probablemente por su facilidad de manejar ceros
- Aun que tiene un deterioro en huecos a mas largo plazo
- Esto se debe probablemente a que XGBoost no depende de una secuencia temporal

### TMAX
#### FB Prophet
<img width="1485" height="510" alt="25037_TMAX_grouped_metrics" src="https://github.com/user-attachments/assets/b09c94ba-56f2-4a3b-a93a-af6080445f89" />
<img width="996" height="547" alt="25037_TMAX_real_vs_pred_plot" src="https://github.com/user-attachments/assets/3c4a8b6d-2445-4c1d-835c-df76d6ee159f" />



#### SARIMAX
<img width="1335" height="508" alt="metrics_table_VAL" src="https://github.com/user-attachments/assets/79e8a077-81a1-4d8d-bd02-cd5a0cb3716d" />
<img width="1017" height="470" alt="line_plot_comparison_VAL" src="https://github.com/user-attachments/assets/43db4835-da05-452b-9206-fdf341b18119" />


#### Tensorflow
<img width="794" height="328" alt="table_metrics_TMAX" src="https://github.com/user-attachments/assets/5161c3ac-e704-41e3-9763-f5b7994bfc6a" />
<img width="1000" height="400" alt="plot_test_TMAX" src="https://github.com/user-attachments/assets/e41d9da2-6a5a-41df-a1a5-da58f904d4ac" />



#### XGBOOST
<img width="794" height="273" alt="tabla_metricas" src="https://github.com/user-attachments/assets/dab63466-9a7c-4a76-98c8-3420079b81a2" />
<img width="1200" height="500" alt="plot_validacion_TMAX" src="https://github.com/user-attachments/assets/9b63ef46-3dd8-4ddc-91db-c9feda15190a" />

## Analisis:
- Podemos observar que Tensorflow es muy consistente, manteniendo un error bajo incluso despues de 2 años
- A corta distancia XGboost es muy preciso, por lo que si se necesita imputar huecos de una semana o menos es el indicado


### TMIN

#### FB Prophet
<img width="1485" height="510" alt="25037_TMIN_grouped_metrics" src="https://github.com/user-attachments/assets/eceaace8-eb36-4580-bc2b-2e1210125144" />
<img width="996" height="547" alt="25037_TMIN_real_vs_pred_plot" src="https://github.com/user-attachments/assets/7a7ecafe-b018-412a-ad0e-4a00fbe9daeb" />


#### SARIMAX
<img width="1335" height="508" alt="metrics_table_VAL" src="https://github.com/user-attachments/assets/a368329a-e7ab-4f3d-92d2-170032149d4c" />
<img width="1017" height="470" alt="line_plot_comparison_VAL" src="https://github.com/user-attachments/assets/7ef30e09-c5f8-406c-9d74-01a8362a07ff" />



#### Tensorflow
<img width="794" height="328" alt="table_metrics_TMIN" src="https://github.com/user-attachments/assets/7f5fe487-e105-4cc4-8891-ef959a936fa9" />
<img width="1000" height="400" alt="plot_test_TMIN" src="https://github.com/user-attachments/assets/fc8390ab-6a59-4447-a377-a7de17a625d4" />



#### XGBOOST
<img width="794" height="273" alt="tabla_metricas" src="https://github.com/user-attachments/assets/cb243b91-3586-4cfc-a38b-d8c0f56ff54a" />
<img width="1200" height="500" alt="plot_validacion_TMIN" src="https://github.com/user-attachments/assets/b893715a-54e1-4a1a-a5af-078a0f20f728" />

## Analisis:
- Se observa que tensorflow sigue altamente competitivo tanto en dias cortos como largos
- Es importante mencionar que SARIMAX se ajusta lo suficientemente rapido para ser considerado si se usa con Interpolacion lineal en huecos cortos


# Conclusion:
Creo que es aparente que Tanto tensorflow como XGboost son modelos constantes, XGboost pierde consistencia en series largas, pero tiene la ventaja de ser resistente a ceros, lo cual es fundamental en datos como precipitacion
Por su parte SARIMAX es pobre en imputacion corta, pero rinde muy bien una vez se adapta a la serie.
Prophet tiene un rendimiento promedio, lo cual es de esperarse cuando su funcion es ser plug and play.
Por ultimo, tensorflow BiLSTM tiene un gran potencial para imputacion de datos faltantes seriales puesto que revisa los datos anteriores y posteriores, ademas de contar con estacionalidad gracias a las transformadas de Fourier.

Para concluir, BiLSTM es el modelo mas robusto en todos los niveles, XGBoost sirve en imputaciones cortas y si es un dato que cuenta con ceros o cambios bruscos en los datos, SARIMAX tiene potencial en huecos largos y Prophet es el mas facil de implementar.


## Referencias 
1-.Taylor, S. J., & Letham, B. (2017). Forecasting at scale.
2-. https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning
3-.Hyndman, R. J., & Athanasopoulos, G. (2018). Forecasting: principles and practice. OTexts. 
4-.Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. Journal of the American statistical association.
5-. Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018). BRITS: Bidirectional Recurrent Imputation for Time Series. Advances in Neural Information Processing Systems.
6-.Vincent, P., Larochelle, H., Bengio, Y., & Manzagol, P. A. (2008). Extracting and composing robust features with denoising autoencoders. Proceedings of the 25th international conference on Machine learning.
7-. https://medium.com/@raghavaggarwal0089/bi-lstm-bc3d68da8bd0
8-. Van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate imputation by chained equations in R. Journal of statistical software
9-. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining
10-.Van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. Journal of Statistical Software
11-. García, E. (2004). Modificaciones al sistema de clasificación climática de Köppen. Instituto de Geografía, Universidad Nacional Autónoma de México.
