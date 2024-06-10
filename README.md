<center><h1>Code Exercise DS</h1></center>
<h2>Contenido</h2>

- [Objetivo](#objetivo)
- [Pasos para instalación](#pasos-para-instalación)
- [Variables](#variables)
    - [Variables no usadas](#variables-no-usadas)
- [Resultados](#resultados)
- [Puesta en producción](#puesta-en-producción)
- [Por mejorar](#por-mejorar)
- [Tecnologías](#tecnologías)
- [Recursos](#recursos)
- [Licencia](#licencia)
- [Autor](#autor)

## Objetivo
Usando la base de datos `MLA_100k_checked_v3.jsonlines` estimar un modelo de aprendizaje automático que permita predecir si un item del marketplace es nuevo o usado.

## Pasos para instalación
1. Clone el repositorio.
2. Abre el proyecto con el editor de código de preferencia.
3. Instale los requerimientos `requirements.txt`

## Variables
* **sellers_address** Ubicación del seller. (country, state, city).
* **warranty**: Texto que puede tomar cualquier forma dependiendo de lo que ingrese el seller.
* **variations**: Un item con multiples variaciones puede indicar que el item es nuevo.
* **sub_status**: Puede tomar el valor de 'deleted', 'expired', 'suspended', 'none'.
* **listing_type_id**: Categoría del tipo de publicación. Puede tomar el valor de 'bronze', 'silver', 'free', 'gold_special', 'gold', 'gold_premium', 'gold_pro'
* **buying_mode**: Puede tomar el valor de 'buy_it_now', 'classified', 'auction'.
* **tags**: Puede tomar los valores de 'free_relist', 'good_quality_thumbnail', 'dragged_bids_and_visits', 'poor_quality_thumbnail', 'dragged_visits'.
* **parent_item_id**: Id del item padre. No es informativo dado que los parent_item_id no aparencen en la lista de id de items, por lo que no se pueden mapear. Seguramente por el tamaño de la muestra.
* **category_id**: Id de la categoría. 
* **last_updated**: Ultima actualización.
* **accepts_mercadopago**: 1 si acepta mercadopago, 0 si no. Solo pocos tienen 0 pero los que son cero la mayoria venden usado.
* **currency_id**: Tipo de moneda. Puede tomar el valor de 'ARG', 'USD'. Hay muy pocos con 'USD'.
* **title**: Titulo de la publicación. Puede contener palabras clave que determinen si es nuevo o usado. 
* **automatic_relist**: Replublicación automática. Son pocas las que tienen en true, pero cuando lo tiene identifica bien la clase de nuevo.
* **status**: Estado de la publicacion. La mayoria son activos y unos pocos pausados. 
* **sold_quantity**: Nuero de ventas. Si ha venidio al menos uno puede ser un indicador que el producto es nuevo.
* **available_quantity**: Numero de unidades disponibles. Si tiene mas de una unidad puede ser un buen indicador de que el producto es nuevo.
* **shipping_local_pick_up**: Tipo de recogida del producto local?
* **shipping_free_shipping**: Tiene o no envio gratuido. Si tiene envio gratuito puede ser un buen indicador de que el producto es nuevo. 
* **shipping_mode**: El modo de entrega puede ayudar a identificar si es nuevo o usado. Toma los valores de 'custom', 'me1', 'me2', 'not_specified'.
  
#### Variables no usadas
* **base_price**: Tiene exactamente los mismos valores que price (revisado)
* **deal_ids**: Contiene solo valores de deal ids sin relevencia. 
* **differential_pricing**: Todos los valores son nulos.
* **catalog_product_id**: Todos los valores son nulos.
* **subtitle**: Todos los valores son nulos.
* **original_price**: Todos los valores son nulos.
* **official_store_id**: Todos los valores son nulos.
* **video_id**: Todos los valores son nulos.
* **site_id**: Toma siempre el valor de MLA (Mercado Libre Argentina)
* **listing_source**: Todos los datos estan en blanco.
* **parent_item_id**: Id del padre, no me dice nada.
* **coverage_areas**: No contiene información. Solo arreglos vacios.
* **descriptions**: Lista de id. Se puede omitir. 
* **international_delivery_mode**: No tiene datos.
* **thumbnail**: No tiene imagenes. Solo una imagen default.
* **secure_thumbnail**: No tiene informacion relevante. Envia a una pagina forbidden.
* **permalink**: No tiene links relevantes. Se revisaron y estan expirados.
* **shipping_methods**: No tiene información. Todo vacio. 
* **shipping_dimensions**: No tiene informacion relevante. 

## Resultados

La siguiente tabla resume los resultados de los diferentes modelos de clasificación que fueron probados usadon la medida de desempeño Accuracy y la medida propuesta de ROC AUC.
> El mejor modelo fue el modelo Logit con un accruacy en test de `0.8878` y un ROC AUC en test de `0.9524`

|                | **Baseline (Logit simple)** | **Logit Mejorado** | **XGBoost** | **SVM** | **NNet** |
|----------------|-----------------------------|--------------------|-------------|---------|----------|
| Train Accuracy | 0.8799                      | 0.9022             | 0.8898      |         |          |
| Test Accuracy  | 0.8611                      | `0.8879`           | 0.8847      |         |          |
| Train ROC AUC  | 0.9478                      | 0.9616             | 0.9600      |         |          |
| Test ROC AUC   | 0.9338                      | 0.9524             | `0.9526`    |         |          |

## Puesta en producción

Para poner en producción el modelo de clasificacion de items nuevos y usados, se recomienda implementar una API haciendo uso del modelo `Logit Mejorado` el cual tiene un muy buen desempeño frente a los otros competidores y además ofrece una baja latencia. 

En el archivo `for_deployment.py` se ha dispuesto un esquema de API sencilla utilizando Flask. Este recibe como input la información en el mismo formato json con el que se entrenó el modelo (el mismo usado en la API de MELI) y entrega como resultado el siguiente formato JSON:

```json
{
    "prediction": 'Nuevo', 
    "probability": 0.872321
}
```

Donde `prediccion` correponde a la prediccion final estimada por el clasificador y `probability` corresponde a la probabilidad estiada del modelo.

## Por mejorar

1. Dado el recurso limitado del tiempo y la capacidad de computo, se siguiere mejorar la optimizacion de los hiperparametros de los modelos lo cual puede mejorar el desempeño.
2. Incluir una etapa de selección de features usando la importancia que tiene cada una de estas en la predicción final.
3. Realziar un analisis post-estimación como SHAP values para entender mejor como interactua cada variable en la probabilidad de predecir nuevo o usado. 

## Tecnologías
*   Python 3.11
*   Git / Github

## Recursos
* [Chi-square test in contingency tables](https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223) - Used to DEA and feature selection

## Licencia
> Este proyecto tiene licencia MIT

## Autor
> **Juan Camilo Díaz Herrera** (<https://github.com/juancadh>)


