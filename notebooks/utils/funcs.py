import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, make_scorer, roc_auc_score, roc_curve, auc
import warnings

def contingency_table_analysis(df, categorical_column_name, target_column_name='target', plot_it=True, figsize=(10, 6)):
    """Tabla de contingencia entre variable categorica y variable objetivo binario.
    Si el valor p es menor que 0.05, podemos rechazar la hipótesis nula y concluir que hay una relación significativa."""
    
    df = df.copy()
    contingency_table = pd.crosstab(df[categorical_column_name], df[target_column_name])
    chi2, p, dof, ex = chi2_contingency(contingency_table)

    print(f'Chi2: {chi2:.2f}')
    print(f'p-value: {p:.4f}')

    # Si el valor p es menor que 0.05, podemos rechazar la hipótesis nula y concluir que hay una relación significativa.
    if p < 0.05:
        print(f"La variable categórica `{categorical_column_name}` separa bien las clases del target.")
    else:
        print(f"La variable categórica `{categorical_column_name}` no separa bien las clases del target.")

    if plot_it:
        plt.figure(figsize=figsize)
        sns.violinplot(data=df, x=categorical_column_name, y=target_column_name)
        plt.title(f'Distribución de la variable categórica `{categorical_column_name}` por target binario')
        plt.xticks(rotation=45, ha='right')
        plt.show()

    return contingency_table

def transform_x(df):
    """
    Transforms the input DataFrame by creating new columns based on existing columns.
    
    Parameters:
    - df (pandas.DataFrame): The input DataFrame to be transformed.
    
    Returns:
    - pandas.DataFrame: The transformed DataFrame.
    """
        
    df = df.copy()

    # Crear variables de ubicación (seller_address)
    df["seller_country"] = df.apply(lambda x: x["seller_address"]["country"]["name"], axis=1)
    df["seller_state"] = df.apply(lambda x: x["seller_address"]["state"]["name"], axis=1)
    df["seller_city"] = df.apply(lambda x: x["seller_address"]["city"]["name"], axis=1)

    # Crear variables de shipping
    df["shipping_local_pick_up"] = df["shipping"].apply(lambda x: x["local_pick_up"] if "local_pick_up" in x else False)
    df["shipping_methods"] = df["shipping"].apply(lambda x: x["methods"] if "methods" in x else [])
    df["shipping_tags"] = df["shipping"].apply(lambda x: x["tags"] if "tags" in x else [])
    df["shipping_free_shipping"] = df["shipping"].apply(lambda x: x["free_shipping"] if "free_shipping" in x else False)
    df["shipping_mode"] = df["shipping"].apply(lambda x: x["mode"] if "mode" in x else 'not_specified')
    df["shipping_dimensions"] = df["shipping"].apply(lambda x: x["dimensions"] if "dimensions" in x else None)

    # Crear una columna para cada variable metodo de pago que no sea mercadopago
    df["non_mercado_pago_payment_methods_2"] = df['non_mercado_pago_payment_methods'].apply(lambda methods: [method['description'] for method in methods])
    unique_non_mercadopago_methods = list(set(np.hstack(df["non_mercado_pago_payment_methods_2"])))
    variables_nmp_payment = []

    if unique_non_mercadopago_methods:
        for method in unique_non_mercadopago_methods:
            var_nmp_payment = f'nmp_payment_{method.lower().replace(" ", "_")}'
            variables_nmp_payment.append(var_nmp_payment)
            df[var_nmp_payment] = df["non_mercado_pago_payment_methods_2"].apply(lambda methods: 1 if method in methods else 0)


    # Crear una columna para cada variable de tags
    unique_tags = list(set(np.hstack(df['tags'])))
    variables_tags= []
    for tag in unique_tags:
        var_tag = f'tag_{tag.lower().replace(" ", "_")}'
        variables_tags.append(var_tag)
        df[var_tag] = df["tags"].apply(lambda tags: 1 if str(tag) in tags else 0)

    # Convertir variable accepts_mercadopago booleana en binaria 
    df["accepts_mercadopago"] = df["accepts_mercadopago"].apply(lambda x: int(x))
    df["automatic_relist"] = df["automatic_relist"].apply(lambda x: int(x))

    # Convertir variable warranty en string cuando es None
    df["warranty"] = df["warranty"].apply(lambda x: 'Nada' if x == None else x)
    df["warranty"] = df["warranty"].apply(lambda x: 'Nada' if x == None else x)

    # Creación de variables relevantes

    # Variaciones
    df["has_variations"] = df.apply(lambda x: 1 if len(x["variations"]) > 0 and len(x["variations"][0]['attribute_combinations']) > 0 else 0, axis=1)
    df["number_variations"] = df.apply(lambda x: len(x["variations"][0]['attribute_combinations']) if len(x["variations"]) > 0 and len(x["variations"][0]['attribute_combinations']) > 0 else 0, axis=1)

    # Atributos
    df["has_attributes"] = df.apply(lambda x: 1 if len(x["attributes"]) > 0 else 0, axis=1)
    df["number_attributes"] = df.apply(lambda x: len(x["attributes"]) if len(x["attributes"]) > 0 else 0, axis=1)

    # Sub_status (suspended)
    df["sub_status"] = df["sub_status"].apply(lambda x: x[0] if len(x) > 0 else 'none')

    # tiene o no Garantia
    df['has_warranty'] = df['warranty'].apply(lambda x: 0 if x == None or x.lower == 'sin garantía' or x.lower == 'sin garantia' or x.lower == 'no tiene' else 1)


    # Sold Quantity
    def sold_quantity_category(x):
        """Toma el valor de 0 si no tiene ventas y el valor de 1 si tiene al menos una venta"""
        return 0 if x == 0 else 1

    df["sold_quantity_category"] = df["sold_quantity"].apply(sold_quantity_category) 

    # Available quantity
    def available_quantity_category(x):
        """Toma el valor de 0 si tiene solo una unidad disponible y el valor de 0 si tiene al mas de una"""
        return 0 if x == 1 else 1

    df["available_quantity_category"] = df["available_quantity"].apply(available_quantity_category)

    # Tiempo en segundos de diferencia entre stop_time y start_time
    df["time_diff"] = (pd.to_datetime(df["stop_time"]) - pd.to_datetime(df["start_time"])).dt.total_seconds()

    # Tiempo en dias que se ha mentenido activa la publicacion desde la creacion hasta la ultima actualizacion
    df["days_active"] = (pd.to_datetime(df["last_updated"]) - pd.to_datetime(df["date_created"])).dt.days

    def days_active_category(x):
        if x == -1:
            return "-1"
        elif x == 0:
            return "0. 0"
        elif x > 0 and x <= 7:
            return "1. 1-7"
        elif x > 7 and x <= 15:
            return "2. 8-15"
        elif x > 15 and x <= 30:
            return "3. 16-30"
        else:
            return "4. 31-more"
        
    df["days_active_category"] = df["days_active"].apply(days_active_category)

    # Eliminar columnas que no se van a usar
    drop_vars_list = ["deal_ids", "base_price", "differential_pricing", "catalog_product_id", "subtitle", "original_price", "official_store_id", 
                "video_id", "site_id", "listing_source", "parent_item_id", "coverage_areas", "descriptions", "international_delivery_mode",
                "thumbnail", "secure_thumbnail", "permalink", "shipping_methods", "shipping_dimensions", "variations", "attributes",
                "seller_address", "shipping", "non_mercado_pago_payment_methods", "non_mercado_pago_payment_methods_2", "tags"]

    df.drop(columns=drop_vars_list, inplace=True)

    return df

def show_evaluation(X_train, y_train, X_test, y_test, pipeline, show_test = False):
    """Presenta la evaluación del desempeño del train, validación cruzada y test de la clasificacion de nuevo o usado para Accuracy y ROC AUC"""

    # Validación cruzada
    accuracy_scoring = make_scorer(accuracy_score)

    accuracy_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=accuracy_scoring)
    roc_auc_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')

    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    # # Predecir en el conjunto de entrenamiento y prueba
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]

    # Evaluar el modelo en train y test
    accuracy_train = accuracy_score(y_train, y_train_pred)
    accuracy_test = accuracy_score(y_test, y_test_pred)

    roc_auc_train = roc_auc_score(y_train, y_prob_train)
    roc_auc_test = roc_auc_score(y_test, y_prob_test)

    # ------- Curva ROC de Train ------
    # Calcular la curva ROC
    fpr, tpr, _ = roc_curve(y_train, y_prob_train)
    roc_auc = auc(fpr, tpr)

    # Plotear la curva
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Train')
    plt.legend(loc="lower right")
    plt.show()

    # ------- Curva ROC de Test ------
    if show_test:
        # Calcular la curva ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        roc_auc = auc(fpr, tpr)

        # Plotear la curva
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Test')
        plt.legend(loc="lower right")
        plt.show()

    print("Evaluation using accuracy:")
    print(f'Accuracy train: {accuracy_train:.4f}')
    print(f'Accuracy validation: {accuracy_scores.mean():.4f}')
    if show_test:
        print(f'Accuracy test: {accuracy_test:.4f}')
    print("")    
    print("Evaluation using ROC AUC:")
    print(f'ROC AUC train: {roc_auc_train:.4f}')
    print(f'ROC AUC validation: {roc_auc_scores.mean():.4f}')
    if show_test:
        print(f'ROC AUC test: {roc_auc_test:.4f}')
        print("")
        print('Reporte de clasificacion del conjunto test:')
        print(classification_report(y_test, y_test_pred))

    return pipeline