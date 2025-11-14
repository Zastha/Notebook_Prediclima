import pandas as pd
import re
import os
from numpy import nan

def txt_csv_data_process_stg2(directorio_entrada):
    """procesar_lote_txt_a_csv(directorio_entrada):
    
    This function receives a directory, detects the .txt files in it and processes them using the limpiar_txt_a_csv function.

    Input:
        directorio_entrada (String): The directory where the .txt files are located.
    Return:
        None
    """
    directorio_salida = os.path.join(directorio_entrada, "limpio")
    os.makedirs(directorio_salida, exist_ok=True)
    for archivo in os.listdir(directorio_entrada):
        if archivo.endswith(".txt"):
            archivo_txt = os.path.join(directorio_entrada, archivo)
            txt_csv_data_cleaning_stg1(archivo_txt, directorio_salida)
            rellenar_fechas_faltantes(os.path.join(directorio_salida, f'NR_{archivo.replace(".txt", "")}.csv'))

def txt_csv_data_cleaning_stg1(txt_file, output_pathfile):
    """txt_csv_data_cleaning_stg1(txt_file, output_pathfile):

    This function receives a .txt file with a specific structure, then it cleans and processes it and saves it as a .csv file.
        
    Input:
       - txt_file (File): A .txt file with a specific structure.
       - output_pathfile (String): The directory where the .csv file will be saved.
        
    Return: 
        None
    """
    with open(txt_file, 'r', encoding='latin-1') as file:
        lineas = file.readlines()
    
    etiquetas = []
    valores = []
    for i in [4, 11, 12, 13]:
        if ':' in lineas[i]:
            clave, valor = map(str.strip, re.split(r':\s*', lineas[i], maxsplit=1))
            valor = valor.replace('�', '')
            if clave in ['LATITUD', 'LONGITUD']:
                valor = valor.replace('°', '')
            if clave == 'ALTITUD':
                valor = re.sub(r'\s*msnm', '', valor)
            etiquetas.append(clave)
            valores.append(valor)
    
    estacion = valores[0] if valores else "Desconocido"
    datos = []
    
    for linea in lineas[19:]:
        if re.match(r'\d{2}/\d{2}/\d{4}', linea):
            valores_linea = re.split(r'\s+', linea.strip())
            if len(valores_linea) == 5:
                valores_linea = [v.replace('°', '') for v in valores_linea]
                datos.append(valores_linea)
    
    columnas = ['FECHA', 'PRECIP', 'EVAP', 'TMAX', 'TMIN'] + etiquetas
    df = pd.DataFrame(datos, columns=columnas[:len(datos[0])])
    
    for i, etiqueta in enumerate(etiquetas):
        df[etiqueta] = valores[i] if i < len(valores) else ""
    
    df.replace('Nulo', None, inplace=True)
    for column in ['PRECIP', 'EVAP', 'TMAX', 'TMIN']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%d/%m/%Y', errors='coerce')
    df = df.dropna(subset=['FECHA'])
    
    df['MONTH'] = df['FECHA'].dt.month
    df['YEAR'] = df['FECHA'].dt.year
    
    for column in ['PRECIP', 'EVAP', 'TMAX', 'TMIN']:
        df[column] = df.groupby(['YEAR', 'MONTH'])[column].transform(lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else nan))
    
    for column in ['PRECIP', 'EVAP', 'TMAX', 'TMIN']:
        df[column] = df[column].astype(str).replace(nan, 'null')
        df[column] = df[column].astype(str).replace('nan', 'null')
    
    archivo_salida = os.path.join(output_pathfile, f'NR_{estacion}.csv')
    df.to_csv(archivo_salida, index=False, encoding='utf-8')
    print(f'Archivo CSV guardado como {archivo_salida}')

def fill_missing_data(csv_file):
    
    """fill_missing_data(archivo_csv):

    This function receives the processed csv file and fills in the missing data by calculating the min-max date range.
        
    Input:
       - csv_file (File): csv file with missing dates.
        
    Return: 
        None
    """
    
    df = pd.read_csv(csv_file, dayfirst=True)
    df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
    
    # Contar número de registros en el Dataframe df
    total_registros_original = df.shape[0]
    
    fecha_min = df['FECHA'].min()
    fecha_max = df['FECHA'].max()
    total_fechas = pd.date_range(start=fecha_min, end=fecha_max, freq='D')
    
    all_dates = pd.DataFrame({'FECHA': total_fechas})
    
    df_completo = all_dates.merge(df, on='FECHA', how='left')
    total_registros_generados = df_completo.shape[0]
    
    fechas_faltantes = total_registros_generados - total_registros_original
    porcentaje_faltante = (fechas_faltantes / total_registros_generados) * 100
    
    df_completo = df_completo.astype(str).replace('nan', 'null')

    df_completo.to_csv(csv_file, index=False, encoding='utf-8')
    
    log_msg = f"Fechas faltantes rellenadas en {csv_file}\nTotal de fechas faltantes: {fechas_faltantes}\nPorcentaje de fechas incompletas: {porcentaje_faltante:.2f}%"
    print(log_msg)

def build_monthly_dataset_files_stg3(directorio_entrada):
    """generar_concentrados_mensuales(directorio_entrada):
    
    This function receives the directory of the raw files, then locates the processed files folder and generates monthly summaries for each station.

    Input:
        directorio_entrada (String): folder where the originla files are located
    Return:
        None
    """
    directorio_limpios = os.path.join(directorio_entrada, "limpio")
    directorio_concentrado = os.path.join(directorio_entrada, "CM")
    os.makedirs(directorio_concentrado, exist_ok=True)
    archivos_csv = [os.path.join(directorio_limpios, f) for f in os.listdir(directorio_limpios) if f.endswith(".csv")]
    
    for archivo in archivos_csv:
        df = pd.read_csv(archivo)
        
        df_mensual = df.groupby(['YEAR', 'MONTH']).agg({
            'PRECIP': 'sum',
            'EVAP': 'mean',
            'TMAX': 'max',
            'TMIN': 'min'
        }).reset_index()
        
        if not df_mensual.empty:
            constantes = ['ESTACION', 'LATITUD', 'LONGITUD', 'ALTITUD']
            for const in constantes:
                df_mensual[const] = df[const].iloc[0] if const in df.columns and not df[const].isna().all() else "Desconocida"
        
            for column in ['PRECIP', 'EVAP', 'TMAX', 'TMIN']:
                df_mensual[column] = df_mensual[column].astype(str).replace(nan, 'null')
                df_mensual[column] = df_mensual[column].astype(str).replace('nan', 'null')
        
            archivo_concentrado = os.path.join(directorio_concentrado, f'CM_{df_mensual["ESTACION"].iloc[0]}.csv')
            df_mensual.to_csv(archivo_concentrado, index=False, encoding='utf-8')
            print(f'Archivo de concentrado mensual guardado como {archivo_concentrado}')

def bundle_combined_results(input_pathfile):
    """bundle_combined_results(input_pathfile):
    
    This function receives the directory of the raw files, then locates the monthly summaries folder and concatenates them into a single .csv file.

    Input:
        input_pathfile (String): folder where the raw files are located
    Return:
        None
    """
    directorio_concentrado = os.path.join(input_pathfile, "CM")
    directorio_final = os.path.join(input_pathfile, "FINAL")
    os.makedirs(directorio_final, exist_ok=True)
    archivos_csv = [os.path.join(directorio_concentrado, f) for f in os.listdir(directorio_concentrado) if f.endswith(".csv")]
    df_final = pd.concat([pd.read_csv(f) for f in archivos_csv], ignore_index=True)
    for column in ['PRECIP', 'EVAP', 'TMAX', 'TMIN']:
        df_final[column] = df_final[column].astype(str).replace(nan, 'null')
        df_final[column] = df_final[column].astype(str).replace('nan', 'null')
    archivo_final = os.path.join(directorio_final, 'FINAL.csv')
    df_final.to_csv(archivo_final, index=False, encoding='utf-8')
    print(f'Archivo final concatenado guardado como {archivo_final}')
