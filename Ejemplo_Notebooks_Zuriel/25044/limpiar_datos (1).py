import pandas as pd
import re
import os
from numpy import nan

def pre_processing_stg(input_file, output_pathfile):
    """pre_processing_stg(txt_file, output_pathfile):

    This function receives a .txt file with a specific structure, 
    then it cleans and processes it and saves it as a .csv file.
        
    Input:
       - input_file (File): A .txt file with a specific structure.
       - output_pathfile (String): The directory where the .csv file will be saved.
        
    Return: 
        None
    """
    
    with open(archivo_txt, 'r', encoding='latin-1') as file:
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
        df[column] = df[column].astype(str).replace(nan, 'null')
        df[column] = df[column].astype(str).replace('nan', 'null')
    
    archivo_salida = os.path.join(directorio_salida, f'NR_{estacion}.csv')
    df.to_csv(archivo_salida, index=False, encoding='utf-8')
    print(f'Archivo CSV guardado como {archivo_salida}')

def batch_txt_a_csv_stg2(inputh_pathfile):
    """batch_txt_a_csv_stg2(input_patfile):
    
    This function receives a directory, detects the .txt files in it and processes 
    them using pre_processing_stg function.

    Input:
        inputh_pathfile (String): The directory where the .txt files are located.
    Return:
        None
    """


    directorio_salida = inputh_pathfile + "/limpio"
    os.makedirs(directorio_salida, exist_ok=True)
    for archivo in os.listdir(inputh_pathfile):
        if archivo.endswith(".txt"):
            archivo_txt = os.path.join(inputh_pathfile, archivo)
            pre_processing_stg(archivo_txt, directorio_salida)
    

def build_monthly_dataset_files(input_pathfile):
    """build_monthly_dataset_files(input_pathfile):
    
    This function receives the directory of the raw files,
    then locates the processed files folder and generates
    monthly summaries for each station.

    Input:
        input_pathfile (String): folder where the original files are located
    Return:
        None
    """
    directorio_limpios = input_pathfile + "/limpio"
    directorio_concentrado = input_pathfile + "/CM"
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
    """bundle combined results(input_pathfile):
    
    This function receives the directory of the raw files, 
    then locates the monthly summaries folder and sumarizes
    them into a single .csv file.

    Input:
        input_pathfile (String): folder where the raw files are located
    Return:
        None
    """


    directorio_concentrado = input_pathfile + "/CM"
    directorio_final = input_pathfile + "/final"
    os.makedirs(directorio_final, exist_ok=True)
    archivos_csv = [os.path.join(directorio_concentrado, f) for f in os.listdir(directorio_concentrado) if f.endswith(".csv")]
    df_final = pd.concat([pd.read_csv(f) for f in archivos_csv], ignore_index=True)
    for column in ['PRECIP', 'EVAP', 'TMAX', 'TMIN']:
        df_final[column] = df_final[column].astype(str).replace(nan, 'null')
        df_final[column] = df_final[column].astype(str).replace('nan', 'null')
    archivo_final = os.path.join(directorio_final, 'final.csv')
    df_final.to_csv(archivo_final, index=False, encoding='utf-8')
    print(f'Archivo final concatenado guardado como {directorio_final}.csv')