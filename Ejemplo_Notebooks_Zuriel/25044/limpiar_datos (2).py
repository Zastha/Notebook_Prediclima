import pandas as pd
import re
import os
from numpy import nan

def limpiar_txt_a_csv(archivo_txt, directorio_salida):
    """limpiar_txt_a_csv(archivo_txt, directorio_salida):

    This function receives a .txt file with a specific structure, then it cleans and processes it and saves it as a .csv file.
        
    Input:
       - archivo_txt (File): A .txt file with a specific structure.
       - directorio_salida (String): The directory where the .csv file will be saved.
        
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
        df[column] = df.groupby(['YEAR', 'MONTH'])[column].transform(lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else nan))
    
    for column in ['PRECIP', 'EVAP', 'TMAX', 'TMIN']:
        df[column] = df[column].astype(str).replace(nan, 'null')
        df[column] = df[column].astype(str).replace('nan', 'null')
    
    archivo_salida = os.path.join(directorio_salida, f'NR_{estacion}.csv')
    df.to_csv(archivo_salida, index=False, encoding='utf-8')
    print(f'Archivo CSV guardado como {archivo_salida}')

def procesar_lote_txt_a_csv(directorio_entrada):
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
            limpiar_txt_a_csv(archivo_txt, directorio_salida)

def generar_concentrados_mensuales(directorio_entrada):
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

def concatenar_concentrados(directorio_entrada):
    """concatenar_concentrados(directorio_entrada):
    
    This function receives the directory of the raw files, then locates the monthly summaries folder and concatenates them into a single .csv file.

    Input:
        directorio_entrada (String): folder where the raw files are located
    Return:
        None
    """
    directorio_concentrado = os.path.join(directorio_entrada, "CM")
    directorio_final = os.path.join(directorio_entrada, "FINAL")
    os.makedirs(directorio_final, exist_ok=True)
    archivos_csv = [os.path.join(directorio_concentrado, f) for f in os.listdir(directorio_concentrado) if f.endswith(".csv")]
    df_final = pd.concat([pd.read_csv(f) for f in archivos_csv], ignore_index=True)
    for column in ['PRECIP', 'EVAP', 'TMAX', 'TMIN']:
        df_final[column] = df_final[column].astype(str).replace(nan, 'null')
        df_final[column] = df_final[column].astype(str).replace('nan', 'null')
    archivo_final = os.path.join(directorio_final, 'FINAL.csv')
    df_final.to_csv(archivo_final, index=False, encoding='utf-8')
    print(f'Archivo final concatenado guardado como {archivo_final}')
