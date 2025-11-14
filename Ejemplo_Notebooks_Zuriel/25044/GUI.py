import limpiar_datos as ld
import tkinter as tk
from tkinter import filedialog
from tkinter import *



if __name__ == "__main__":
    # Crear la ventana
    ventana = tk.Tk()
    ventana.title("Limpiar Datos")
    ventana.geometry("300x250")

    # Crear la etiqueta
    etiqueta = tk.Label(ventana, text="Selecciona la carpeta con los datos a limpiar")
    etiqueta.pack()

    # Crear la entrada mediante un botón que deje seleccionar una carpeta
    def seleccionar_carpeta():
        carpeta = tk.filedialog.askdirectory()
        if not carpeta:
            return
        else:
            print(carpeta)
            etiqueta_txt_csv = tk.Label(ventana, text="Limpiando archivos .txt a .csv")
            etiqueta_txt_csv.pack()
            ld.procesar_lote_txt_a_csv(carpeta)
            etiqueta_txt_csv.destroy()
            etiqueta_concentrados = tk.Label(ventana, text="Generando concentrados mensuales")
            etiqueta_concentrados.pack()
            
            ld.generar_concentrados_mensuales(carpeta)
            
            ld.concatenar_concentrados(carpeta)
            etiqueta_concentrados.destroy()
            etiqueta_concatenados = tk.Label(ventana, text="Concatenando concentrados")
            etiqueta_concatenados.pack()
            etiqueta_concatenados.after( 1000, lambda: etiqueta_concatenados.destroy())
            etiqueta_final = tk.Label(ventana, text="Datos limpiados")
            etiqueta.after(1000, lambda: etiqueta_final.pack())
            etiqueta_final.after( 2000, lambda: etiqueta_final.destroy())
            print("Datos limpiados")
            
    
    boton = tk.Button(ventana, text="Seleccionar carpeta", command=seleccionar_carpeta)
    boton.pack()
    
 
    
    

    # Mostrar la ventana
    ventana.mainloop()