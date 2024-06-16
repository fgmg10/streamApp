# Importar librerías necesarias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pip
pip.main(["install","openpyxl"])

# Función para cargar datos desde un archivo
def cargar_datos(archivo):
    if archivo is not None:
        if archivo.name.endswith('.xlsx') or archivo.name.endswith('.xls'):
            df = pd.read_excel(archivo)
        elif archivo.name.endswith('.csv'):
            df = pd.read_csv(archivo)
        else:
            return None, 'Formato de archivo no compatible. Por favor selecciona un archivo Excel (.xlsx, .xls) o CSV (.csv).'
        return df, None
    return None, None

# Función principal para la interfaz de usuario
def main():
    st.title('Análisis de Datos con Streamlit')
    
    # Cargar archivo usando file_uploader
    archivo = st.file_uploader('Cargar archivo Excel o CSV', type=['xlsx', 'xls', 'csv'])
    
    # Manejar advertencias y errores
    advertencia = None
    if archivo is None:
        advertencia = 'Por favor selecciona un archivo para cargar.'
    
    df, error = cargar_datos(archivo)
    
    # Mostrar advertencias en un dropdown
    opciones_advertencias = [advertencia]
    if error:
        opciones_advertencias.append(error)
    
    advertencia_elegida = st.selectbox('Advertencias', opciones_advertencias)
    
    # Mostrar advertencia seleccionada
    if advertencia_elegida:
        st.warning(advertencia_elegida)
        return
    
    if df is not None:
        # Mostrar una vista previa de los datos
        st.subheader('Vista Previa de Datos')
        st.write(df.head())
        
        # Opciones para análisis y gráficos
        st.sidebar.subheader('Opciones de Análisis')
        opciones = st.sidebar.selectbox('Seleccionar opción de análisis', ['Histograma', 'Gráfico de Líneas', 'Diagrama de Barras'])
        
        # Generar gráfico según la opción seleccionada
        st.subheader('Visualización')
        
        if opciones == 'Histograma':
            columna = st.selectbox('Seleccionar columna para el histograma', df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[columna], bins=20, kde=True, ax=ax)
            st.pyplot(fig)
        
        elif opciones == 'Gráfico de Líneas':
            # Suponiendo que tenemos una columna de fechas y una de valores numéricos para hacer un gráfico de líneas
            fecha_columna = st.selectbox('Seleccionar columna de fechas', df.columns)
            valor_columna = st.selectbox('Seleccionar columna de valores numéricos', df.columns)
            fig, ax = plt.subplots()
            ax.plot(df[fecha_columna], df[valor_columna])
            ax.set_xlabel(fecha_columna)
            ax.set_ylabel(valor_columna)
            ax.set_title('Gráfico de Líneas')
            st.pyplot(fig)
        
        elif opciones == 'Diagrama de Barras':
            columna_categorica = st.selectbox('Seleccionar columna categórica', df.columns)
            fig, ax = plt.subplots()
            df[columna_categorica].value_counts().plot(kind='bar', ax=ax)
            st.pyplot(fig)
        
        st.sidebar.text('Desarrollado por: Frank')

if __name__ == '__main__':
    main()
