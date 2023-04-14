import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# import openpyxl

# RUN WITH: streamlit run CG_app.py
# Se realiza la selección del grupo de control para Clientes Perfectos.

# st.markdown(
#     """"""
#     <style>
#     .main {
#     background-color: #F5F5F5;
#     }
#     <syle>
#     """""",
#     unsafe_allow_html=True
# )


@st.cache_data
def get_data(filename):
    data = pd.read_excel(filename)
    return data


st.image("controlgroup.png", caption="", width=700)
st.sidebar.title("Bienvenido!")

# Seleccionar archivo de grupos de control
uploaded_file = st.sidebar.file_uploader("Choose a file")
df = get_data(uploaded_file)

################################# Main page #######################################
st.title("Selección de grupo de control para Cliente Perfecto")

st.header("Formato requerido del dataset")
# Mostrar formato y nombre esperado de las columnas
# Año	Mes	    CodClien final	DescriClien final	CP	    Canal	Tipo Cliente	Categoria	Familia	 VN	   Ton
# int	string	int	            string	            string	string	string	        string	    string	 float float
st.write('Para realizar los cálculos correctamente, este es el nombre que debe tener cada columna:')
dict_format = {'Año': {'Nombre': 'Año', 'Tipo': 'int'},
               'Mes': {'Nombre': 'Mes', 'Tipo': 'string'},
               'Código del cliente': {'Nombre': 'CodClien final', 'Tipo': 'int'},
               #'Descripción del cliente': {'Nombre': 'DescriClien final', 'Tipo': 'string'},
               'Cliente Perfecto': {'Nombre': 'CP', 'Tipo': 'string'},
               'Canal': {'Nombre': 'Canal', 'Tipo': 'string'},
               'Tipo de Cliente': {'Nombre': 'Tipo Cliente', 'Tipo': 'string'},
               # 'Categoría': {'Nombre': 'Categoria', 'Tipo': 'string'},
               # 'Familia': {'Nombre': 'Familia', 'Tipo': 'string'},
               'Venta': {'Nombre': 'VN', 'Tipo': 'float'},
               'Toneladas': {'Nombre': 'Ton', 'Tipo': 'float'},
               }
dict_format = pd.DataFrame.from_dict(dict_format)
st.write(dict_format)

st.header("Resumen dataset")

st.write('Vista previa')
st.write(df.head())

st.write('Periodos presentes en el archivo')
df_tabla = df.pivot_table(
    index=['Año'],
    aggfunc={'Mes': ['unique', 'nunique']}
)
df_tabla.columns = ["Num_meses", "Mes"]
st.write(df_tabla)

st.write("Número de clientes:", df["CodClien final"].nunique())
st.write("Número de clientes perfectos:", df.loc[df["CP"] == "CP", "CodClien final"].nunique())
st.write("Número de clientes no perfectos:", df.loc[df["CP"] == "NO CP", "CodClien final"].nunique())


# Se crea el campo fecha, añadir loop para validar si es numérico o string
condlist = [df['Mes'] == 'Ene',
            df['Mes'] == 'Feb',
            df['Mes'] == 'Mar',
            df['Mes'] == 'Abr',
            df['Mes'] == 'May',
            df['Mes'] == 'Jun',
            df['Mes'] == 'Jul',
            df['Mes'] == 'Ago',
            df['Mes'] == 'Set',
            df['Mes'] == 'Oct',
            df['Mes'] == 'Nov',
            df['Mes'] == 'Dic']
choicelist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
df['Mes2'] = np.select(condlist, choicelist, default='unknown')

df['Fecha'] = np.where(df['Mes2'].astype(str).str.len() == 1,
                       df['Año'].astype(str) + '0' + df['Mes2'].astype(str) + '01',
                       df['Año'].astype(str) + df['Mes2'].astype(str) + '01')
df['Fecha'] = pd.to_datetime(df['Fecha'])

# Ver si se repite algun cliente en CP y No CP
df_check = df.pivot_table(
    index=['CodClien final'],
    aggfunc={'CP': 'nunique'}
)
if df_check['CP'].nunique() != 1:
    st.write("Hay clientes que presentan ambos status CP y NO CP!")

df2 = df.copy()


st.header("Filtros aplicados")

# Montos mayores a cero
df = df[df["VN"] > 0]
st.write('Filtro 1: Montos mayores a cero')
st.write(df['CodClien final'].nunique())
st.write(df.groupby('CP')['CodClien final'].nunique())

# Selección del periodo a considerar
st.write('Filtro 2: Periodo')
# Slider para selección de 1 año o ambos
time_frame = st.selectbox('Desea seleccionar un año o utilizar toda la ventana de tiempo?',
                           ['1 año', 'Todos los periodos'], index=0)
if time_frame == '1 año':
    option_year = st.selectbox('Selecciona el año a considerar: ', df['Año'].sort_values().unique(), index=0)
    df = df[df["Año"] == option_year]
st.write(df['CodClien final'].nunique())
st.write(df.groupby('CP')['CodClien final'].nunique())

# Selección de mínimo número de meses
st.write('Filtro 3: Selección de mínimo número de meses')
option_min_months = st.slider('Selecciona el número de meses: ', min_value=1, max_value=12, value=8)
df_mes = df.pivot_table(
    index=['CodClien final', 'CP', 'Fecha'],
    aggfunc={'VN':'sum', 'Familia':'nunique',}
).reset_index()
df_mes.rename(columns={'VN': 'Monto', 'Familia': 'Mix_familias'}, inplace=True)
df_conteo_meses = df_mes.pivot_table(
    index=['CodClien final', 'CP'],
    aggfunc={"Fecha":['nunique']}
)
df_conteo_meses.columns = ["cant_meses"]
df_conteo_meses.reset_index(inplace=True)
df_mes = df_mes[df_mes['CodClien final'].isin(df_conteo_meses
                                              [((df_conteo_meses['cant_meses'] >= option_min_months)
                                                & (df_conteo_meses['CP'] == 'NO CP'))
                                               | (df_conteo_meses['CP'] == 'CP')]
                                              ['CodClien final'].unique())].copy()
st.write(df_mes['CodClien final'].nunique())
st.write(df_mes.groupby('CP')['CodClien final'].nunique())

# Selección de mínimo número de categorías
st.write('Filtro 4: Selección de mínimo número de categorías')
option_min_cat = st.slider('Selecciona el número de categorías: ', min_value=1, max_value=10, value=6)
df_cliente = df_mes.pivot_table(
    index=['CodClien final', 'CP'],
    aggfunc={'Monto': ['mean', 'std'],
             'Mix_familias': 'max'}
)
df_cliente.columns = ["mix_familias_max", "ticket_prom_mes", "ticket_std"]
df_cliente.reset_index(inplace=True)
df_cliente = df_cliente[((df_cliente['mix_familias_max'] >= option_min_cat) & (df_cliente['CP'] == 'NO CP'))
                        | (df_cliente['CP'] == 'CP')]
st.write(df_cliente['CodClien final'].nunique())
st.write(df_cliente.groupby('CP')['CodClien final'].nunique())

# Se excluyen tickets promedios atípicos
st.write('Filtro 5: Exclusión de tickets promedios atípicos')
# option_min_per = st.selectbox('Selecciona el percentil inferior: ', [0.01,0.05,0.1])
# option_max_per = st.selectbox('Selecciona el percentil superior: ', [0.99,0.95,0.9])
P1 = df_cliente["ticket_prom_mes"].quantile(0.01)
P99 = df_cliente["ticket_prom_mes"].quantile(0.99)
df_cliente = df_cliente[(df_cliente['CP'] == 'CP') |
                        ((df_cliente["ticket_prom_mes"] >= P1) &
                         (df_cliente["ticket_prom_mes"] <= P99) &
                         (df_cliente['CP'] == 'NO CP'))]
st.write(df_cliente['CodClien final'].nunique())
st.write(df_cliente.groupby('CP')['CodClien final'].nunique())

# st.subheader('Distribución de ticket promedio por grupo')
fig, ax = plt.subplots()
sns.boxplot(x="CP", y="ticket_prom_mes", data=df_cliente, hue='CP')
plt.title('Distribución de ticket promedio por grupo')
st.pyplot(fig)

# Selección de deciles
st.write('Filtro 6: Exclusión de deciles inferiores')
# option_min_dec = st.selectbox('Selecciona el mínimo decil (mientras más alto mayor TP y más cercano a CP):',
# [3,4,5,6,7,8,9,10])
df_cliente["deciles"] = pd.qcut(df_cliente["ticket_prom_mes"], q=10, labels=np.arange(1, 11)).astype('int')
df_cliente = df_cliente[(df_cliente['CP']=='CP') | ((df_cliente['CP']=='NO CP') & (df_cliente['deciles'] >= 8))]
st.write('Esta es la cantidad de clientes por grupo que pasan los filtros (NO CP debe ser mayor a CP):')
st.write(df_cliente.groupby('CP')['CodClien final'].nunique())


st.header("Grupo de control seleccionado")


# Funcion que selecciona el grupo de control
def asignacion_grupos(df, seed, n_sample, use_dict, dict_canal):
    # Se consideran todos clientes CP
    grupo1 = df.loc[df['CP'] == 'CP', ['CodClien final']]
    grupo1["groups"] = 'CP'

    # Se valida si utilizar diccionario para asignar num de clientes por canal
    if use_dict:
        grupo2 = pd.DataFrame({})
        for key in dict_canal:
            grupoc = (
                df.loc[(df['CP'] == 'NO CP') & (df['Canal'] == key), ['CodClien final']]
                .apply(lambda s: s.sample(n=dict_canal[key], random_state=seed))
                .reset_index()
            ).drop("index", axis=1)
            grupoc["groups"] = 'NO_CP'
            grupo2 = pd.concat([grupo2, grupoc])
    else:
        grupo2 = (
            df.loc[df['CP'] == 'NO CP', ['CodClien final']]
            .apply(lambda s: s.sample(n=n_sample, random_state=seed))
            .reset_index()
        ).drop("index", axis=1)
        grupo2["groups"] = 'NO_CP'

    grupo_concatenado = pd.concat([grupo1, grupo2])

    return grupo_concatenado


def create_groups(df, df_mes, seed, n_sample, show=False, use_dict=False, dict_canal=None):
    total_group = asignacion_grupos(df=df, seed=seed, n_sample=n_sample, use_dict=use_dict, dict_canal=dict_canal)

    df_piloto = df_mes.merge(
        # right = total_group[['deciles', 'CodClien final', 'groups']],
        right=total_group[['CodClien final', 'groups']],
        on=["CodClien final"],
        how="inner"
    )

    graph = df_piloto.pivot_table(
        index=["Fecha", "groups"],
        aggfunc={"Monto": "mean"}
    ).reset_index()
    graph['Fecha'] = graph['Fecha'].astype("str")

    # Comparando standard dev.
    print_std = graph.pivot_table(
        index="Fecha",
        aggfunc={"Monto": "std"}
    )

    if show:
        sns.lineplot(x="Fecha", y="Monto", data=graph, hue="groups")
        plt.xticks(rotation=60)
        plt.show()

    return df_piloto, print_std.Monto.mean()


if df_cliente.loc[df_cliente['CP'] == 'NO CP', 'CodClien final'].nunique() \
        < df_cliente.loc[df_cliente['CP'] == 'CP', 'CodClien final'].nunique():

    st.write("No hay suficientes clientes NO CP para generar el grupo de control")

else:

    # Se asigna el numero de clientes deseado (igual al total de CP)
    n_sample = df2.loc[df2['CP'] == 'CP', 'CodClien final'].nunique()

    lista_std = dict()
    seeds = []
    lst = []
    for seed_id in tqdm(range(1001)):
        _, mean_std = create_groups(df_cliente, df_mes, seed_id, n_sample)
        seeds.append(seed_id)
        lst.append(mean_std)
    lista_std["seed"] = seeds
    lista_std["mean_std"] = lst

    temp_std = pd.DataFrame(lista_std)
    con_nueva_meto = temp_std.sort_values("mean_std").reset_index(drop=True)
    st.write('Este es la desviacion del top mejores grupos encontrados:', con_nueva_meto.head())

    BEST_SEED = con_nueva_meto.seed[0]
    df_piloto_elegido, _ = create_groups(df_cliente, df_mes, BEST_SEED, n_sample=n_sample)

    # Se prepara el dataset final
    df_piloto_elegido['groups'] = df_piloto_elegido.groups.astype('category')

    fig, ax = plt.subplots()
    sns.lineplot(x="Fecha", y="Monto", data = df_piloto_elegido, hue="groups", palette=sns.cubehelix_palette(2))
    plt.title("Mean std: " + str(round(con_nueva_meto.mean_std[0], 2)))
    st.pyplot(fig)

    # Mostrar flujo de filtros?

    # Se arma la base final con grupo control seleccionado
    df_final = df_piloto_elegido.pivot_table(
        index=["CodClien final", "groups"],
        aggfunc={"Monto": "mean"}
    )
    df_final.columns = ["ticket_prom_mes"]
    df_final.reset_index(inplace=True)

    # Se revisa distribución de ticket por canal
    df3 = df2.sort_values(by=['CodClien final', 'Fecha']).pivot_table(
        index = ['CodClien final'],
        aggfunc = {"Canal": "last",
                   "Tipo Cliente": "last",
                   "Ton": "sum",
                   "VN": ["sum", "mean"]}
    ).reset_index()
    df3.columns = ["CodClien final", "Canal", "Tipo Cliente", "Ton_Sum", "Venta_Prom", "Venta_Sum"]

    df_final_tabla = df_final.merge(df3, how='inner', on='CodClien final')

    st.subheader('Número de clientes y ticket por canal:')
    tabla_canal = df_final_tabla[df_final_tabla['groups']=='NO_CP'].pivot_table(
        index=["Canal"],
        aggfunc={"ticket_prom_mes":["min", "max", "mean"], "CodClien final": "nunique"}
    ).reset_index()
    tabla_canal.columns = ["Canal", "n_clientes", "ticket_prom_max", "ticket_prom_prom", "ticket_prom_min"]
    st.write(tabla_canal)

    st.subheader('Número de clientes y ticket por tipo de cliente:')
    tabla_tipo_cliente = df_final_tabla[df_final_tabla['groups'] == 'NO_CP'].pivot_table(
        index=["Tipo Cliente"],
        aggfunc={"ticket_prom_mes":["min", "max", "mean"], "CodClien final": "nunique"}
    ).reset_index()
    tabla_tipo_cliente.columns = ["Tipo Cliente", "n_clientes", "ticket_prom_max", "ticket_prom_prom", "ticket_prom_min"]
    st.write(tabla_tipo_cliente)

    # Preguntar si se exporta el archivo con grupo seleccionado
    export_choice = st.selectbox('Desea exportar este grupo de control? ', ['Sí', 'No'], index=1)
    if export_choice == 'Sí':
        df_final[df_final['groups'] == 'NO_CP'].to_excel("groups_output_bolivia.xlsx", index=False)
        st.write('Se exportó la base correctamente')
