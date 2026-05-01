# Proyecto_1_MCF_1
# García Rodríguez Marco Antonio
# Hernandez Alcantara Cristina Geraldine
# Mendoza Aragón Rogelio
# Moreno Ventura Miguel Angel
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, norm

st.title("Visualización de Rendimientos de Acciones")
st.header("Proyecto MCF")

# INCISO (A)

@st.cache_data
def obtener_datos(stocks):
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Activo "BZ=F" (petroleo)
stocks_lista = ['BZ=F']

with st.spinner("Descargando datos..."):
    df_precios = obtener_datos(stocks_lista)
    df_rendimientos = calcular_rendimientos(df_precios)

# Selector
stock_seleccionado = st.selectbox("Selecciona una acción", stocks_lista)

if stock_seleccionado:

  
    # INCISO (b)
    
    st.header("Inciso (b): Análisis estadístico")

    rendimiento_medio = df_rendimientos[stock_seleccionado].mean()
    skewness = skew(df_rendimientos[stock_seleccionado])
    exceso_kurtosis = kurtosis(df_rendimientos[stock_seleccionado], fisher=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Rendimiento Medio Diario", f"{rendimiento_medio:.4%}")
    col2.metric("Sesgo (Skewness)", f"{skewness:.4f}")
    col3.metric("Exceso de Curtosis", f"{exceso_kurtosis:.4f}")

    # Reporte
    st.subheader("Reporte estadístico de rendimientos")

    st.write(f"""
    Para el activo seleccionado (**{stock_seleccionado}**) se obtuvieron los siguientes resultados:

    - **Media de los rendimientos diarios:** {rendimiento_medio:.4%}  
    - **Sesgo (Skewness):** {skewness:.4f}  
    - **Exceso de curtosis:** {exceso_kurtosis:.4f}  

    La media representa el rendimiento promedio diario del activo.  
    El sesgo indica la asimetría de la distribución, mientras que la curtosis mide la presencia de colas pesadas en comparación con una distribución normal.
    """)

    # Interpretación
    if skewness > 0:
        st.write("El sesgo positivo indica mayor probabilidad de rendimientos extremos positivos.")
    elif skewness < 0:
        st.write("El sesgo negativo indica mayor probabilidad de pérdidas extremas.")
    else:
        st.write("La distribución es aproximadamente simétrica.")

    if exceso_kurtosis > 0:
        st.write("La curtosis positiva indica colas pesadas (mayor riesgo extremo).")
    elif exceso_kurtosis < 0:
        st.write("La curtosis negativa indica colas ligeras.")
    else:
        st.write("La distribución es similar a la normal.")

    # Gráfico
    st.subheader(f"Gráfico de Rendimientos: {stock_seleccionado}")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(df_rendimientos.index, df_rendimientos[stock_seleccionado])
    ax.axhline(y=0, linestyle='--')
    st.pyplot(fig)

    # Histograma
    st.subheader("Distribución de Rendimientos")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df_rendimientos[stock_seleccionado], bins=30)
    ax.axvline(rendimiento_medio, linestyle='dashed')
    st.pyplot(fig)

    # Shapiro
    st.subheader("Test de Normalidad (Shapiro-Wilk)")
    stat, p = shapiro(df_rendimientos[stock_seleccionado])

    st.write(f"**Shapiro-Wilk Test Statistic:** {stat:.4f}")
    st.write(f"**P-value:** {p:.4f}")

    if p > 0.05:
        st.success("La distribución parece ser normal (No se rechaza H0)")
    else:
        st.error("La distribución NO es normal (Se rechaza H0)")

    # QQ plot
    st.subheader("Q-Q Plot")
    fig, ax = plt.subplots()
    stats.probplot(df_rendimientos[stock_seleccionado], dist="norm", plot=ax)
    st.pyplot(fig)

    # INCISO (c)
    
    st.header("Inciso (c): VaR y ES")

    returns = df_rendimientos[stock_seleccionado].dropna()

    alphas = [0.95, 0.975, 0.99]

    mean = np.mean(returns)
    std = np.std(returns)

    df_t, loc_t, scale_t = stats.t.fit(returns)

    n_sims = 100000
    sim_norm = np.random.normal(mean, std, n_sims)

    resultados = []

    for alpha in alphas:

        VaR_norm = norm.ppf(1 - alpha, mean, std)
        ES_norm = mean - std * norm.pdf(norm.ppf(1 - alpha)) / (1 - alpha)

        VaR_t = stats.t.ppf(1 - alpha, df_t, loc_t, scale_t)
        ES_t = returns[returns <= VaR_t].mean()

        VaR_hist = returns.quantile(1 - alpha)
        ES_hist = returns[returns <= VaR_hist].mean()

        VaR_mc = np.percentile(sim_norm, (1 - alpha) * 100)
        ES_mc = sim_norm[sim_norm <= VaR_mc].mean()

        resultados.append({
            "Alpha": alpha,
            "VaR Normal": VaR_norm,
            "ES Normal": ES_norm,
            "VaR t-student": VaR_t,
            "ES t-student": ES_t,
            "VaR Histórico": VaR_hist,
            "ES Histórico": ES_hist,
            "VaR MonteCarlo": VaR_mc,
            "ES MonteCarlo": ES_mc
        })

    df_resultados = pd.DataFrame(resultados).map(lambda x: f"{x:.4%}")
    df_resultados.iloc[:, 1:] = df_resultados.iloc[:, 1:]

    st.dataframe(df_resultados)


# INCISO (D)

window = 252

# Rolling mean y std (pero desplazados)
rolling_mean = returns.rolling(window).mean()
rolling_std = returns.rolling(window).std()

# VaR paramétrico normal
VaR_95_norm = norm.ppf(0.05, rolling_mean, rolling_std)
VaR_99_norm = norm.ppf(0.01, rolling_mean, rolling_std)

ES_95_norm = rolling_mean - rolling_std * norm.pdf(norm.ppf(0.05)) / 0.05
ES_99_norm = rolling_mean - rolling_std * norm.pdf(norm.ppf(0.01)) / 0.01

#Var Historico
VaR_95_hist = returns.rolling(window).quantile(0.05)
VaR_99_hist = returns.rolling(window).quantile(0.01)

ES_95_hist = returns.rolling(window).apply(lambda x: x[x <= np.quantile(x, 0.05)].mean())
ES_99_hist = returns.rolling(window).apply(lambda x: x[x <= np.quantile(x, 0.01)].mean())


df_roll = pd.DataFrame({
    "Returns": returns,
    "VaR 95 Norm": VaR_95_norm,
    "VaR 99 Norm": VaR_99_norm,
    "ES 95 Norm": ES_95_norm,
    "ES 99 Norm": ES_99_norm,
    "VaR 95 Hist": VaR_95_hist,
    "VaR 99 Hist": VaR_99_hist,
    "ES 95 Hist": ES_95_hist,
    "ES 99 Hist": ES_99_hist
}).dropna()
st.header("Inciso (d):Rolling VaR y ES")
st.subheader("Rolling VaR y ES")

fig, ax = plt.subplots(figsize=(14,6))

ax.plot(df_roll.index, df_roll["Returns"], label="Rendimientos", alpha=0.6)

ax.plot(df_roll.index, df_roll["VaR 95 Norm"], label="VaR 95% Normal")
ax.plot(df_roll.index, df_roll["VaR 99 Norm"], label="VaR 99% Normal")

ax.plot(df_roll.index, df_roll["VaR 95 Hist"], label="VaR 95% Hist", linestyle='--')
ax.plot(df_roll.index, df_roll["VaR 99 Hist"], label="VaR 99% Hist", linestyle='--')

ax.plot(df_roll.index, df_roll["ES 95 Norm"], label="ES 95% Normal", linestyle=':')
ax.plot(df_roll.index, df_roll["ES 99 Norm"], label="ES 99% Normal", linestyle=':')

ax.plot(df_roll.index, df_roll["ES 95 Hist"], label="ES 95% Hist", linestyle='-.')
ax.plot(df_roll.index, df_roll["ES 99 Hist"], label="ES 99% Hist", linestyle='-.')

ax.axhline(0, linestyle='--')

ax.legend()
ax.set_title("Rolling VaR y ES (252 días)")
st.pyplot(fig)

#Interpretacion
st.subheader("Interpretación de VaR y ES (Rolling Window)")

st.write("""
El VaR (Value at Risk) y el ES (Expected Shortfall) fueron calculados utilizando ventanas móviles de 252 días,
lo que nos permite analizar cómo evoluciona el riesgo del activo a lo largo del tiempo.

En la gráfica se comparan los rendimientos reales con las estimaciones de VaR y ES bajo dos enfoques:
histórico y paramétrico (normal).
""")

# Violaciones del VaR 
violaciones_95 = (df_roll["Returns"] < df_roll["VaR 95 Norm"]).sum()
violaciones_99 = (df_roll["Returns"] < df_roll["VaR 99 Norm"]).sum()

total = len(df_roll)

st.write(f"""
- Para un nivel de confianza del 95%, Teoricamente, las perdidas deberian exceder el VaR estimado el **5% del tiempo**.
  En este caso, se observaron **{violaciones_95} violaciones** de un total de **{total} observaciones**
  ({violaciones_95/total:.2%}).

- Para un nivel de confianza del 99%, el umbral de tolerancia teorico es de el **1%**.
En este caso, se observaron **{violaciones_99} violaciones**
  ({violaciones_99/total:.2%}).

- Los resultados del backtesting nos dice que mientras que al 95% de confianza
el modelo presenta una calibración aceptable ({violaciones_95/total:.2%}) numero de violaciones empíricas frente al 5% teórico), 
al nivel del 99% el numero de violaciones observadas ({violaciones_99/total:.2%}) superan por más del doble la tolerancia esperada (1%).
""")

st.write("""
Interpretación:

- Cuando los rendimientos caen por debajo del VaR, se dice que ocurre una **Excepcion (VaR violations)**.
- Si el número de excepciones es mayor al esperado, el modelo está subestimando el riesgo.
- Si es menor, el modelo puede ser demasiado conservador.

En general:

*   **El VaR Parametrico (normal):** Este enfoque ajusta una distribución teórica. Sin embargo, los "hechos estilizados" 
    de los activos financieros demuestran que los retornos no siguen una distribución normal; presentan asimetría y colas pesadas
    (distribuciones leptocúrticas con curtosis $\kappa > 3$). Obligar a los datos a ajustarse a una curva normal genera "riesgo de modelo"
    y subestima drásticamente la probabilidad de eventos extremos.
*   **El VaR Histórico:** Al basarse completamente en la distribución empírica de los retornos pasados, 
    este método evita el error de asumir normalidad. No obstante, tiene una debilidad teórica fundamental: "este método no es capaz de capturar valores extremos (si es que no existen en los datos)".
    Es decir, si la ventana histórica no incluye un evento de estrés severo, el modelo asumirá equivocadamente que el riesgo de cola es inexistente.
*   **Expected Shortfall (ES):** La principal desventaja del VaR es que "no considera los riesgos de cola"; solo indica el umbral,
    pero ignora la magnitud de los valores extremos. El ES (o Conditional VaR) resuelve esto calculando matemáticamente el valor esperado de la pérdida.

""")

# INCISO (E)


st.header("Inciso (e)")

# Total de observaciones
n = len(df_roll)

resultados_violaciones = []

# Niveles
niveles = ["95", "99"]

for nivel in niveles:
  
    # VaR Normal
    var_norm = df_roll[f"VaR {nivel} Norm"]
    viol_var_norm = (df_roll["Returns"] < var_norm).sum()

    # ES Normal
    es_norm = df_roll[f"ES {nivel} Norm"]
    viol_es_norm = (df_roll["Returns"] < es_norm).sum()

    # VaR Histórico
    var_hist = df_roll[f"VaR {nivel} Hist"]
    viol_var_hist = (df_roll["Returns"] < var_hist).sum()

    # ES Histórico
    es_hist = df_roll[f"ES {nivel} Hist"]
    viol_es_hist = (df_roll["Returns"] < es_hist).sum()

    resultados_violaciones.append({
        "Nivel": f"{nivel}%",
        
        "Violaciones VaR Normal": viol_var_norm,
        "% VaR Normal": viol_var_norm / n,
        
        "Violaciones ES Normal": viol_es_norm,
        "% ES Normal": viol_es_norm / n,
        
        "Violaciones VaR Hist": viol_var_hist,
        "% VaR Hist": viol_var_hist / n,
        
        "Violaciones ES Hist": viol_es_hist,
        "% ES Hist": viol_es_hist / n
    })

# Crear tabla
df_violaciones = pd.DataFrame(resultados_violaciones)

# Formato porcentaje
for col in df_violaciones.columns:
    if "%" in col:
        df_violaciones[col] = df_violaciones[col].apply(lambda x: f"{x:.2%}")

st.dataframe(df_violaciones)


# INCISO (f)

st.header("Inciso (f): VaR con volatilidad móvil")

returns = df_rendimientos[stock_seleccionado].dropna()

window = 252

# Volatilidad rolling 
rolling_vol = returns.rolling(window).std()

# Cuantiles normales
q_05 = norm.ppf(0.05)
q_01 = norm.ppf(0.01)

# VaR
VaR_95_vol = q_05 * rolling_vol
VaR_99_vol = q_01 * rolling_vol

# DataFrame
df_vol = pd.DataFrame({
    "Returns": returns,
    "VaR 95 Vol": VaR_95_vol,
    "VaR 99 Vol": VaR_99_vol
}).dropna()

# Grafica
st.subheader("VaR con volatilidad móvil vs Rendimientos")

fig, ax = plt.subplots(figsize=(14,6))

ax.plot(df_vol.index, df_vol["Returns"], label="Rendimientos", alpha=0.6)

ax.plot(df_vol.index, df_vol["VaR 95 Vol"], label="VaR 95% (Volatilidad)")
ax.plot(df_vol.index, df_vol["VaR 99 Vol"], label="VaR 99% (Volatilidad)")

ax.axhline(0, linestyle='--')

ax.set_title("VaR con volatilidad móvil (252 días)")
ax.legend()

st.pyplot(fig)

# Violaciones 

st.subheader("Violaciones del VaR (Volatilidad móvil)")

n = len(df_vol)

viol_95 = (df_vol["Returns"] < df_vol["VaR 95 Vol"]).sum()
viol_99 = (df_vol["Returns"] < df_vol["VaR 99 Vol"]).sum()

df_viol_vol = pd.DataFrame({
    "Nivel": ["95%", "99%"],
    "Violaciones": [viol_95, viol_99],
    "Porcentaje": [viol_95/n, viol_99/n]
})

df_viol_vol["Porcentaje"] = df_viol_vol["Porcentaje"].apply(lambda x: f"{x:.2%}")

st.dataframe(df_viol_vol)


# streamlit run Proyecto_1_MCF_1.py
----->para problemas con el directorio
# cd "Proyecto MCF"
# streamlit run Proyecto_1_MCF_1.py          
