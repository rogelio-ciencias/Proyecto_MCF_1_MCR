# Proyecto_1_MCF_1
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import kurtosis, skew, shapiro, norm

st.title("Visualización de Rendimientos de Acciones")
st.header("Proyecto  ")
# ===============================
# INCISO (A)
# ===============================
@st.cache_data
def obtener_datos(stocks):
    df = yf.download(stocks, start="2010-01-01")['Close']
    return df

@st.cache_data
def calcular_rendimientos(df):
    return df.pct_change().dropna()

# Activo (S&P 100)
stocks_lista = ['^OEX']

with st.spinner("Descargando datos..."):
    df_precios = obtener_datos(stocks_lista)
    df_rendimientos = calcular_rendimientos(df_precios)

# Selector
stock_seleccionado = st.selectbox("Selecciona una acción", stocks_lista)

if stock_seleccionado:

    # ===============================
    # INCISO (b)
    # ===============================
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

    # ===============================
    # INCISO (c)
    # ===============================
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

    df_resultados = pd.DataFrame(resultados)
    df_resultados.iloc[:, 1:] = df_resultados.iloc[:, 1:].applymap(lambda x: f"{x:.4%}")

    st.dataframe(df_resultados)

# streamlit run Proyecto_1_MCF_1.py





