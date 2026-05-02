import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm, t
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go


# ==============================
# CONFIGURACIÓN
# ==============================
st.set_page_config(page_title="Análisis del Trigo", layout="wide")

st.title("Análisis del Trigo")

# ==============================
# AUTOR
# ==============================
st.markdown("""
Alumno: Carlos Yeriel Santoyo Cortés  
Facultad de Ciencias
""")

# ==============================
# DESCRIPCIÓN DEL ACTIVO
# ==============================
st.subheader("Descripción del activo")

st.write("""
El activo analizado corresponde al contrato de futuros del trigo, identificado en Yahoo Finance con el ticker **ZW=F**.

El trigo es una **materia prima agrícola (commodity)** ampliamente negociada en los mercados financieros internacionales. 
Su precio está influenciado por diversos factores, entre los que destacan:

- 🌦️ Condiciones climáticas  
- 🌍 Oferta y demanda global  
- 📦 Niveles de inventarios  
- 🚢 Costos de transporte  
- 🌐 Eventos geopolíticos  

Los datos utilizados en este análisis fueron obtenidos automáticamente desde **Yahoo Finance**, considerando información diaria desde el **1 de enero de 2010** hasta la fecha más reciente disponible.
""")
# ======================================================================
# 🔴 INCISO A
# ======================================================================

ticker = "ZW=F"

df = yf.download(
    ticker,
    start="2010-01-01",
    interval="1d"
)

# Si Yahoo devuelve columnas MultiIndex, las limpiamos
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

# ======================================================================
# 🔵 INCISO B: Calculo de rendimientos
# ======================================================================

# Calculamos el rendimiento diario del trigo.
# Fórmula:
# Rendimiento = (Precio de hoy / Precio de ayer) - 1
df["Returns"] = df["Close"].pct_change()

# Eliminamos los valores vacíos generados por pct_change().
# El primer día no tiene rendimiento porque no existe un precio anterior.
df = df.dropna(subset=["Returns"])


# ==============================
# GRÁFICAS
# ==============================

# Gráfica del precio de cierre del trigo
st.subheader("Precio del trigo")
st.line_chart(df["Close"])

# Gráfica de los rendimientos diarios
st.subheader("Rendimientos diarios del trigo")
st.line_chart(df["Returns"])


# ==============================
# ESTADÍSTICAS DESCRIPTIVAS
# ==============================

# Guardamos la serie de rendimientos en una variable.
# Esto evita repetir df["Returns"] muchas veces.
rendimientos = df["Returns"]

# Media diaria de los rendimientos
media = rendimientos.mean()

# Volatilidad diaria aproximada
desviacion = rendimientos.std()

# Sesgo: mide si la distribución está cargada hacia pérdidas o ganancias extremas
sesgo = rendimientos.skew()

# Exceso de curtosis: mide qué tan pesadas son las colas de la distribución
exceso_curtosis = rendimientos.kurt()

st.subheader("Métricas de rendimiento")

# Creamos una tabla para mostrar las estadísticas principales
tabla_estadisticas = pd.DataFrame({
    "Media": [media],
    "Desviación estándar": [desviacion],
    "Sesgo": [sesgo],
    "Exceso de curtosis": [exceso_curtosis]
}, index=["Trigo (ZW=F)"])

# Mostramos la tabla con formato de 6 decimales
st.dataframe(tabla_estadisticas.style.format("{:.6f}"))

# ==============================
# INTERPRETACIONES
# ==============================
st.subheader(" Interpretaciones")

with st.expander("Ver interpretación de la media"):
    st.write(f"Media de los rendimientos: {media:.6f}")

    st.write("""
    La media representa el rendimiento diario promedio del trigo.

    En este caso, su valor es cercano a cero, lo que indica que no existe una tendencia
    diaria clara en el comportamiento del precio, sino que este fluctúa constantemente.

    Aunque una media positiva (o negativa) podría sugerir una ligera tendencia alcista
    (o bajista) en el largo plazo, en series financieras diarias este valor suele ser
    muy pequeño en comparación con la volatilidad del activo.

    Por ello, la media tiene una capacidad limitada para explicar o predecir movimientos
    en el corto plazo.
    """)

with st.expander("Ver interpretación del sesgo"):
    st.write(f"Valor del sesgo: β = {sesgo:.6f}")

    if sesgo > 0:
        st.success("β > 0: sesgo positivo")
        st.write("""
        La distribución tiene una cola derecha más larga. Esto indica mayor presencia de rendimientos extremos positivos.
        """)
    elif sesgo < 0:
        st.error("β < 0: sesgo negativo")
        st.write("""
        La distribución tiene una cola izquierda más larga. Esto indica mayor presencia de rendimientos extremos negativos.
        """)
    else:
        st.info("β = 0: distribución aproximadamente simétrica")

with st.expander("Ver interpretación de la curtosis"):
    st.write(f"Valor del exceso de curtosis: κ - 3 = {exceso_curtosis:.6f}")

    if exceso_curtosis > 0:
        st.warning("κ - 3 > 0: exceso de curtosis positivo")
        st.write("""
        La distribución es leptocúrtica. Esto indica colas pesadas y mayor presencia de eventos extremos que en una distribución normal.
        """)
    elif exceso_curtosis < 0:
        st.info("κ - 3 < 0: exceso de curtosis negativo")
        st.write("""
        La distribución es platicúrtica. Esto indica colas más ligeras que una distribución normal.
        """)
    else:
        st.success("κ - 3 = 0: curtosis similar a la normal")

        # ==============================
## ==============================
# HISTOGRAMA + KDE + NORMAL
# ==============================

st.subheader("Distribución de los rendimientos")

# Creamos la figura y el eje donde se dibujará la gráfica
fig, ax = plt.subplots(figsize=(10, 5))

# ------------------------------
# Histograma
# ------------------------------
# Muestra cómo se distribuyen los rendimientos diarios.
# stat="density" convierte la frecuencia en densidad,
# permitiendo compararlo con la KDE y la curva normal.
sns.histplot(
    rendimientos,
    bins=50,
    stat="density",
    ax=ax,
    color="lightblue",
    edgecolor="black",
    alpha=0.7
)

# ------------------------------
# KDE
# ------------------------------
# La KDE suaviza la distribución observada.
# Sirve para ver mejor la forma real de los rendimientos.
sns.kdeplot(
    rendimientos,
    ax=ax,
    color="blue",
    linewidth=2,
    label="KDE (distribución empírica)"
)

# ------------------------------
# Curva normal teórica
# ------------------------------
# Creamos una distribución normal usando la media y desviación estándar
# de los rendimientos reales.
x = np.linspace(rendimientos.min(), rendimientos.max(), 1000)
y = norm.pdf(x, loc=media, scale=desviacion)

ax.plot(
    x,
    y,
    color="red",
    linewidth=2,
    label="Normal teórica"
)

# ------------------------------
# Detalles visuales
# ------------------------------
ax.set_title("Distribución de rendimientos diarios del trigo")
ax.set_xlabel("Rendimiento diario")
ax.set_ylabel("Densidad")
ax.legend()

# Mostramos la gráfica en Streamlit
st.pyplot(fig)

# ------------------------------
# Interpretación
# ------------------------------
st.write("""
El histograma muestra cómo se distribuyen los rendimientos diarios del trigo.

La curva azul representa la distribución empírica suavizada mediante KDE,
mientras que la curva roja representa una distribución normal teórica construida
con la media y la desviación estándar de los rendimientos.

Si la curva azul se aleja mucho de la curva roja, puede indicar que los rendimientos
no siguen una distribución normal. Esto suele pasar en series financieras, donde
pueden existir asimetrías, valores extremos o colas pesadas.
""")


# ======================================================================
# 🔴 INCISO C: MÉTRICAS DE RIESGO (VaR y Expected Shortfall)
# ======================================================================

st.markdown("---")
st.header(" VaR y Expected Shortfall (ES)")

# --------------------------------------------------
# CONVERSIÓN A PÉRDIDAS
# --------------------------------------------------
# En riesgo financiero se trabaja con pérdidas:
# pérdida = -rendimiento
# Así, valores positivos representan pérdidas.
losses = -rendimientos.dropna()

# --------------------------------------------------
# NIVELES DE CONFIANZA
# --------------------------------------------------
# 95%, 97.5% y 99% son estándares en finanzas
alphas = [0.95, 0.975, 0.99]

# Lista donde guardaremos todos los resultados
resultados = []

# --------------------------------------------------
# PARÁMETROS DE LA DISTRIBUCIÓN NORMAL
# --------------------------------------------------
mu = losses.mean()
sigma = losses.std()

# --------------------------------------------------
# AJUSTE DE DISTRIBUCIÓN t-STUDENT
# --------------------------------------------------
# Se usa porque captura mejor colas pesadas (eventos extremos)
from scipy.stats import t

df_t, loc_t, scale_t = t.fit(losses)

# --------------------------------------------------
# MONTE CARLO (basado en t-Student)
# --------------------------------------------------
np.random.seed(123)  # reproducibilidad
n_sim = 100000

mc_losses = t.rvs(df_t, loc=loc_t, scale=scale_t, size=n_sim)

# --------------------------------------------------
# CÁLCULO DE VaR Y ES PARA CADA NIVEL
# --------------------------------------------------
for alpha in alphas:

    # ==========================
    # 1. NORMAL PARAMÉTRICO
    # ==========================
    z = norm.ppf(alpha)

    var_normal = mu + sigma * z

    # Expected Shortfall (cola de la distribución)
    es_normal = mu + sigma * (norm.pdf(z) / (1 - alpha))

    resultados.append({
        "Método": "Paramétrico Normal",
        "Alpha": alpha,
        "VaR": var_normal,
        "ES": es_normal
    })

    # ==========================
    # 2. t-STUDENT PARAMÉTRICO
    # ==========================
    q_t = t.ppf(alpha, df_t)

    var_t = loc_t + scale_t * q_t

    es_t = loc_t + scale_t * (
        ((df_t + q_t**2) / (df_t - 1)) *
        (t.pdf(q_t, df_t) / (1 - alpha))
    )

    resultados.append({
        "Método": "Paramétrico t-Student",
        "Alpha": alpha,
        "VaR": var_t,
        "ES": es_t
    })

    # ==========================
    # 3. HISTÓRICO
    # ==========================
    # Percentil directamente de los datos
    var_hist = losses.quantile(alpha)

    # Promedio de pérdidas extremas
    es_hist = losses[losses >= var_hist].mean()

    resultados.append({
        "Método": "Histórico",
        "Alpha": alpha,
        "VaR": var_hist,
        "ES": es_hist
    })

    # ==========================
    # 4. MONTE CARLO
    # ==========================
    var_mc = np.quantile(mc_losses, alpha)

    es_mc = mc_losses[mc_losses >= var_mc].mean()

    resultados.append({
        "Método": "Monte Carlo",
        "Alpha": alpha,
        "VaR": var_mc,
        "ES": es_mc
    })




# ==============================
# TABLA FINAL
# ==============================
tabla_var_es = pd.DataFrame(resultados)


tabla_var_es["Interpretación"] = tabla_var_es.apply(
    lambda row: (
        f"VaR indica una pérdida de hasta {row['VaR']:.2%}; "
        f"ES estima una pérdida de {row['ES']:.2%}"
    ),
    axis=1
)

st.dataframe(
    tabla_var_es.style.format({
        "Alpha": "{:.3f}",
        "VaR": "{:.4%}",
        "ES": "{:.4%}"
    })
)
# ==============================
# INTERPRETACIÓN 
# ==============================
st.subheader("Resumen de riesgo")

with st.expander("Ver interpretación del VaR y Expected Shortfall"):
    st.markdown("""
    Riesgo

    - El VaR indica las pérdidas bajo condiciones normales, pero no captura completamente eventos extremos.  
    - El ES muestra pérdidas promedio en escenarios adversos, reflejando mejor el riesgo real.  

    Modelos

    - El modelo normal tiende a subestimar el riesgo.  
    - t-Student y Monte Carlo capturan mejor colas pesadas y eventos extremos.  

    Conclusión

    - El VaR no es una medida coherente (no siempre refleja el beneficio de diversificación).  
    - El ES es más robusto y adecuado para medir riesgo extremo.
    """)



# ======================================================================
# 🔴 INCISO D:  Rolling VaR y Expected Shortfall
# ======================================================================


st.subheader(" Rolling VaR y Expected Shortfall")

st.markdown("""
Esta sección calcula el **VaR** y el **Expected Shortfall (ES)** de forma dinámica usando
una ventana móvil de **252 rendimientos**, equivalente aproximadamente a un año bursátil.
""")

# Parámetros
window = 252
alphas_rolling = [0.95, 0.99]

# Retornos y pérdidas
returns = df["Returns"].dropna()
losses = -returns

# DataFrame para resultados rolling
rolling_results = pd.DataFrame(index=returns.index)

# P&L: rendimientos reales observados
rolling_results["P&L"] = returns

# ==============================
# Cálculo Rolling
# ==============================
for alpha in alphas_rolling:

    # VaR Histórico
    rolling_results[f"VaR Hist {alpha:.0%}"] = (
        losses.rolling(window).quantile(alpha)
    )

    # ES Histórico
    rolling_results[f"ES Hist {alpha:.0%}"] = losses.rolling(window).apply(
        lambda x: x[x >= np.quantile(x, alpha)].mean(),
        raw=False
    )

    # Parámetros rolling para normal
    rolling_mu = losses.rolling(window).mean()
    rolling_sigma = losses.rolling(window).std()

    z = norm.ppf(alpha)

    # VaR Normal
    rolling_results[f"VaR Normal {alpha:.0%}"] = (
        rolling_mu + rolling_sigma * z
    )

    # ES Normal
    rolling_results[f"ES Normal {alpha:.0%}"] = (
        rolling_mu + rolling_sigma * norm.pdf(z) / (1 - alpha)
    )

# ==============================
# Desfase predictivo
# ==============================

# Hasta este punto, Pandas ya calculó VaR y ES usando ventanas de 252 días.
# Por ejemplo:
# r1, r2, ..., r252  → calcula VaR/ES
# r2, r3, ..., r253  → calcula VaR/ES
#
# Pero todavía falta mover esas estimaciones un día adelante,
# porque el VaR/ES calculado con r1,...,r252 debe compararse contra r253.

# Seleccionamos todas las columnas de riesgo:
# VaR Histórico, ES Histórico, VaR Normal y ES Normal.
# No incluimos "P&L" porque ese es el rendimiento real observado.
risk_cols = [col for col in rolling_results.columns if col != "P&L"]

# Movemos las columnas de VaR y ES un día hacia adelante.
# Esto hace que:
# VaR/ES calculado con r1,...,r252  → se use para evaluar r253
# VaR/ES calculado con r2,...,r253  → se use para evaluar r254
#
# En otras palabras, evitamos usar información del mismo día
# y hacemos que el modelo use solo información pasada.
rolling_results[risk_cols] = rolling_results[risk_cols].shift(1)

# Después del shift quedan valores vacíos al inicio,
# porque los primeros días no tienen suficientes datos previos
# para calcular una ventana de 252 rendimientos.
# Por eso eliminamos esas filas vacías.
rolling_results = rolling_results.dropna()

# ==============================
# Diseño interactivo
# ==============================
st.markdown("###  Serie de tiempo ")

dias_mostrar = st.slider(
    "Selecciona cuántos días recientes quieres visualizar:",
    min_value=250,
    max_value=min(2500, len(rolling_results)),
    value=min(1000, len(rolling_results)),
    step=50
)

rolling_plot = rolling_results.tail(dias_mostrar)

tab95, tab99 = st.tabs(["VaR y ES 95%", "VaR y ES 99%"])


# ==============================
# Función para graficar
# ==============================

# Esta función recibe un nivel de confianza en formato texto, por ejemplo:
# "95%" o "99%".
# Con ese valor selecciona automáticamente las columnas correspondientes
# de VaR y ES para graficarlas.
def graficar_rolling(alpha_label):

    # Creamos una figura vacía de Plotly.
    # Aquí iremos agregando cada línea de la gráfica.
    fig = go.Figure()

    # ==============================
    # P&L real
    # ==============================

    # Esta línea representa los rendimientos reales observados del trigo.
    # Si el valor es positivo, hubo ganancia.
    # Si el valor es negativo, hubo pérdida.
    fig.add_trace(go.Scatter(
        x=rolling_plot.index,          # Fechas en el eje X
        y=rolling_plot["P&L"],         # Rendimientos reales en el eje Y
        mode="lines",                  # Gráfica de líneas
        name="P&L real",               # Nombre que aparece en la leyenda
        line=dict(width=1)             # Grosor de la línea
    ))

    # ==============================
    # VaR Histórico
    # ==============================

    # El VaR histórico se calculó usando la distribución empírica
    # de las pérdidas dentro de cada ventana móvil de 252 días.
    #
    # Como el VaR fue calculado como pérdida positiva,
    # se multiplica por -1 para mostrarlo en la gráfica como rendimiento negativo.
    fig.add_trace(go.Scatter(
        x=rolling_plot.index,
        y=-rolling_plot[f"VaR Hist {alpha_label}"],
        mode="lines",
        name=f"VaR Histórico {alpha_label}",
        line=dict(width=2)
    ))

    # ==============================
    # ES Histórico
    # ==============================

    # El ES histórico representa la pérdida promedio en los escenarios
    # donde se supera el VaR histórico.
    #
    # También se multiplica por -1 porque queremos visualizarlo
    # como una pérdida en el eje negativo.
    #
    # La línea punteada ayuda a distinguir el ES del VaR.
    fig.add_trace(go.Scatter(
        x=rolling_plot.index,
        y=-rolling_plot[f"ES Hist {alpha_label}"],
        mode="lines",
        name=f"ES Histórico {alpha_label}",
        line=dict(width=2, dash="dash")
    ))

    # ==============================
    # VaR Normal
    # ==============================

    # El VaR normal se calcula suponiendo que las pérdidas siguen
    # una distribución normal dentro de cada ventana móvil.
    #
    # Igual que antes, se multiplica por -1 para mostrarlo como pérdida.
    fig.add_trace(go.Scatter(
        x=rolling_plot.index,
        y=-rolling_plot[f"VaR Normal {alpha_label}"],
        mode="lines",
        name=f"VaR Normal {alpha_label}",
        line=dict(width=2)
    ))

    # ==============================
    # ES Normal
    # ==============================

    # El ES normal representa la pérdida promedio esperada
    # más allá del VaR bajo el supuesto de normalidad.
    #
    # Se grafica con línea punteada para diferenciarlo visualmente.
    fig.add_trace(go.Scatter(
        x=rolling_plot.index,
        y=-rolling_plot[f"ES Normal {alpha_label}"],
        mode="lines",
        name=f"ES Normal {alpha_label}",
        line=dict(width=2, dash="dash")
    ))

    # ==============================
    # Línea de referencia en cero
    # ==============================

    # Esta línea separa ganancias y pérdidas:
    # arriba de cero = ganancia
    # abajo de cero = pérdida
    fig.add_hline(
        y=0,
        line_dash="dot",
        annotation_text="Cero",
        annotation_position="top left"
    )

    # ==============================
    # Diseño de la gráfica
    # ==============================

    fig.update_layout(
        title=f"Rolling VaR y Expected Shortfall ({alpha_label})",
        xaxis_title="Fecha",
        yaxis_title="Rendimiento / pérdida",

        # hovermode="x unified" muestra todos los valores
        # de las líneas para una misma fecha al pasar el cursor.
        hovermode="x unified",

        # Altura de la gráfica en píxeles.
        height=550,

        # Configuración de la leyenda.
        # Se coloca debajo de la gráfica para que no estorbe.
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.35,
            xanchor="center",
            x=0.5
        )
    )

    # La función devuelve la figura ya construida.
    # Después se muestra en Streamlit con st.plotly_chart(...)
    return fig

# ==============================
# Gráfica 95%
# ==============================
with tab95:
    st.plotly_chart(
        graficar_rolling("95%"),
        use_container_width=True
    )

    with st.expander("Ver interpretación 95%"):
        st.markdown("""
        **Interpretación**

        - El nivel de confianza de **95%** mide pérdidas relativamente frecuentes.
        - Cuando el **P&L real** cae por debajo del VaR, la pérdida fue mayor a la esperada.
        - El **ES** se ubica más abajo porque mide pérdidas promedio en escenarios extremos.
        - El modelo histórico usa directamente los datos pasados, mientras que el normal asume una distribución normal.
        """)


# ==============================
# Gráfica 99%
# ==============================
with tab99:
    st.plotly_chart(
        graficar_rolling("99%"),
        use_container_width=True
    )

    with st.expander("Ver interpretación 99%"):
        st.markdown("""
        **Interpretación**

        - El nivel de confianza de **99%** es más conservador que el 95%.
        - Sus líneas de VaR y ES suelen estar más alejadas de cero porque representan pérdidas más extremas.
        - Si el P&L cruza el VaR al 99%, se trata de un evento poco frecuente y de alto impacto.
        - Este nivel es útil para analizar escenarios de estrés o crisis.
        """)

# ======================================================================
# 🔴 INCISO E: BACKTESTING - VIOLACIONES DE VaR Y EXPECTED SHORTFALL
# ======================================================================

st.markdown("---")
st.header(" Violaciones de VaR y Expected Shortfall")

# --------------------------------------------------
# OBJETIVO DEL INCISO
# --------------------------------------------------
# En este inciso se evalúa qué tan eficientes fueron las estimaciones
# de riesgo calculadas previamente con ventanas móviles.
#
# Una violación ocurre cuando la pérdida real observada supera la
# pérdida estimada por el VaR o el Expected Shortfall.
#
# De acuerdo con la nota del ejercicio:
# Una buena estimación debe generar un porcentaje de violaciones
# menor al 2.5%.

st.subheader("Tabla de violaciones")

# Lista donde se guardarán los resultados de cada modelo
violaciones_resultados = []

# Número total de observaciones disponibles para evaluar las violaciones
n_total = len(rolling_results)

# Umbral indicado en la nota del ejercicio
umbral_violaciones = 0.025


# --------------------------------------------------
# CÁLCULO DE VIOLACIONES
# --------------------------------------------------

for alpha in alphas_rolling:

    # Convertimos el nivel de confianza a texto.
    # Ejemplo: 0.95 -> "95%"
    label = f"{alpha:.0%}"

    # Columnas que contienen las estimaciones de riesgo
    # que se van a comparar contra el rendimiento real.
    columnas_riesgo = [
        f"VaR Hist {label}",
        f"ES Hist {label}",
        f"VaR Normal {label}",
        f"ES Normal {label}"
    ]

    for col in columnas_riesgo:

        # VaR y ES fueron calculados como pérdidas positivas.
        # Sin embargo, los rendimientos negativos representan pérdidas.
        # Por eso multiplicamos por -1 para convertir el límite de riesgo
        # a la misma escala que el rendimiento real.
        limite_riesgo = -rolling_results[col]

        # Se considera violación cuando el rendimiento real observado
        # cae por debajo del límite estimado por el VaR o ES.
        #
        # Ejemplo:
        # Si el VaR es 3%, el límite en retornos es -3%.
        # Hay violación si el rendimiento real fue menor a -3%.
        violaciones = rolling_results["P&L"] < limite_riesgo

        # Número total de violaciones
        num_violaciones = violaciones.sum()

        # Porcentaje de violaciones respecto al tamaño de la muestra
        porcentaje_violaciones = num_violaciones / n_total

        # Guardamos los resultados en una lista
        violaciones_resultados.append({
            "Nivel de confianza": alpha,
            "Medida": col,
            "Violaciones": num_violaciones,
            "Porcentaje": porcentaje_violaciones
        })


# --------------------------------------------------
# TABLA DE RESULTADOS
# --------------------------------------------------

tabla_violacione = pd.DataFrame(violaciones_resultados)

st.dataframe(
    tabla_violaciones.style.format({
        "Nivel de confianza": "{:.3f}",
        "Porcentaje": "{:.2%}"
    })
)



st.subheader("Interpretación de violaciones")

with st.expander("Ver interpretación"):

    st.markdown("""
    **Idea clave**

    Las violaciones muestran las ocasiones en las que la pérdida real fue mayor
    que la pérdida estimada por el VaR o por el Expected Shortfall.

    En términos prácticos, una violación significa que el modelo subestimó el
    riesgo para ese periodo, porque la pérdida observada fue más severa que la
    pérdida que el modelo había anticipado.

    **Criterio del ejercicio**

    De acuerdo con la nota del ejercicio, una buena estimación debe generar un
    porcentaje de violaciones menor al **2.5%**.

    Por lo tanto:

    - Si el porcentaje de violaciones es menor al **2.5%**, la estimación puede
      considerarse prudente.
    - Si el porcentaje de violaciones es mayor al **2.5%**, el modelo no cumple
      con el criterio solicitado y puede estar subestimando el riesgo.
    """)

    st.markdown("""
    **Interpretación de resultados**

    Al comparar los porcentajes de violaciones obtenidos para cada medida de
    riesgo, se observa que los modelos no tienen el mismo desempeño.

    Las estimaciones con nivel de confianza del **95%** suelen generar más
    violaciones, ya que el umbral de pérdida es menos extremo. Por esta razón,
    algunas medidas al 95% pueden superar el límite del **2.5%** establecido
    en el ejercicio.

    En cambio, las estimaciones con nivel de confianza del **99%** tienden a
    generar menos violaciones, porque utilizan un umbral de pérdida más
    conservador. Esto significa que son más estrictas al estimar eventos
    extremos.

    Además, el **Expected Shortfall (ES)** suele presentar menos violaciones que
    el VaR, ya que no solo considera un percentil específico de la distribución,
    sino también el comportamiento promedio de las pérdidas más severas.
    """)

    st.markdown("""
    **Conclusión**

    Con base en el criterio del **2.5%**, las mejores estimaciones son aquellas
    que presentan un porcentaje de violaciones por debajo de dicho umbral.

    Si una medida supera ese porcentaje, entonces no cumple completamente con
    la condición establecida en el ejercicio, porque permitió demasiados casos
    en los que la pérdida real fue mayor que la pérdida estimada.

    En general, los modelos más conservadores, especialmente aquellos asociados
    con niveles de confianza más altos y con Expected Shortfall, ofrecen una
    mejor protección frente a pérdidas extremas.

    Por lo tanto, para fines de administración de riesgo, es preferible utilizar
    una medida que no subestime el riesgo, incluso si esto implica obtener una
    estimación más conservadora.
    """)

# ======================================================================
# 🔴 INCISO F VAR una volatilidad movil y asumiendo una distribucion normal
# ======================================================================

st.subheader(" VaR con volatilidad móvil")

st.markdown("""
Este modelo estima el VaR usando una **volatilidad móvil de 252 días**
y suponiendo que los rendimientos siguen una distribución normal.
""")

# ==============================
# Parámetro fijo del ejercicio
# ==============================

# La ventana se fija en 252 rendimientos porque representa aproximadamente
# un año bursátil. Esta es la ventana indicada en el ejercicio.
window = 252

# ==============================
# Controles interactivos
# ==============================

# Creamos dos columnas para que los controles se vean más ordenados.
col1, col2 = st.columns(2)

with col1:
    # El usuario puede elegir entre VaR al 95% o al 99%.
    # Esto NO cambia la ventana, solo cambia el nivel de confianza.
    nivel_confianza = st.selectbox(
        "Nivel de confianza",
        options=[0.95, 0.99],
        index=0,
        format_func=lambda x: f"{x:.0%}"
    )

with col2:
    # Este slider solo controla cuántos días se muestran en la gráfica.
    # No cambia el cálculo del VaR.
    # Sirve como zoom visual para no saturar la gráfica con toda la serie.
    dias_mostrar = st.slider(
        "Días recientes a visualizar",
        min_value=250,
        max_value=2500,
        value=1000,
        step=50
    )

# ==============================
# Preparación de datos
# ==============================

# Tomamos los rendimientos diarios del trigo y eliminamos valores vacíos.
returns = df["Returns"].dropna()

# Convertimos el nivel de confianza en alpha.
# Ejemplo:
# confianza = 95%  → alpha = 5%
# confianza = 99%  → alpha = 1%
alpha = 1 - nivel_confianza

# Calculamos la volatilidad móvil.
# Para cada día, se toma la desviación estándar de los últimos 252 rendimientos.
# Esto hace que la volatilidad cambie a través del tiempo.
rolling_sigma = returns.rolling(window).std()

# Creamos un DataFrame para guardar:
# 1. El rendimiento real observado (P&L)
# 2. El VaR estimado con volatilidad móvil
var_volatilidad = pd.DataFrame(index=returns.index)

# P&L real:
# Es el rendimiento diario observado del trigo.
# Si es positivo, hubo ganancia.
# Si es negativo, hubo pérdida.
var_volatilidad["P&L"] = returns

# ==============================
# Cálculo del VaR
# ==============================

# Obtenemos el cuantil de la normal estándar.
# Para 95% de confianza usamos alpha = 0.05, cuyo cuantil es aprox -1.645.
# Para 99% de confianza usamos alpha = 0.01, cuyo cuantil es aprox -2.33.
q_alpha = norm.ppf(alpha)

# Nombre dinámico de la columna según el nivel elegido.
# Ejemplo: "VaR Vol 95%" o "VaR Vol 99%".
var_col = f"VaR Vol {nivel_confianza:.0%}"

# Fórmula:
# VaR_t = q_alpha * sigma_t
#
# Como q_alpha es negativo, el VaR queda como rendimiento negativo.
# Esto permite compararlo directamente con el P&L real en la gráfica.
var_volatilidad[var_col] = q_alpha * rolling_sigma

# ==============================
# Desfase predictivo
# ==============================

# Hasta aquí, el VaR se calcula usando ventanas de 252 datos.
#
# Pero para que sea predictivo, debemos moverlo un día hacia adelante:
#
# r1, r2, ..., r252  → calculan VaR → se compara con r253
# r2, r3, ..., r253  → calculan VaR → se compara con r254
#
# Esto evita usar información del mismo día que queremos evaluar.
risk_cols = [col for col in var_volatilidad.columns if col != "P&L"]

# Movemos las columnas de VaR un día hacia adelante.
# La columna P&L no se mueve porque representa lo que realmente ocurrió ese día.
var_volatilidad[risk_cols] = var_volatilidad[risk_cols].shift(1)

# Eliminamos filas vacías.
# Los NaN aparecen porque:
# 1. Al inicio no hay 252 datos suficientes para calcular volatilidad.
# 2. El shift(1) genera un valor vacío adicional.
var_volatilidad = var_volatilidad.dropna()

# ==============================
# Selección de datos para graficar
# ==============================

# Nos aseguramos de no pedir más días de los disponibles.
dias_mostrar = min(dias_mostrar, len(var_volatilidad))

# Tomamos solo los últimos "dias_mostrar" datos para visualizar.
# Esto funciona como zoom visual.
var_plot = var_volatilidad.tail(dias_mostrar)

# ==============================
# Métricas de violaciones
# ==============================

# Una violación ocurre cuando el rendimiento real cae por debajo del VaR estimado.
#
# Ejemplo:
# P&L = -4%
# VaR = -2%
# Como -4% < -2%, la pérdida fue mayor a la esperada → violación.
violaciones = var_volatilidad["P&L"] < var_volatilidad[var_col]

# Mostramos métricas rápidas.
m1, m2 = st.columns(2)

with m1:
    st.metric("Número de violaciones", int(violaciones.sum()))

with m2:
    st.metric("Porcentaje de violaciones", f"{violaciones.mean():.2%}")

# ==============================
# Gráfica interactiva
# ==============================

st.subheader(" Serie de tiempo")

# Creamos una figura interactiva con Plotly.
fig = go.Figure()

# Línea de P&L real.
# Muestra los rendimientos diarios observados.
fig.add_trace(go.Scatter(
    x=var_plot.index,
    y=var_plot["P&L"],
    mode="lines",
    name="P&L real",
    line=dict(width=1)
))

# Línea del VaR estimado con volatilidad móvil.
# Esta línea representa el umbral de pérdida estimado para cada día.
fig.add_trace(go.Scatter(
    x=var_plot.index,
    y=var_plot[var_col],
    mode="lines",
    name=var_col,
    line=dict(width=2)
))

# Línea horizontal en cero.
# Sirve para separar ganancias de pérdidas.
fig.add_hline(
    y=0,
    line_dash="dot",
    annotation_text="Cero",
    annotation_position="top left"
)

# Diseño de la gráfica.
fig.update_layout(
    title=f"VaR con volatilidad móvil de 252 días ({nivel_confianza:.0%})",
    xaxis_title="Fecha",
    yaxis_title="Rendimiento / pérdida",
    hovermode="x unified",
    height=550,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5
    )
)

# Mostramos la gráfica en Streamlit.
st.plotly_chart(fig, use_container_width=True)

# Convertimos la lista en DataFrame para mostrarla como tabla.
# ==============================

# ==============================
# INTERPRETACIÓN 
# ==============================

with st.expander("Interpretación del VaR con volatilidad móvil"):
    st.markdown(f"""
    Contexto

    En este modelo, el VaR se estima utilizando una **volatilidad móvil de 252 días** bajo el supuesto de
    normalidad, lo que permite capturar cambios dinámicos en la variabilidad del mercado.

    Resultados

    - Para un nivel de confianza del **{nivel_confianza:.0%}**, se observa un porcentaje de violaciones de **{violaciones.mean():.2%}**.
    - Una violación ocurre cuando el rendimiento real es inferior al umbral estimado por el VaR.

    Evaluación del modelo**

    - Si el porcentaje de violaciones es cercano al nivel teórico esperado, el modelo se considera **bien calibrado**.  
    - Un porcentaje menor indica que el modelo es **conservador**, es decir, sobreestima el riesgo.  
    - Un porcentaje mayor sugiere que el modelo **subestima el riesgo**, lo cual es indeseable en gestión financiera.

     Criterio práctico

    - De acuerdo con el criterio establecido, una estimación adecuada debería presentar un porcentaje de violaciones inferior al **2.5%**, privilegiando modelos conservadores.

     Conclusión

    El enfoque basado en volatilidad móvil permite adaptar el VaR a las condiciones cambiantes del mercado.
    Sin embargo, al depender únicamente de la desviación estándar, puede no capturar completamente eventos extremos,
    por lo que su desempeño debe evaluarse cuidadosamente frente a otras metodologías.
    """)

# ==============================
# CONCLUSIONES FINALES
# ==============================

st.subheader("📌 Conclusiones finales")

with st.expander("Ver conclusiones del análisis"):
    st.markdown("""
    **📊 Síntesis del análisis**

    A lo largo del estudio se evaluaron distintas metodologías para la medición del riesgo del trigo,
    incluyendo enfoques históricos, paramétricos (normal y t-Student) y modelos dinámicos con ventanas móviles.

    **📈 Principales hallazgos**

    - El **VaR histórico** presenta un buen desempeño en niveles de confianza moderados (95%), pero tiende a subestimar el riesgo en escenarios extremos.
    - El **VaR bajo supuestos normales** resulta más conservador, aunque puede no reflejar adecuadamente la presencia de colas pesadas en los datos.
    - Los modelos basados en **t-Student y Monte Carlo** capturan mejor eventos extremos, evidenciando una mayor robustez frente a distribuciones no normales.
    - El **Expected Shortfall (ES)** demuestra ser una medida más consistente del riesgo extremo, al considerar la magnitud de las pérdidas más severas.
    - El modelo de **volatilidad móvil** permite adaptar el riesgo a las condiciones del mercado, reflejando cambios en la volatilidad a lo largo del tiempo.

    **⚠️ Evaluación del riesgo**

    - Se observa que los modelos que presentan un menor número de violaciones tienden a ser más conservadores.
    - En la práctica financiera, es preferible utilizar modelos que **no subestimen el riesgo**, incluso si esto implica sobreestimarlo ligeramente.
    - El criterio de mantener las violaciones por debajo del **2.5%** resulta adecuado para garantizar estimaciones prudentes.

    **🎯 Conclusión general**

    El análisis evidencia que no existe un único modelo óptimo, sino que cada enfoque presenta ventajas y limitaciones.
    Sin embargo, para fines de gestión de riesgo, resulta recomendable utilizar metodologías que capturen adecuadamente
    eventos extremos, como el **Expected Shortfall** o modelos con distribuciones más flexibles, complementados con
    enfoques dinámicos como la volatilidad móvil.

    En conjunto, una combinación de modelos proporciona una visión más completa y robusta del riesgo del activo.
    """)
