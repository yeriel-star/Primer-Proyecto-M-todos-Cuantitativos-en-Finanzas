# Análisis de riesgo del trigo (ZW=F)

## Descripción

Este proyecto analiza el comportamiento y riesgo del contrato de futuros del trigo (ZW=F) utilizando datos históricos obtenidos de Yahoo Finance.

Se estudian los rendimientos del activo y se aplican distintas metodologías de medición de riesgo, incluyendo Value at Risk (VaR) y Expected Shortfall (ES), tanto en enfoques estáticos como dinámicos.

---

## Autor

Carlos Yeriel Santoyo Cortés
Facultad de Ciencias

---

## Objetivos

* Analizar la distribución de los rendimientos del trigo
* Evaluar la presencia de eventos extremos
* Comparar distintas metodologías de medición de riesgo
* Implementar modelos dinámicos de riesgo

---

## Metodología

### Datos

* Fuente: Yahoo Finance
* Ticker: ZW=F
* Periodo: 2010 – actualidad
* Frecuencia: diaria

### Análisis realizado

* Cálculo de rendimientos
* Estadísticos descriptivos (media, desviación estándar, sesgo, curtosis)
* Análisis de distribución (histograma, KDE vs normal)
* Cálculo de VaR y Expected Shortfall:

  * Paramétrico normal
  * Paramétrico t-Student
  * Histórico
  * Monte Carlo
* Rolling VaR y Expected Shortfall (ventana móvil de 252 días)
* Validación mediante violaciones
* VaR con volatilidad móvil

---

## Resultados clave

* Los rendimientos no siguen una distribución normal
* Existe sesgo positivo y curtosis elevada (colas pesadas)
* El VaR tiende a subestimar el riesgo en escenarios extremos
* El Expected Shortfall es una medida más robusta
* Modelos como t-Student y Monte Carlo capturan mejor eventos extremos
* El uso de ventanas móviles permite capturar cambios en la volatilidad

---

## Interpretación personal

A partir del análisis realizado, se pueden destacar las siguientes observaciones:

* El trigo presenta alta volatilidad en el corto plazo, lo que implica que no es un activo adecuado para estrategias especulativas sin una adecuada gestión de riesgo.

* La presencia de colas pesadas indica que los eventos extremos ocurren con mayor frecuencia de lo que asumiría una distribución normal, lo que aumenta la incertidumbre en el comportamiento del precio.

* El sesgo positivo sugiere una mayor probabilidad de movimientos extremos al alza, lo cual podría representar oportunidades en horizontes de largo plazo.

* En términos de medición de riesgo, el Expected Shortfall resulta más confiable que el VaR, ya que captura mejor la magnitud de las pérdidas en escenarios adversos.

* Los modelos dinámicos, como la volatilidad móvil, permiten adaptarse a cambios en las condiciones del mercado, lo cual es relevante para el monitoreo continuo del riesgo.

En conjunto, el trigo puede considerarse un activo con potencial en el largo plazo, pero que requiere una adecuada gestión del riesgo debido a su alta variabilidad y la presencia de eventos extremos.

---

## Conclusiones

No existe un único modelo óptimo para medir el riesgo financiero.

Sin embargo:

* El Expected Shortfall es una medida más adecuada para evaluar riesgo extremo
* Los modelos que consideran colas pesadas ofrecen estimaciones más realistas
* Los enfoques dinámicos permiten capturar cambios en el mercado

Por lo tanto, una combinación de metodologías proporciona una visión más completa y robusta del riesgo.

---

## Tecnologías utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Plotly
* Streamlit
* SciPy

---

## Ejecución

Instalar dependencias:

```bash
pip install streamlit yfinance pandas numpy matplotlib seaborn plotly scipy
```

Ejecutar la aplicación:

```bash
streamlit run app.py
```

---

## Notas

* Se requiere conexión a internet para la descarga de datos
* Proyecto desarrollado con fines académicos
