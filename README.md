# Proyecto_MCF_1_MCR

De: 

García Rodríguez Marco Antonio

Hernandez Alcantara Cristina Geraldine

Mendoza Aragón Rogelio

Moreno Ventura Miguel Angel


Para este primer proyecto elegimos al petróleo como activo a analizar, esto principalmente por su relevancia en el mercado actual, además de que es un activo que tiene muchas fluctuaciones (es muy volátil). 
Vamos a usar medidas de riesgo como el VaR histórico y paramétrico (bajo diferentes distribuciones como la normal o t-student) y el Expected Shortfall con la finalidad de compararlas para conocer el riesgo real de nuestro activo.

Viendo la gráfica con la distribución de los rendimientos tal vez podríamos decir que tiene un poco forma de distribución normal, pero al realizar el test de normalidad vemos que rechazamos esta hipótesis, pues tenemos una curtosis κ= 12.2449 > 3 (indica una distribución leptucurtica con menor concentración en la mediana, y colas pesadas), además el sesgo= -0.4724 < 0 (indica que la cola izquierda es más larga y más pesada que la derecha).

En la parte del inciso c) calculamos el VaR y ES para nuestros datos con diferentes intervalos de comfianza (α = 0.95, 0.975, y 0.99) y bajo la asunción de distribuciones normal, t-student, además bajo una aproximación hisórica y Monte Carlo.
Viendo la tabla de resultados notamos que entre mayor sea nuestro α, mayor será el VaR, esto se debe a que ampliamos la confianza de la máxima pérdida esperada en los retornos de nuestro activo; también aumenta el ES, pues este se vuelve más sensible a los "riesgos extremos" (las colas de nuestra distribución) por lo que también aumenta el promedio de estas máximas pérdidas esperadas.
Por ejemplo, cuando α= 0.95, el VaR Normal=-3.74%, esto nos dice que en el 95% de los días se espera que la pérdida no sea mayor al 3.74%, mientras que el ES Normal= -4.70%, lo que significa que en el 5% de los días en que la pérdida es mayor al VaR, en promedio tendrémos una pérdida del 4.70%.

Para los incisos d, e y f, aplicamos ventanas móviles de 252 días con el fin de evaluar dinámicamente el riesgo y realizar el backtesting de nuestros modelos. Al analizar las violaciones, observamos que mientras el modelo normal al 95% presenta una calibración aceptable con un 5.26% de excepciones, al nivel del 99% el número de fallos (2.09%) supera por más del doble la tolerancia esperada, lo que nos indica una clara subestimación del riesgo extremo y la presencia de riesgo de modelo. Incluso cuando intentamos corregir esto ajustando el VaR por volatilidad móvil en el inciso f, encontramos que las violaciones al 99% siguen siendo elevadas (2.14%), lo que nos confirma que la asimetría y las colas pesadas de nuestro activo requieren modelos más robustos que la simple distribución normal para capturar los eventos de estrés rea

