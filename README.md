# Proyecto_MCF_1_MCR
Para este primer proyecto elegimos al petróleo como activo a analizar, esto principalmente por su relevancia en el mercado, además de que es un activo que tiene muchas fluctuaciones (es muy volátil). 
Vamos a usar medidas de riesgo como el VaR histórico y paramétrico (bajo diferentes distribuciones como la normal o t-student) y el Expected Shortfall con la finalidad de compararlas para conocer el riesgo real de nuestro activo.

Viendo la gráfica con la distribución de los rendimientos tal vez podríamos decir que tiene un poco forma de distribución normal, pero al realizar el test de normalidad vemos que rechazamos esta hipótesis, pues tenemos una curtosis κ= 12.2449 > 3 (indica una distribución leptucurtica con menor concentración en la mediana, y colas pesadas), además el sesgo= -0.4724 < 0 (indica que la cola izquierda es más larga y m´as pesada que la derecha).
