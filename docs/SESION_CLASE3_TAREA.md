# Guía de Sesión y Asignación Clase 3: Red Neuronal Simple y Funciones de Activación

Documento de continuidad de sesión para la Clase 3 del curso de Introducción a las Redes Neuronales.

---

## 1. Análisis de la Sesión y Material de Clase

Basado en el registro fotográfico de la sesión presencial:

### A. Diapositiva: Práctica `1_04_red_neuronal_simple` (3 Puntos)
* **Duración estimada:** 20 min.
* **Puntaje total:** 3 puntos distribuidos equitativamente (1 punto por TODO).
* **Objetivos específicos:**
  1. **TODO 1 (1 pt):** Implementar la función de activación sigmoide utilizando `np.exp(-h)`.
     * *Alerta de error típico:* Escribir `np.exp(h)` sin el signo menos produce una salida errónea de `0.5671`. Con `-h`, la salida esperada es exactamente **`0.4329`**.
  2. **TODO 2 (1 pt):** Implementar la clase `Neurona` con sus métodos `combinacion_lineal` y `forward`.
     * *Pista:* Utilizar `np.dot(self.W, X) + self.b` para la suma ponderada.
  3. **TODO 3 (1 pt):** Implementar la activación por tangente hiperbólica (`tanh`), mostrar sus salidas y compararla conceptual y numéricamente con la sigmoide.
     * *Comprobación numérica:* Para $h = -0.27$, la salida de `tanh(-0.27)` debe ser exactamente **`-0.2636`**.

### B. Pizarra Docente: Análisis Teórico de Funciones de Activación
La pizarra complementa el fundamento matemático de las funciones no lineales y sus derivadas:

1. **Función Sigmoide:**
   $$\sigma(h) = \frac{1}{1 + e^{-h}}$$
   * Evaluación de puntos clave:
     * $h = -3 \implies \sigma(-3) \approx 0.0474$ (en la pizarra aproximado a $0.0452$)
     * $h = -1 \implies \sigma(-1) \approx 0.2689$
     * $h = 0 \implies \sigma(0) = \frac{1}{1 + e^0} = \frac{1}{2} = 0.5$
     * $h = 1 \implies \sigma(1) \approx 0.7311$
     * $h = 3 \implies \sigma(3) \approx 0.9526$

2. **Derivada de la Sigmoide:**
   $$\sigma'(h) = \sigma(h) \cdot \big(1 - \sigma(h)\big)$$
   * Cálculo en los puntos analizados:
     * $\sigma'(3) = 0.9526 \cdot (1 - 0.9526) \approx 0.0452$
     * $\sigma'(1) = 0.7311 \cdot (1 - 0.7311) \approx 0.1966$
     * $\sigma'(0) = 0.5 \cdot (1 - 0.5) = 0.25$ (*máximo gradiente posible*)

3. **Función ReLU (Rectified Linear Unit):**
   $$f(h) = \max(0, h)$$
   * Se ilustra en la pizarra su linealidad para $h > 0$ y su capacidad para resolver el desvanecimiento del gradiente frente a funciones saturables como la sigmoide.

---

## 2. Solución Completa del Cuaderno `1_04_red_neuronal_simple.ipynb`

### TODO 1: Función Sigmoide (1 punto)
```python
def sigmoide(h):
    return 1 / (1 + np.exp(-h))
```

### TODO 2: Clase Neurona (1 punto)
```python
class Neurona:
    def __init__(self, W, b, activacion):
        self.W = W
        self.b = b
        self.activacion = activacion

    def combinacion_lineal(self, X):
        return np.dot(self.W, X) + self.b

    def forward(self, X):
        h = self.combinacion_lineal(X)
        return self.activacion(h)
```

### Comprobación Básica:
```python
inputs = np.array([0.7, -0.3])
pesos = np.array([0.1, 0.8])
sesgo = -0.1

clasificador_log = Neurona(pesos, sesgo, sigmoide)
output = clasificador_log.forward(inputs)
print(f'Output: {output:.4f}')  # Imprime: 0.4329
```

### TODO 3: Tangente Hiperbólica y Comparación (1 punto)
```python
def tangente_hiperbolica(h):
    return np.tanh(h)

clasificador_tanh = Neurona(pesos, sesgo, tangente_hiperbolica)
output_tanh = clasificador_tanh.forward(inputs)
print(f'Output tanh: {output_tanh:.4f}')  # Imprime: -0.2636
```

### Cuadro Comparativo: Sigmoide vs. Tanh

| Criterio | Función Sigmoide ($\sigma$) | Tangente Hiperbólica ($\tanh$) |
| :--- | :--- | :--- |
| **Fórmula** | $\sigma(h) = \frac{1}{1 + e^{-h}}$ | $\tanh(h) = \frac{e^h - e^{-h}}{e^h + e^{-h}}$ |
| **Rango de salida** | $(0, 1)$ | $(-1, 1)$ |
| **Salida en $h=0$** | $0.5$ | $0.0$ (*centrada en cero*) |
| **Comportamiento $h \to -\infty$** | Tiende a $0$ | Tiende a $-1$ |
| **Comportamiento $h \to +\infty$** | Tiende a $1$ | Tiende a $1$ |
| **Fórmula de la derivada** | $\sigma'(h) = \sigma(h)(1 - \sigma(h))$ | $\tanh'(h) = 1 - \tanh^2(h)$ |
| **Derivada máxima ($h=0$)** | $0.25$ | $1.0$ (gradiente 4 veces mayor) |
| **Caso de uso principal** | Capas de salida en clasificación binaria (probabilidad) | Capas ocultas para mitigar el sesgo unidireccional y desvanecimiento de gradiente |

---

## 3. Enlace para Google Colab

El cuaderno resuelto puede abrirse y ejecutarse directamente en la nube:
👉 **[Abrir 1_04_red_neuronal_simple.ipynb en Google Colab](https://colab.research.google.com/github/pibenitezq126/redes-neuronales-practicas/blob/main/notebooks/1_04_red_neuronal_simple.ipynb)**
