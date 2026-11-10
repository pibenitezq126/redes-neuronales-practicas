# Guía de Sesión y Asignación Clase 2: El Perceptrón

Documento de continuidad de sesión para retomar el trabajo en cualquier equipo o entorno.

---

## 1. Resumen de la Asignación (Clase 2)

Basado en las diapositivas de la Clase 2 (slides 19, 20 y 21):

* **Práctica exploratoria (sin calificación directa):**
  * `docs/practica_02.py`: Ejecutar con `python docs/practica_02.py`. Demuestra la convergencia en compuertas linealmente separables (AND y OR en 4 épocas) y la no convergencia en XOR (tras 100 épocas).
  * `notebooks/1_02_perceptron_AND.ipynb`: Ajuste manual de pesos para compuerta AND.
* **Tarea Calificada Obligatoria (10 puntos en total):**
  1. `notebooks/1_01_mcculloch_pitts.ipynb` (**3 puntos**).
  2. `notebooks/1_03_1_perceptron_simple.ipynb` (**3 puntos**).
  3. `notebooks/1_03_2_regla_de_aprendizaje.ipynb` (**4 puntos**).
* **Lectura requerida:** Capítulo *"Neurona artificial moderna"* del libro de referencia (Dr. J. Irving Vasquez).

---

## 2. Preparación Rápida en un Nuevo Equipo

Al clonar este repositorio en una nueva máquina, ejecutar los siguientes comandos en PowerShell o terminal:

```bash
# 1. Crear entorno virtual aislado con Python 3.11 o 3.12
python -m venv .venv

# 2. Activar entorno virtual
# En Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# En Linux/macOS:
# source .venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Registrar kernel 'nni' para Jupyter
python -m ipykernel install --user --name nni --display-name "Python (nni)"

# 5. Iniciar JupyterLab
jupyter-lab notebooks/
```

---

## 3. Guía de Solución Paso a Paso de los Notebooks

### Cuaderno 1: `notebooks/1_01_mcculloch_pitts.ipynb` (3 puntos)

* **Fundamento Teórico:**
  * Si $\exists \iota_i = 1$ con $\iota_i \in I$ (inhibición activa) $\implies C(E, I) = 0$.
  * Si $\sum e_i \ge u$ con $e_i \in E$ y no hay inhibición $\implies C(E, I) = 1$.
  * En otro caso $\implies C(E, I) = 0$.
  * *Restricción estricta del notebook:* No usar funciones predefinidas de NumPy.

* **TODO 1 (2 puntos) - Función `neuronaMyP`:**
```python
def neuronaMyP(E, I, u):
    # Regla 1: Si alguna entrada inhibitoria está activa (1), salida = 0
    for inh in I:
        if inh == 1:
            return 0
    # Regla 2: Suma de excitatorias mayor o igual al umbral u
    if sum(E) >= u:
        return 1
    # Regla 3: En cualquier otro caso
    return 0
```

* **TODO 2 (1 punto) - Comprobación con vectores arbitrarios:**
```python
# Comprobación de compuertas lógicas estándar
entradas = [(0,0), (0,1), (1,0), (1,1)]

# AND (excitatorias con umbral 2)
print("AND:", [neuronaMyP([x1, x2], [0], 2) for x1, x2 in entradas])

# OR (excitatorias con umbral 1)
print("OR:", [neuronaMyP([x1, x2], [0], 1) for x1, x2 in entradas])

# NOT (entrada entra como inhibitoria, umbral 0)
print("NOT:", [neuronaMyP([0], [x], 0) for x in (0, 1)])
```

---

### Cuaderno Auxiliar: `notebooks/1_02_perceptron_AND.ipynb`

* **TODO - Ajuste de pesos manuales:**
```python
weight1 = 1.0
weight2 = 1.0
bias = -1.5
```
*(Da como salida "Bien! Acertaste todas")*.

---

### Cuaderno 2: `notebooks/1_03_1_perceptron_simple.ipynb` (3 puntos)

* **TODO 1 (1 punto) - `combinacion_lineal` sin `np.dot()`:**
```python
def combinacion_lineal(W, X, b):
    assert len(W) == len(X), "W y X deben tener la misma dimensión"
    suma = sum(wi * xi for wi, xi in zip(W, X)) + b
    return suma
```

* **TODO 2 (1 punto) - Función de activación `escalon`:**
```python
def escalon(h, a=0):
    if isinstance(h, np.ndarray):
        return np.where(h >= a, 1, 0)
    return 1 if h >= a else 0
```

* **Verificación de prueba (Slide 20):**
  * `inputs = [0.7, -0.3]`, `weights = [0.1, 0.8]`, `bias = -0.1`
  * Cálculo: $h = 0.7(0.1) + (-0.3)(0.8) - 0.1 = -0.27$
  * Como $h < 0$, la salida es `0`.

* **TODO 3 (1 punto) - Pesos para compuerta OR por prueba y error:**
```python
W_or = np.array([1.0, 1.0])
b_or = -0.5

X_or = [np.array([0, 0]), np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
for x in X_or:
    print(f"{x} -> {perceptron(W_or, x, b_or, escalon)}")
# Resultados esperados: 0, 1, 1, 1
```

---

### Cuaderno 3: `notebooks/1_03_2_regla_de_aprendizaje.ipynb` (4 puntos)

* **TODO 1 (1 punto) - Inicialización aleatoria con distribución normal:**
```python
clasificador.W = np.random.normal(loc=0, scale=1, size=(2,))
clasificador.b = np.random.normal(loc=0, scale=1)
```

* **TODO 2 (1 punto) - Función de pérdida SAE (Suma de Errores Absolutos):**
$$\mathcal{L} = \sum |y - \hat{y}|$$
```python
def perdidaSAE(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))
```

* **TODO 3 (1 punto) - Implementación de la regla de aprendizaje de Rosenblatt:**
$$W \leftarrow W + \eta \cdot (y - \hat{y}) \cdot x$$
$$b \leftarrow b + \eta \cdot (y - \hat{y})$$
```python
class ReglaDeAprendizaje:
    def __init__(self, tasa_aprendizaje=0.1):
        self.tasa_aprendizaje = tasa_aprendizaje

    def actualizar(self, perceptron, X, y):
        for xi, target in zip(X.values, y.values):
            output = perceptron.inferencia(xi)
            error = target - output
            perceptron.W = perceptron.W + self.tasa_aprendizaje * error * xi
            perceptron.b = perceptron.b + self.tasa_aprendizaje * error
```

* **TODO 4 (1-2 puntos) - Ajuste de hiperparámetros:**
  * Indicación exacta de la Diapositiva 20: *"sube las épocas a 20-50 y mira cómo baja la pérdida"*.
```python
tasa_aprendizaje = 0.1
n_epocas = 30  # Rango óptimo entre 20 y 50 épocas
```
  * Resultado: Se observa cómo el error desciende progresivamente en la gráfica de pérdida y la recta de decisión (`DecisionBoundaryDisplay`) separa adecuadamente las dos clases de puntos.
