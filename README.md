# Prácticas y Tareas: Curso de Redes Neuronales

Repositorio personal para el desarrollo de prácticas, tareas y soluciones del curso de Introducción a las Redes Neuronales.

---

## 📌 Guía de Tarea - Clase 2 (10 Puntos)

Toda la explicación teórica, fórmulas y soluciones paso a paso de los cuadernos evaluados están documentadas en:
👉 **[`docs/SESION_CLASE2_TAREA.md`](docs/SESION_CLASE2_TAREA.md)**

* **1_01_mcculloch_pitts.ipynb** (3 pts): Implementación de la neurona de McCulloch-Pitts con entradas excitatorias, inhibitorias y umbral.
* **1_03_1_perceptron_simple.ipynb** (3 pts): Combinación lineal manual (sin `np.dot`), función escalón y pesos de compuerta OR.
* **1_03_2_regla_de_aprendizaje.ipynb** (4 pts): Inicialización normal, pérdida SAE, regla de actualización de Rosenblatt y calibración de épocas (20-50).

---

## 🚀 Cómo continuar en cualquier equipo

### 1. Clonar el repositorio
```bash
git clone https://github.com/pibenitezq126/redes-neuronales-practicas.git
cd redes-neuronales-practicas
```

### 2. Crear y activar el entorno virtual

**En Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**En Linux o macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar librerías y registrar el Kernel
```bash
pip install -r requirements.txt
python -m ipykernel install --user --name nni --display-name "Python (nni)"
```

### 4. Abrir los cuadernos
```bash
jupyter-lab notebooks/
```
*(O abre la carpeta en VS Code y selecciona el kernel `Python (nni)`)*.

---

## Estructura del Repositorio

| Carpeta / archivo | Qué contiene |
|---|---|
| [`docs/SESION_CLASE2_TAREA.md`](docs/SESION_CLASE2_TAREA.md) | **Guía completa y soluciones de la asignación Clase 2** |
| [`docs/GUIA_PASO_A_PASO_CLASE1.md`](docs/GUIA_PASO_A_PASO_CLASE1.md) | Clase 1 desde cero: instalación paso a paso y explicación teórica |
| [`docs/02_Principios.md`](docs/02_Principios.md) | Módulo 2: McCulloch-Pitts y perceptrón |
| [`docs/practica_02.py`](docs/practica_02.py) | Práctica en Python puro: McCulloch-Pitts y perceptrón (AND, OR, XOR) |
| [`notebooks/`](notebooks/) | Cuadernos Jupyter con los ejercicios a resolver |

---

## Créditos y Referencias

Basado en el material docente del **Dr. J. Irving Vasquez** ([jivg.org](https://jivg.org)):
- Cuadernos base: [irvingvasquez/cv2course_intro_nn](https://github.com/irvingvasquez/cv2course_intro_nn) (CC BY 4.0).
- Libro de referencia: [Redes Neuronales](https://irvingvasquez.github.io/libro_redes_neuronales/) (CC BY-NC 4.0).

