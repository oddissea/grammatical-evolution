# Algoritmo Memético basado en Evolución Gramatical para Cálculo de Integrales Indefinidas

Este repositorio contiene la implementación de un algoritmo evolutivo basado en evolución gramatical para resolver el problema del cálculo de integrales indefinidas. Transformamos el problema matemático en uno de optimización donde buscamos expresiones que, al derivarlas, aproximen una función objetivo dada.

## Descripción del proyecto

El cálculo de integrales indefinidas constituye un problema fundamental del análisis matemático que requiere determinar una función F(x) que sea la integral de una función dada f(x). Formalmente, buscamos:

F(x) = ∫ f(x) dx

Por el teorema fundamental del cálculo, este problema equivale a encontrar una función F(x) cuya derivada sea igual a f(x):

dF(x)/dx = f(x)

Nuestro algoritmo utiliza evolución gramatical para evolucionar expresiones matemáticas que resuelvan este problema, implementando además mecanismos de búsqueda local y control adaptativo de parámetros para mejorar el rendimiento.

## Características

- **Evolución gramatical**: Implementación completa del algoritmo de evolución gramatical según O'Neill y Ryan (2001)
- **Tres variantes**:
  - **EG-Base**: Implementación básica de evolución gramatical
  - **EG-BL**: Evolución gramatical con mecanismos de búsqueda local
  - **Memético**: Algoritmo memético completo con evolución gramatical, búsqueda local y control adaptativo de parámetros
- **Gramática BNF** para generar expresiones matemáticas que incluyen operaciones aritméticas, funciones trigonométricas, exponenciales y logarítmicas
- **Evaluación numérica** de derivadas para comparar con la función objetivo
- **Manejo de restricciones** para satisfacer condiciones iniciales específicas
- **Visualización de resultados** con comparativas entre soluciones exactas y aproximadas
- **Análisis de rendimiento** mediante los índices TE (Tasa de Éxito), VAMM (Valor de Adaptación Medio del Mejor) y PEX (Promedio de Evaluaciones hasta el Éxito)

## Problemas abordados

El sistema ha sido evaluado en tres problemas específicos:

1. **Problema 1**: f(x) = (1/4)(3x² - 2x + 1), x ∈ [-2,2] y F(0) = -1/4
   - Solución exacta: F(x) = (1/4)(x² + 1)(x - 1)

2. **Problema 2**: f(x) = ln(1 + x) + x/(1+x), x ∈ [0,5] y F(0) = 0
   - Solución exacta: F(x) = x·ln(1 + x)

3. **Problema 3**: f(x) = e^x(sin(x) + cos(x)), x ∈ [-2,2] y F(0) = 0
   - Solución exacta: F(x) = e^x·sin(x)

## Estructura del código

- `main.py`: Script principal que configura los experimentos y ejecuta el algoritmo
- `grammatical_evolution.py`: Implementa el núcleo de evolución gramatical
- `evaluation.py`: Contiene funciones para evaluar las expresiones y calcular derivadas numéricas
- `constraints.py`: Implementa mecanismos para manejar las restricciones
- `local_search.py`: Define los algoritmos de búsqueda local
- `parameter_control.py`: Implementa el control adaptativo de parámetros

## Resultados

El algoritmo ha conseguido tasas de éxito de:
- 20% para el problema 1 con la variante EG-BL
- 100% para el problema 2 con la variante Memética
- 60% para el problema 3 con todas las variantes

Los resultados muestran que la hibridación de técnicas evolutivas con búsqueda local puede ser muy efectiva para problemas de optimización simbólica.

## Requisitos

- Python 3.8+
- NumPy
- Matplotlib
- pandas
- tqdm

## Uso

```bash
# Ejecutar todos los problemas con todas las variantes
python main.py

# Ejecutar un problema específico
python main.py --problem 1

# Ejecutar con una variante específica
python main.py --variant "Memético"

# Modo de prueba rápida
python main.py --test-mode
```

Los resultados se guardan en el directorio `results/` incluyendo gráficos comparativos y archivos CSV con métricas detalladas.

## Referencias

- O'Neill, M., & Ryan, C. (2001). Grammatical evolution. IEEE Transactions on Evolutionary Computation, 5(4), 349-358.
- Carmona Suárez, E. J., & Fernández Galán, S. (2020). Fundamentos de la Computación Evolutiva. Barcelona: Marcombo.

## Autor

Fernando H. Nasser-Eddine López

## Licencia

Este proyecto se distribuye bajo la licencia MIT.