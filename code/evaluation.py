# ************************************************************************
# * evaluation.py
# *
# * Funciones para evaluación de expresiones matemáticas y cálculo
# * de derivadas numéricas en el problema de integrales indefinidas
# *
# * Autor: Fernando H. Nasser-Eddine López
# * Email: fnassered1@alumno.uned.es
# * Versión: 1.0.0
# * Fecha: 09/04/2025
# *
# * Asignatura: Computación Evolutiva
# * Máster en Investigación en Inteligencia Artificial - UNED
# ************************************************************************

import numpy as np
import math
from constraints import ConstraintHandler


class IntegralEvaluator:
    """
    Clase para evaluar la calidad de las soluciones candidatas
    en el problema de cálculo de integrales indefinidas.
    """

    def __init__(self, target_function, interval, initial_condition, h=1e-5):
        """
        Inicializa el evaluador de integrales.

        Args:
            target_function: Función objetivo a integrar
            interval: Tupla con el intervalo [a, b] donde evaluar
            initial_condition: Tupla (x0, F(x0)) con el valor inicial conocido
            h: Paso para el cálculo de la derivada numérica
        """
        self.target_function = target_function
        self.interval = interval
        self.initial_point = initial_condition[0]
        self.initial_value = initial_condition[1]
        self.h = h
        self.constraint_handler = ConstraintHandler()

    # Modificación para el archivo evaluation.py
    # Encuentra la función _evaluate_math_expression y actualízala así:

    @staticmethod
    def _evaluate_math_expression(expression, x_value):
        """
        Evalúa una expresión matemática en x_value con manejo mejorado para prevenir overflows.
        """
        # Reemplazar las funciones matemáticas por sus equivalentes en numpy
        expression = expression.replace('sen', 'np.sin')
        expression = expression.replace('cos', 'np.cos')
        expression = expression.replace('exp', 'np.exp')

        # Reemplazar ln con una función segura que maneja valores problemáticos
        expression = expression.replace('ln', 'safe_log')

        try:
            # Crear un entorno seguro con funciones controladas
            def safe_log(x):
                # Devuelve un valor alto pero no infinito si x <= 0
                if np.isscalar(x):
                    return np.log(x) if x > 0 else 1000.0
                else:
                    # Para arrays, maneja cada elemento
                    result_ = np.zeros_like(x, dtype=float)
                    mask = x > 0
                    result_[mask] = np.log(x[mask])
                    result_[~mask] = 1000.0
                    return result_

            # Función exponencial segura con límites
            def safe_exp(x):
                if np.isscalar(x):
                    # Limitar exponentes muy grandes para prevenir overflow
                    if x > 709:  # ln(máximo valor de float)
                        return 1000.0
                    return np.exp(x)
                else:
                    result_ = np.zeros_like(x, dtype=float)
                    mask = x <= 709
                    result_[mask] = np.exp(x[mask])
                    result_[~mask] = 1000.0
                    return result_

            # Función segura para multiplicación
            def safe_multiply(a, b):
                try:
                    product = a * b
                    # Verificar si el resultado es demasiado grande
                    if np.isscalar(product) and abs(product) > 1e100:
                        return 1000.0 * np.sign(product)
                    return product
                except (TypeError, ValueError):
                    return 1000.0

            # Crear un entorno seguro con funciones controladas
            safe_dict = {
                'x': x_value,
                'np': np,
                'safe_log': safe_log,
                'safe_exp': safe_exp,
                'safe_multiply': safe_multiply
            }

            # Reemplazar exp con la versión segura
            expression = expression.replace('np.exp', 'safe_exp')

            # Evaluar la expresión con manejo de errores
            with np.errstate(all='ignore'):  # Suprimir advertencias de numpy
                result = eval(expression, {"__builtins__": {}}, safe_dict)

            # Comprobar si el resultado es válido
            if np.isnan(result) or np.isinf(result):
                return 1000.0 * (1 if np.isnan(result) else np.sign(result))

            # Limitar valores extremadamente grandes pero no infinitos
            if np.isscalar(result) and abs(result) > 1e100:
                return 1000.0 * np.sign(result)

            return result
        except (ArithmeticError, ValueError, SyntaxError, NameError, TypeError, OverflowError):
            # Manejo ampliado de excepciones
            return 1000.0

    def _numerical_derivative(self, expression, x_value):
        """
        Calcula la derivada numérica de la expresión en x_value con mejor manejo de errores.
        """
        try:
            f_x = self._evaluate_math_expression(expression, x_value)
            f_x_plus_h = self._evaluate_math_expression(expression, x_value + self.h)

            # Si alguno de los valores es muy grande pero no infinito, limitar su magnitud
            max_value = 1e50
            if abs(f_x) > max_value:
                f_x = max_value * np.sign(f_x)
            if abs(f_x_plus_h) > max_value:
                f_x_plus_h = max_value * np.sign(f_x_plus_h)

            # Calcular derivada con protección contra overflow
            with np.errstate(all='ignore'):
                diff = f_x_plus_h - f_x

                # Si la diferencia es demasiado grande, limitar su valor
                if abs(diff) > max_value:
                    diff = max_value * np.sign(diff)

                derivative = diff / self.h

                # Verificar que el resultado es un número válido
                if math.isnan(derivative):
                    return 1000.0

                # Limitar el valor máximo de la derivada
                if abs(derivative) > max_value:
                    return max_value * np.sign(derivative)

                return derivative
        except (ArithmeticError, ValueError, SyntaxError, NameError, TypeError):
            return 1000.0  # Valor alto pero no infinito


    def _check_initial_condition(self, expression):
        """
        Verifica si la expresión satisface la condición inicial.
        Devuelve el error cuadrático.
        """
        value = self._evaluate_math_expression(expression, self.initial_point)
        if math.isinf(value):
            return float('inf')

        return (value - self.initial_value) ** 2

    def evaluate(self, expression, n=100):
        """
        Evalúa la calidad de una expresión como solución al problema de integración.
        Utiliza la fórmula especificada en la Actividad 3 con penalización gradual.
        """
        # Verificar si la expresión satisface la condición inicial
        initial_condition_error = self._check_initial_condition(expression)
        max_error = 1000.0  # Establecer un límite alto pero no infinito para el error

        if math.isinf(initial_condition_error):
            initial_condition_error = max_error

        # Calcular el error en los puntos del intervalo
        a, b = self.interval
        dx = (b - a) / n

        error_sum = 0
        valid_points = 0
        hit_count = 0  # Contador de puntos que cumplen el umbral de error

        for i in range(n + 1):
            x_value = a + i * dx

            # Calcular derivada numérica de la expresión candidata
            derivative = self._numerical_derivative(expression, x_value)

            # Calcular valor de la función objetivo
            target_value = self.target_function(x_value)

            # Si la derivada es muy grande pero no infinita, limitarla
            if abs(derivative) > max_error:
                derivative = max_error * np.sign(derivative)

            # Calcular error absoluto
            abs_error = abs(derivative - target_value)

            # Aplicar ponderación según fórmula de la Actividad 3
            if abs_error <= 0.1:  # U = 10^(-1)
                weight = 1  # K0 = 1
                hit_count += 1  # Incrementar contador de hits
            else:
                weight = 10  # K1 = 10

            # Añadir error ponderado a la suma
            try:
                # Verificar si tenemos valores problemáticos
                if math.isinf(abs_error) or math.isnan(abs_error):
                    abs_error = max_error

                # Limitar el error absoluto a un valor máximo razonable
                if abs_error > max_error:
                    abs_error = max_error

                error_sum += weight * abs_error
                valid_points += 1
            except (ArithmeticError, ValueError):
                # Si ocurre cualquier error numérico, incrementar error_sum con un valor alto
                error_sum += weight * max_error
                valid_points += 1

        # Calcular error promedio (evitar división por cero)
        average_error = error_sum / max(1, valid_points)

        # Ajustar el fitness según el porcentaje de "hits" conseguidos
        hit_percentage = hit_count / (n + 1)
        # Bonificación por alto porcentaje de hits
        if hit_percentage > 0.9:
            average_error *= 0.8  # 20% de bonificación para soluciones con > 90% de hits

        # Aplicar penalización adaptativa para la condición inicial
        return self.constraint_handler.apply_penalty(average_error, initial_condition_error)

    def lex_rank(self, expression, n=50):
        """
        Calcula el ranking lexicográfico de una expresión.
        Primero evalúa si cumple la restricción, luego la calidad de la aproximación.

        Args:
            expression: Expresión a evaluar
            n: Número de puntos de muestreo

        Returns:
            Tupla (rank, value) para ordenación lexicográfica
        """
        initial_condition_error = self._check_initial_condition(expression)
        if math.isinf(initial_condition_error):
            return 1, float('inf')

        # Calcular error de aproximación (similar a evaluate pero sin penalización)
        a, b = self.interval
        dx = (b - a) / n

        error_sum = 0
        for i in range(n + 1):
            x_value = a + i * dx
            derivative = self._numerical_derivative(expression, x_value)
            target_value = self.target_function(x_value)

            if math.isinf(derivative):
                return 1, float('inf')

            abs_error = abs(derivative - target_value)
            if abs_error <= 0.1:
                weight = 1
            else:
                weight = 10

            error_sum += weight * abs_error

        average_error = error_sum / (n + 1)

        return self.constraint_handler.lex_rank(average_error, initial_condition_error)

    def repair_expression(self, expression):
        """
        Intenta reparar una expresión para que cumpla la condición inicial.

        Args:
            expression: Expresión a reparar

        Returns:
            Expresión reparada
        """
        return self.constraint_handler.repair_expression(
            expression,
            self.initial_point,
            self.initial_value,
            self._evaluate_math_expression
        )


# Definición de las funciones objetivo para los tres problemas

def problem1_function(x):
    """
    Problema 1: f(x) = (1/4) * (3x^2 - 2x + 1)
    Intervalo: [-2, 2]
    Condición inicial: F(0) = -1/4
    Solución: F(x) = (1/4) * (x^2 + 1)(x - 1)
    """
    return (1 / 4) * (3 * x ** 2 - 2 * x + 1)


def problem2_function(x):
    """
    Problema 2: f(x) = ln(1 + x) + x/(1 + x)
    Intervalo: [0, 5]
    Condición inicial: F(0) = 0
    Solución: F(x) = x*ln(1 + x)
    """
    return np.log(1 + x) + x / (1 + x)


def problem3_function(x):
    """
    Problema 3: f(x) = e^x * (sin(x) + cos(x))
    Intervalo: [-2, 2]
    Condición inicial: F(0) = 0
    Solución: F(x) = e^x * sin(x)
    """
    return np.exp(x) * (np.sin(x) + np.cos(x))


# Crear evaluadores para cada problema
evaluator1 = IntegralEvaluator(
    target_function=problem1_function,
    interval=[-2, 2],
    initial_condition=(0, -0.25)
)

evaluator2 = IntegralEvaluator(
    target_function=problem2_function,
    interval=[0, 5],
    initial_condition=(0, 0)
)

evaluator3 = IntegralEvaluator(
    target_function=problem3_function,
    interval=[-2, 2],
    initial_condition=(0, 0)
)