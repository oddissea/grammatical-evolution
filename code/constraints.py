# ************************************************************************
# * constraints.py
# *
# * Implementación de mecanismos de manejo de restricciones para
# * evolución gramatical con penalización adaptativa y reparación
# *
# * Autor: Fernando H. Nasser-Eddine López
# * Email: fnassered1@alumno.uned.es
# * Versión: 1.0.0
# * Fecha: 09/04/2025
# *
# * Asignatura: Computación Evolutiva
# * Máster en Investigación en Inteligencia Artificial - UNED
# ************************************************************************

import math


class ConstraintHandler:
    """
    Clase que implementa diferentes estrategias de manejo de restricciones
    para el problema de cálculo de integrales indefinidas.
    """

    def __init__(self, initial_penalty_factor=10.0, adaptation_rate=0.1):
        """
        Inicializa el controlador de restricciones.

        Args:
            initial_penalty_factor: Factor inicial de penalización
            adaptation_rate: Tasa de adaptación del factor de penalización
        """
        self.penalty_factor = initial_penalty_factor
        self.adaptation_rate = adaptation_rate
        self.best_init_error = float('inf')
        self.threshold = 1e-3  # Umbral para considerar satisfecha la restricción


    def apply_penalty(self, fitness_value, init_error):
        """
        Aplica una penalización más suave y escalonada al fitness,
        evitando saltos bruscos que rechacen en exceso a los individuos cercanos a frontera.
        """
        max_penalty = 500.0  # reducir un poco el máximo
        base_penalty = self.penalty_factor

        if init_error < 0.01:
            # Error muy pequeño, apenas penalizamos
            return fitness_value
        elif init_error < 0.1:
            # Penalización pequeña
            penalized = fitness_value + base_penalty * init_error * 0.5
        else:
            # Error grande, penalización mayor pero no infinita
            penalized = fitness_value + base_penalty * init_error * 2.0

        # Limitar por arriba
        if penalized > max_penalty:
            penalized = max_penalty

        # Ajustar factor adaptativamente
        self._adapt_penalty_factor(init_error)
        return penalized

    def _adapt_penalty_factor(self, current_error):
        """
        Adapta el factor de penalización según el error actual.
        Si el error disminuye, reduce la penalización; si aumenta, la incrementa.
        """
        if current_error < self.best_init_error:
            # Si el error ha mejorado, reducimos la penalización
            self.penalty_factor = max(1.0, self.penalty_factor * (1 - self.adaptation_rate))
            self.best_init_error = current_error
        elif current_error > self.best_init_error:
            # Si el error ha empeorado, aumentamos la penalización
            self.penalty_factor = min(100.0, self.penalty_factor * (1 + self.adaptation_rate))

    def repair_expression(self, expression, x0, target_value, safe_eval_func):
        """
        Intenta reparar una expresión para que cumpla la condición inicial.

        Args:
            expression: Expresión a reparar (string)
            x0: Punto donde se debe cumplir la condición inicial
            target_value: Valor que debe tomar la expresión en x0
            safe_eval_func: Función para evaluar expresiones de manera segura

        Returns:
            Expresión reparada (string)
        """
        try:
            # Evaluar la expresión en x0
            current_value = safe_eval_func(expression, x0)

            if math.isinf(current_value) or math.isnan(current_value):
                return expression

            # Calcular el término de corrección
            correction = target_value - current_value

            if abs(correction) < self.threshold:
                # Si ya cumple la restricción, no es necesario reparar
                return expression

            # Aplicar una corrección parcial (50% del error)
            partial_correction = 0.5 * correction

            # Crear la expresión corregida
            if partial_correction >= 0:
                repaired_expr = f"({expression} + {partial_correction})"
            else:
                repaired_expr = f"({expression} - {abs(partial_correction)})"

            return repaired_expr
        except (ValueError, ArithmeticError, TypeError, NameError):
            # Manejo específico de excepciones comunes en evaluación de expresiones
            return expression

    def lex_rank(self, fitness_value, init_error):
        """
        Implementa un ranking lexicográfico para la selección.
        Primero considera si se cumple la restricción, luego el valor de fitness.

        Args:
            fitness_value: Valor de fitness de la función objetivo
            init_error: Error en la condición inicial

        Returns:
            Tupla (rank, value) para ordenación lexicográfica
        """
        if init_error <= self.threshold:
            # Si cumple la restricción, el primer valor es 0 y el segundo es el fitness
            return 0, fitness_value
        else:
            # Si no cumple la restricción, el primer valor es 1 y el segundo es el error
            return 1, init_error