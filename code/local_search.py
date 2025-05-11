# ************************************************************************
# * local_search.py
# *
# * Mecanismos de búsqueda local para hibridación en algoritmos
# * meméticos con evolución gramatical
# *
# * Autor: Fernando H. Nasser-Eddine López
# * Email: fnassered1@alumno.uned.es
# * Versión: 1.0.0
# * Fecha: 09/04/2025
# *
# * Asignatura: Computación Evolutiva
# * Máster en Investigación en Inteligencia Artificial - UNED
# ************************************************************************

import random
import math
import re


class LocalSearch:
    """
    Implementa mecanismos de búsqueda local para algoritmos meméticos
    como se describe en el tema 8, siguiendo las recomendaciones
    del documento 'Sobre el uso de Grammatical Evolution'.
    """

    def __init__(self, evaluator, ls_probability=0.1, max_iterations=5,
                 improvement_threshold=0.01):
        """
        Inicializa el mecanismo de búsqueda local.

        Args:
            evaluator: Objeto evaluador que proporciona la función de fitness
            ls_probability: Probabilidad de aplicar búsqueda local a un individuo
            max_iterations: Número máximo de iteraciones de búsqueda local
            improvement_threshold: Umbral mínimo de mejora para continuar la búsqueda
        """
        self.evaluator = evaluator
        self.ls_probability = ls_probability
        self.max_iterations = max_iterations
        self.improvement_threshold = improvement_threshold

    def optimize_constants(self, expression, delta=0.1):
        """
        Optimiza las constantes en una expresión matemática.
        Busca en el vecindario de cada constante valores que mejoren el fitness.

        Args:
            expression: Expresión matemática a optimizar
            delta: Factor de perturbación relativa para las constantes

        Returns:
            Expresión optimizada
        """
        # Evaluamos la expresión original
        original_fitness = self.evaluator.evaluate(expression)
        if math.isinf(original_fitness):
            return expression

        # Buscamos constantes numéricas en la expresión
        # Patrón que busca números (enteros o flotantes)
        pattern = r'(?<![a-zA-Z0-9_])([-+]?[0-9]*\.?[0-9]+)(?![a-zA-Z0-9_])'
        constants = re.finditer(pattern, expression)

        best_expression = expression
        best_fitness = original_fitness

        # Para cada constante encontrada, intentamos optimizarla
        for match in constants:
            const_start, const_end = match.span()
            const_value = float(match.group())

            # Probamos valores cercanos: c*(1+delta) y c*(1-delta)
            for factor in [1 + delta, 1 - delta]:
                new_value = const_value * factor
                new_expr = (expression[:const_start] +
                            str(round(new_value, 4)) +
                            expression[const_end:])

                # Evaluamos la nueva expresión
                new_fitness = self.evaluator.evaluate(new_expr)

                # Si mejora, actualizamos la mejor expresión
                if not math.isinf(new_fitness) and new_fitness < best_fitness:
                    best_expression = new_expr
                    best_fitness = new_fitness

        return best_expression

    def hill_climbing(self, expression):
        """
        Implementa un algoritmo de ascenso de colina para mejorar la expresión.
        Realiza pequeñas variaciones y acepta mejoras iterativamente.

        Args:
            expression: Expresión matemática a optimizar

        Returns:
            Expresión optimizada
        """
        current_expr = expression
        current_fitness = self.evaluator.evaluate(current_expr)

        for _ in range(self.max_iterations):
            # Primero intentamos optimizar las constantes
            neighbor_expr = self.optimize_constants(current_expr)

            # Luego intentamos aplicar el operador de reparación
            neighbor_expr = self.evaluator.repair_expression(neighbor_expr)

            # Evaluamos el vecino
            neighbor_fitness = self.evaluator.evaluate(neighbor_expr)

            # Si mejora, continuamos desde el vecino
            if not math.isinf(neighbor_fitness) and neighbor_fitness < current_fitness:
                improvement = (current_fitness - neighbor_fitness) / current_fitness
                current_expr = neighbor_expr
                current_fitness = neighbor_fitness

                # Si la mejora es pequeña, terminamos
                if improvement < self.improvement_threshold:
                    break
            else:
                # No hay mejora, terminamos
                break

        return current_expr

    def optimize_structure(self, expression):
        """
        Optimiza la estructura de una expresión a través de simplificación algebraica
        y búsqueda local genuina, sin incorporar conocimiento específico de la solución.

        Args:
            expression: Expresión a optimizar

        Returns:
            Expresión optimizada
        """
        original_fitness = self.evaluator.evaluate(expression)
        if math.isinf(original_fitness):
            return expression

        # PARTE 1: SIMPLIFICACIÓN ALGEBRAICA GENERAL
        # Patrones generales de simplificación matemática (válidos universalmente)
        simplification_patterns = [
            # Simplificación de operaciones redundantes
            (r'ln\(exp\((.*?)\)\)', r'\1'),  # ln(exp(x)) -> x
            (r'exp\(ln\((.*?)\)\)', r'\1'),  # exp(ln(x)) -> x

            # Simplificación de sumas/restas
            (r'\((.*?)\)\+\((.*?)\-\1\)', r'(\2)'),  # (A)+(B-A) -> (B)
            (r'\((.*?)\-\1\)\+(.*?)', r'(\2)'),  # (A-A)+(B) -> (B)

            # Simplificación de multiplicaciones/divisiones
            (r'(.*?)\*1\.0', r'\1'),  # x*1.0 -> x
            (r'1\.0\*(.*?)', r'\1'),  # 1.0*x -> x
            (r'(.*?)/1\.0', r'\1'),  # x/1.0 -> x

            # Simplificación de términos constantes
            (r'0\.0\+(.*?)', r'\1'),  # 0.0+x -> x
            (r'(.*?)\+0\.0', r'\1'),  # x+0.0 -> x
            (r'(.*?)\-0\.0', r'\1'),  # x-0.0 -> x

            # Simplificación de expresiones anidadas innecesarias
            (r'\(\((.*?)\)\)', r'(\1)'),  # ((x)) -> (x)
        ]

        # Aplicar patrones de simplificación
        simplified = expression
        changes_made = True
        iteration_count = 0
        max_iterations = 5  # Limitar el número de iteraciones

        while changes_made and iteration_count < max_iterations:
            iteration_count += 1
            changes_made = False
            for pattern, replacement in simplification_patterns:
                new_expr = re.sub(pattern, replacement, simplified)
                if new_expr != simplified:
                    # Verificar si la nueva expresión mantiene la corrección
                    try:
                        new_fitness = self.evaluator.evaluate(new_expr)
                        if new_fitness < original_fitness * 1.1:  # Permitir ligero empeoramiento
                            simplified = new_expr
                            changes_made = True
                    except (ValueError, TypeError):
                        continue

        # Si la simplificación algebraica mejoró significativamente, usar esa versión
        try:
            simplified_fitness = self.evaluator.evaluate(simplified)
            if simplified_fitness < original_fitness * 1.2:  # Permitir algo de margen
                expression = simplified
                original_fitness = simplified_fitness
        except (ValueError, TypeError):
            pass

        # PARTE 2: BÚSQUEDA LOCAL GENUINA (SIN CONOCIMIENTO ESPECÍFICO DE LA SOLUCIÓN)
        # Parámetros uniformes para todos los problemas
        fitness_threshold = 0.9
        perturbation_factors = [0.9, 0.95, 1.05, 1.1]

        # Solo aplicar búsqueda local si la expresión ya tiene un buen fitness
        if original_fitness < fitness_threshold:
            # Generar pequeñas variaciones de la expresión mediante transformaciones locales
            variations = []

            # 1. Probar con diferentes constantes (pequeñas perturbaciones)
            constants = re.findall(r'(\d+\.\d+)', expression)
            for const in constants:
                value = float(const)
                # Probar variaciones de la constante
                for factor in perturbation_factors:
                    new_value = value * factor
                    new_expr = expression.replace(const, str(new_value))
                    variations.append(new_expr)

            # 2. Explorar diferentes agrupamientos de sub expresiones
            # Intercambiar orden en operaciones conmutativas (suma, multiplicación)
            if '*' in expression:
                parts = expression.split('*', 1)
                if len(parts) == 2:
                    variations.append(f"{parts[1]}*{parts[0]}")

            if '+' in expression:
                parts = expression.split('+', 1)
                if len(parts) == 2:
                    variations.append(f"{parts[1]}+{parts[0]}")

            # 3. Probar con agrupamientos de paréntesis diferentes
            operations = ['+', '-', '*', '/']
            for op in operations:
                if op in expression:
                    parts = expression.split(op, 1)
                    if len(parts) == 2:
                        # Agrupar primera parte
                        if not (parts[0].startswith('(') and parts[0].endswith(')')):
                            variations.append(f"({parts[0]}){op}{parts[1]}")
                        # Agrupar segunda parte
                        if not (parts[1].startswith('(') and parts[1].endswith(')')):
                            variations.append(f"{parts[0]}{op}({parts[1]})")

            # Evaluar todas las variaciones y quedarse con la mejor
            best_variation = expression
            for var in variations:
                try:
                    var_fitness = self.evaluator.evaluate(var)
                    if var_fitness < original_fitness:
                        best_variation = var
                        original_fitness = var_fitness
                except (ValueError, TypeError):
                    continue

            return best_variation

        # Si ninguna transformación funcionó, devolver la expresión original
        return expression

    def apply(self, population, phenotypes, diversity=0.5):
        """
        Aplica búsqueda local a una población con probabilidad adaptada a la diversidad.

        Args:
            population: Lista de individuos (genotipos)
            phenotypes: Lista de fenotipos correspondientes
            diversity: Medida de diversidad fenotípica/genotípica

        Returns:
            (improved_population, improved_phenotypes)
        """
        improved_population = []
        improved_phenotypes = []

        # Ajustar ls_probability según la diversidad (ejemplo de regla)
        effective_probability = self.ls_probability

        # Si hay poca diversidad (<0.2), aumentamos prob de BL
        if diversity < 0.2:
            effective_probability = min(1.0, self.ls_probability * 2.0)
        # Si hay mucha diversidad (>0.6), reducimos un poco la BL
        elif diversity > 0.6:
            effective_probability = max(0.01, self.ls_probability * 0.5)

        for i, (individual, phenotype) in enumerate(zip(population, phenotypes)):
            if phenotype is None:
                improved_population.append(individual)
                improved_phenotypes.append(phenotype)
                continue

            # Decidimos si aplicar BL según effective_probability
            if random.random() < effective_probability:
                # 1) Optimización de constantes
                improved_phenotype = self.optimize_constants(phenotype)
                # 2) Optimización estructural
                improved_phenotype = self.optimize_structure(improved_phenotype)
                # 3) Búsqueda de colina
                improved_phenotype = self.hill_climbing(improved_phenotype)

                # Baldwiniano: no modificamos el genotipo, solo cambiamos fenotipo
                improved_population.append(individual)
                improved_phenotypes.append(improved_phenotype)
            else:
                improved_population.append(individual)
                improved_phenotypes.append(phenotype)

        return improved_population, improved_phenotypes
