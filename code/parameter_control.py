# ************************************************************************
# * parameter_control.py
# *
# * Control adaptativo de parámetros para evolución gramatical
# * y algoritmos meméticos
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


class ParameterController:
    """
    Implementa mecanismos de control adaptativo de parámetros para
    algoritmos evolutivos según lo descrito en el tema 12.
    """

    def __init__(self, init_crossover_rate=0.9, init_mutation_rate=0.1,
                 min_mutation_rate=0.01, max_mutation_rate=0.3,
                 adaptation_rate=0.1, stagnation_threshold=5):
        """
        Inicializa el controlador de parámetros.

        Args:
            init_crossover_rate: Tasa inicial de cruce
            init_mutation_rate: Tasa inicial de mutación
            min_mutation_rate: Tasa mínima de mutación permitida
            max_mutation_rate: Tasa máxima de mutación permitida
            adaptation_rate: Velocidad de adaptación de parámetros
            stagnation_threshold: Generaciones sin mejora para considerar estancamiento
        """
        self.crossover_rate = init_crossover_rate
        self.mutation_rate = init_mutation_rate
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.adaptation_rate = adaptation_rate
        self.stagnation_threshold = stagnation_threshold

        self.best_fitness_history = []
        self.stagnation_counter = 0

    def update_parameters(self, current_best_fitness, population_diversity):
        """
        Actualiza los parámetros basándose en el rendimiento actual y la diversidad.

        Args:
            current_best_fitness: Mejor valor de fitness en la generación actual
            population_diversity: Medida de diversidad de la población
        """
        # Actualizar historial de fitness
        if self.best_fitness_history:
            if current_best_fitness < min(self.best_fitness_history):
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1

        self.best_fitness_history.append(current_best_fitness)

        # Comprobar estancamiento
        if self.stagnation_counter >= self.stagnation_threshold:
            # Estamos estancados: aumentar exploración
            self._increase_exploration(population_diversity)
            self.stagnation_counter = 0
        else:
            # No estamos estancados: ajuste normal basado en diversidad
            self._adaptive_adjustment(population_diversity)

    def _increase_exploration(self, diversity):
        """
        Aumenta la exploración cuando se detecta estancamiento.
        """
        # Aumentar la tasa de mutación para favorecer la exploración
        self.mutation_rate = min(self.max_mutation_rate,
                                 self.mutation_rate * (1 + self.adaptation_rate))

        # Reducir ligeramente la tasa de cruce si la diversidad es baja
        if diversity < 0.3:  # Umbral arbitrario para diversidad baja
            self.crossover_rate = max(0.7, self.crossover_rate * (1 - self.adaptation_rate / 2))

    def _adaptive_adjustment(self, diversity):
        """
        Ajuste adaptativo normal basado en la diversidad de la población.
        """
        if diversity < 0.2:  # Diversidad muy baja
            # Aumentar mutación, reducir cruce
            self.mutation_rate = min(self.max_mutation_rate,
                                     self.mutation_rate * (1 + self.adaptation_rate))
            self.crossover_rate = max(0.7, self.crossover_rate * (1 - self.adaptation_rate / 4))

        elif diversity > 0.6:  # Diversidad alta
            # Reducir mutación, aumentar cruce
            self.mutation_rate = max(self.min_mutation_rate,
                                     self.mutation_rate * (1 - self.adaptation_rate))
            self.crossover_rate = min(0.95, self.crossover_rate * (1 + self.adaptation_rate / 4))

        # Para diversidad en rango medio (0.2-0.6) mantenemos un equilibrio

    def get_parameters(self):
        """
        Devuelve los parámetros actuales.

        Returns:
            Diccionario con los valores actuales de los parámetros
        """
        return {
            'crossover_rate': self.crossover_rate,
            'mutation_rate': self.mutation_rate
        }

    @staticmethod
    def calculate_diversity(population, phenotypes):
        """
        Calcula una medida de diversidad basada en fenotipos.

        Args:
            population: Lista de individuos (genotipos)
            phenotypes: Lista de fenotipos correspondientes

        Returns:
            Valor de diversidad entre 0 y 1
        """
        if not phenotypes or len(phenotypes) <= 1:
            return 0

        unique_phenotypes = set(phenotypes)

        # Diversidad fenotípica
        phenotypic_diversity = len(unique_phenotypes) / len(phenotypes)

        # También podemos considerar la diversidad genotípica
        avg_length = sum(len(ind) for ind in population) / len(population)
        length_std = math.sqrt(sum((len(ind) - avg_length) ** 2 for ind in population) / len(population))

        # Normalizar la desviación estándar de longitud
        norm_length_std = min(1.0, length_std / (0.5 * avg_length))

        # Combinación ponderada
        return 0.7 * phenotypic_diversity + 0.3 * norm_length_std