# ************************************************************************
# * grammatical_evolution.py
# *
# * Implementación de evolución gramatical para cálculo de integrales
# * indefinidas usando optimización evolutiva
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
import re


class GrammaticalEvolution:
    def __init__(self, grammar,
                 codon_size=8,
                 max_wrapping=3,  # Valor pequeño pero distinto de cero (entre 1 y 4)
                 crossover_rate=0.9,
                 mutation_rate=0.1,
                 duplication_rate=0.05):
        self.grammar = grammar
        self.codon_size = codon_size
        self.max_wrapping = max_wrapping
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.duplication_rate = duplication_rate

    @staticmethod
    def _decode_codon(codon):
        """Convierte un codón binario a entero"""
        return int(codon, 2)

    @staticmethod
    def _map_codon_to_rule(codon_value, num_rules):
        """Mapea un valor de codón a una regla de producción"""
        return codon_value % num_rules


    def genotype_to_phenotype(self, genotype):
        """Mapea un genotipo (cadena binaria) a un fenotipo (expresión) y
        registra si se utilizó wrapping en el proceso."""
        # Dividir el genotipo en codones
        codons = [genotype[i:i + self.codon_size]
                  for i in range(0, len(genotype), self.codon_size)]

        current_symbol = "<expr>"  # Símbolo inicial
        codon_index = 0
        wrapping_count = 0
        wrapping_used = False  # Flag para registrar si se usó wrapping

        # Contador para limitar la complejidad de la expresión
        complexity = 0
        max_complexity = 100  # Límite arbitrario para prevenir explosiones

        # Mientras queden símbolos no terminales por expandir
        while "<" in current_symbol and wrapping_count < self.max_wrapping and complexity < max_complexity:
            complexity += 1

            # Si se acabaron los codones, volver a empezar (wrapping)
            if codon_index >= len(codons):
                codon_index = 0
                wrapping_count += 1
                wrapping_used = True  # Se activó el wrapping

            # Buscar el primer no terminal
            start = current_symbol.find("<")
            end = current_symbol.find(">") + 1
            symbol = current_symbol[start:end]

            # Obtener las reglas para este símbolo
            rules = self.grammar[symbol[1:-1]]

            # Obtener el valor del codón y seleccionar una regla
            codon_value = self._decode_codon(codons[codon_index])
            rule_index = self._map_codon_to_rule(codon_value, len(rules))

            # Reemplazar el símbolo por la regla seleccionada
            current_symbol = current_symbol[:start] + rules[rule_index] + current_symbol[end:]

            codon_index += 1

        # Si la expresión sigue teniendo no-terminales o es demasiado compleja
        if "<" in current_symbol or complexity >= max_complexity:
            return "x", True  # Devolver una expresión simple por defecto y señalar wrapping

        # Validar la expresión antes de devolverla
        try:
            # Verificar si hay patrones problemáticos y corregirlos
            if '(' in current_symbol and ')(' in current_symbol:
                current_symbol = re.sub(r'\)\(', ')*(', current_symbol)

            # También puedes verificar otros patrones problemáticos
            current_symbol = current_symbol.replace('x/1.0', 'x')  # Simplificar divisiones por 1
            current_symbol = current_symbol.replace('1.0*x', 'x')  # Simplificar multiplicaciones por 1
        except (re.error, ValueError, TypeError):  # Excepciones específicas que podrían ocurrir
            # Si hay algún error en la corrección, usar la expresión original
            pass

        return current_symbol, wrapping_used

    def adjust_wrapping(self, wrapping_stats, generation, total_generations, problem_id=1):
        """
        Ajuste mejorado del parámetro wrapping basado en las recomendaciones
        del documento 'Sobre el uso de Grammatical Evolution'.

        Args:
            wrapping_stats: Porcentaje de individuos que usaron wrapping
            generation: Generación actual
            total_generations: Número total de generaciones
            problem_id: Identificador del problema (1, 2 o 3)

        Returns:
            Mensaje de acción recomendada (o None)
        """
        # Comportamiento diferenciado por problema
        if problem_id == 1 or problem_id == 2:
            # Estrategia mejorada para los problemas 1 y 2
            if generation < total_generations * 0.3:
                # En etapas iniciales, mantener wrapping adecuado
                self.max_wrapping = max(3, min(4, self.max_wrapping)) if problem_id == 1 else max(2, min(3,
                                                                                                         self.max_wrapping))
                return None

            # Después, ajustar según el porcentaje de uso
            if wrapping_stats > 0.4:  # Uso excesivo de wrapping
                return "INCREASE_CHROMOSOME_SIZE"
            elif wrapping_stats > 0.2:  # Uso moderado
                return None
            elif wrapping_stats > 0.05:  # Uso bajo
                if self.max_wrapping > 1:
                    self.max_wrapping -= 1
            else:  # Uso muy bajo o nulo
                if problem_id == 1 and generation > total_generations * 0.6:
                    self.max_wrapping = 0
                else:
                    # Para problema 2, mantener al menos 1
                    self.max_wrapping = max(1, self.max_wrapping - 1)
        elif problem_id == 3:
            # Para el problema 3, mantener EXACTAMENTE el comportamiento original
            if generation > total_generations * 0.5:
                if wrapping_stats > 0.2 and self.max_wrapping > 1:
                    self.max_wrapping -= 1

            # Para problema 3: Reducir a 1 en etapas avanzadas
            if generation > total_generations * 0.8:
                self.max_wrapping = 1

        return None




    def initialize_population(self, population_size, min_codons=5, max_codons=30):
        """
        Inicializa una población diversa con individuos de diferentes tamaños.
        Se mejora la aleatoriedad en la distribución de longitudes.
        """
        population = []

        # Aproximadamente un 50% de la población con distribuciones escalonadas
        # y el otro 50% puramente aleatoria en el rango [min_codons, max_codons].
        step_size = (max_codons - min_codons) // 3 if (max_codons - min_codons) >= 3 else 1

        # Primera mitad: escalonada
        for size_group in range(3):
            size_min = min_codons + size_group * step_size
            size_max = min_codons + (size_group + 1) * step_size
            if size_max > max_codons:
                size_max = max_codons
            group_count = population_size // 6  # un sexto por grupo

            for _ in range(group_count):
                num_codons = random.randint(size_min, size_max)
                genotype = ''.join(random.choice(['0', '1'])
                                   for _ in range(num_codons * self.codon_size))
                population.append(genotype)

        # Segunda mitad: completamente aleatoria
        while len(population) < population_size:
            num_codons = random.randint(min_codons, max_codons)
            genotype = ''.join(random.choice(['0', '1'])
                               for _ in range(num_codons * self.codon_size))
            population.append(genotype)

        return population

    @staticmethod
    def selection(population, fitness_values, tournament_size=3, tournament_prob=0.8):
        """
        Selección por torneo para escoger individuos para reproducción.
        Usa un tamaño de torneo mayor (3) para aumentar la presión selectiva.

        Args:
            population: Lista de genotipos (cadenas binarias)
            fitness_values: Lista de valores de aptitud correspondientes
            tournament_size: Tamaño del torneo
            tournament_prob: Probabilidad de seleccionar al mejor individuo del torneo

        Returns:
            Índice del individuo seleccionado
        """
        # Seleccionar individuos aleatorios para el torneo
        tournament_indices = random.sample(range(len(population)), tournament_size)

        # Ordenar por fitness (asumiendo problema de minimización)
        tournament_indices.sort(key=lambda idx: fitness_values[idx])

        # Con probabilidad tournament_prob seleccionar al mejor, si no seleccionar al azar
        if random.random() < tournament_prob:
            return tournament_indices[0]  # El mejor
        else:
            return random.choice(tournament_indices)  # Cualquiera del torneo

    def crossover(self, parent1, parent2):
        """
        Operador de cruce de un punto mejorado.
        Asegura que los puntos de cruce sean válidos para ambos padres y
        que los hijos resultantes no excedan el tamaño máximo permitido.
        """
        if random.random() > self.crossover_rate:
            return parent1, parent2

        # Determinar tamaños máximos para los puntos de cruce
        max_point1 = len(parent1) - 1
        max_point2 = len(parent2) - 1

        # Asegurar que los puntos de cruce sean múltiplos del tamaño del codón
        # para mantener la integridad de los codones
        max_codon1 = max_point1 // self.codon_size
        max_codon2 = max_point2 // self.codon_size

        if max_codon1 == 0 or max_codon2 == 0:
            return parent1, parent2

        # Seleccionar puntos de cruce a nivel de codón
        codon_crosspoint1 = random.randint(1, max_codon1)
        codon_crosspoint2 = random.randint(1, max_codon2)

        # Convertir a posiciones de bits
        crosspoint1 = codon_crosspoint1 * self.codon_size
        crosspoint2 = codon_crosspoint2 * self.codon_size

        # Crear descendientes
        child1 = parent1[:crosspoint1] + parent2[crosspoint2:]
        child2 = parent2[:crosspoint2] + parent1[crosspoint1:]

        return child1, child2

    def crossover_with_common_point(self, parent1, parent2):
        """Cruce con punto común para preservar mejor la estructura"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        # Encontrar el tamaño mínimo entre los dos padres
        min_len = min(len(parent1), len(parent2))
        if min_len <= self.codon_size:
            return parent1, parent2

        # Seleccionar un punto de cruce común
        crosspoint = random.randint(self.codon_size, min_len - self.codon_size)
        crosspoint = (crosspoint // self.codon_size) * self.codon_size  # Alinear con codones

        # Crear descendientes
        child1 = parent1[:crosspoint] + parent2[crosspoint:]
        child2 = parent2[:crosspoint] + parent1[crosspoint:]

        return child1, child2

    def mutation(self, individual):
        """
        Operador de mutación de punto mejorado.
        Invierte bits aleatorios con probabilidad mutation_rate,
        con mayor énfasis en los codones situados a la derecha
        para preservar mejor la localidad.
        """
        if len(individual) == 0:
            return individual

        mutated = list(individual)

        # Calcular el número de codones
        num_codons = len(individual) // self.codon_size

        # Matriz de probabilidades que favorece la mutación en codones a la derecha
        # (menos disruptiva para la propiedad de localidad)
        probabilities = []
        for i in range(num_codons):
            # Incrementar la probabilidad de mutación a medida que nos movemos
            # hacia codones más a la derecha (menos significativos)
            codon_prob = self.mutation_rate * (1 + 0.5 * i / num_codons)
            probabilities.extend([codon_prob] * self.codon_size)

        # Extender si es necesario para cubrir el individuo
        probabilities.extend([self.mutation_rate] * (len(individual) - len(probabilities)))

        # Aplicar mutación
        for i in range(len(individual)):
            if random.random() < probabilities[i]:
                mutated[i] = '1' if mutated[i] == '0' else '0'  # Invertir bit

        return ''.join(mutated)

    def duplication(self, individual):
        """
        Operador de duplicación mejorado.
        Duplica una sección aleatoria del cromosoma y la añade al final,
        comprobando que no se exceda un tamaño máximo razonable.
        """
        if random.random() > self.duplication_rate:
            return individual

        # Establecer un tamaño máximo razonable (por ejemplo, 30 codones)
        max_size = 30 * self.codon_size

        # Longitud en codones
        num_codons = len(individual) // self.codon_size

        if num_codons <= 1:
            return individual

        # Seleccionar el número de codones a duplicar (entre 1 y la mitad del cromosoma)
        num_to_duplicate = random.randint(1, max(1, num_codons // 2))

        # Verificar que no exceda el tamaño máximo tras la duplicación
        if len(individual) + (num_to_duplicate * self.codon_size) > max_size:
            num_to_duplicate = (max_size - len(individual)) // self.codon_size
            if num_to_duplicate <= 0:
                return individual  # Ya está en el tamaño máximo

        # Seleccionar posición de inicio
        start_codon = random.randint(0, num_codons - num_to_duplicate)

        # Convertir a posiciones de bits
        start_bit = start_codon * self.codon_size
        end_bit = start_bit + (num_to_duplicate * self.codon_size)

        # Duplicar la sección seleccionada
        section_to_duplicate = individual[start_bit:end_bit]

        # Añadir al final del cromosoma
        return individual + section_to_duplicate

    def structural_mutation(self, individual, rate=0.04):
        """
        Operador de mutación estructural mejorado.
        Realiza cambios más significativos para ayudar a escapar de óptimos locales,
        pero sin operaciones específicas por problema.
        """
        if random.random() > rate:
            return individual

        # Dividir el genotipo en codones
        codons = [individual[i:i + self.codon_size]
                  for i in range(0, len(individual), self.codon_size)]

        if len(codons) <= 2:
            return individual

        # Seleccionar una operación estructural aleatoria
        operations = ['insert', 'delete', 'replace', 'swap', 'duplicate_segment']
        operation = random.choice(operations)

        if operation == 'insert' and len(codons) < 30:  # Limitar tamaño máximo
            # Insertar un codón aleatorio en una posición aleatoria
            new_codon = ''.join(random.choice(['0', '1']) for _ in range(self.codon_size))
            position = random.randint(0, len(codons))
            codons.insert(position, new_codon)

        elif operation == 'delete' and len(codons) > 3:
            # Eliminar un codón aleatorio
            position = random.randint(0, len(codons) - 1)
            codons.pop(position)

        elif operation == 'replace':
            # Reemplazar un codón aleatorio
            position = random.randint(0, len(codons) - 1)
            new_codon = ''.join(random.choice(['0', '1']) for _ in range(self.codon_size))
            codons[position] = new_codon

        elif operation == 'swap' and len(codons) > 3:
            # Intercambiar dos codones aleatorios
            pos1, pos2 = random.sample(range(len(codons)), 2)
            codons[pos1], codons[pos2] = codons[pos2], codons[pos1]

        elif operation == 'duplicate_segment' and 2 < len(codons) < 25:
            # Duplicar un segmento aleatorio
            segment_length = random.randint(1, min(3, len(codons) // 2))
            start_pos = random.randint(0, len(codons) - segment_length)
            segment = codons[start_pos:start_pos + segment_length]

            # Insertar en una posición aleatoria
            insert_pos = random.randint(0, len(codons))
            codons = codons[:insert_pos] + segment + codons[insert_pos:]

        # Reconstruir el genotipo
        return ''.join(codons)