# ************************************************************************
# * main.py
# *
# * Script principal para experimentos de cálculo de integrales
# * indefinidas mediante evolución gramatical
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
import matplotlib.pyplot as plt
import random
import argparse
import warnings
import os
import re
import types
from tqdm import tqdm
import pandas as pd
from collections import defaultdict


from grammatical_evolution import GrammaticalEvolution
from evaluation import evaluator1, evaluator2, evaluator3, problem1_function, problem2_function, problem3_function
from local_search import LocalSearch
from parameter_control import ParameterController

# Gramática única para todos los problemas según recomendaciones
GRAMMAR = {
    "expr": [
        "<expr><op><expr>",
        "(<expr><op><expr>)",
        "<pre_op>(<expr>)",
        "<var>"
    ],
    "op": ["+", "-", "*", "/"],
    "pre_op": ["sen", "cos", "exp", "ln"],
    "var": ["x", "1.0"]
}

# Configuración general para todos los problemas
ALGORITHM_CONFIG = {
    'grammar': GRAMMAR,
    'max_wrapping': 3,  # Valor pequeño pero distinto de cero (entre 1 y 4)
    'mutation_rate': 0.1,
    'duplication_rate': 0.05,  # Añadir este parámetro
    'ls_probability': 0.15,
    'population_size': 250,
    'tournament_size': 3,
    'structural_mutation_rate': 0.04,
    'elitism_size': 1,  # Elitismo para preservar el mejor individuo
    'min_codons': 8,  # Cromosomas más largos inicialmente
    'max_codons': 30  # Cromosomas más largos inicialmente
}

def validate_expression(expression):
    """
    Valida la sintaxis de una expresión antes de evaluarla, evitando warnings.
    También realiza una corrección preliminar de patrones problemáticos,
    como ')(' => ')*(' y '))(' => '))*(('.

    Si aparece un SyntaxWarning (p. ej. 'tuple' object is not callable),
    se trata como excepción y devuelve False.
    """

    # Reemplazos por librerías:
    test_expr = (expression
                 .replace('sen', 'np.sin')
                 .replace('cos', 'np.cos')
                 .replace('exp', 'np.exp')
                 .replace('ln', 'np.log'))

    # Correcciones de patrones que suelen causar "tuple object is not callable"
    if ')(' in test_expr:
        test_expr = re.sub(r'\)\(', ')*(', test_expr)
    if '))(' in test_expr:
        test_expr = re.sub(r'\)\)\(', '))*((', test_expr)

    try:
        # Capturamos warnings de tipo SyntaxWarning como errores
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("error", category=SyntaxWarning)
            compile(test_expr, "<string>", "eval")

        # Si llega aquí, no ha habido SyntaxError ni SyntaxWarning
        return True
    except (SyntaxError, Warning):
        # Cualquier SyntaxError o SyntaxWarning hace que devuelva False
        return False

# Función para ejecutar el algoritmo con la configuración unificada
def run_experiment(problem_id, evaluator, use_local_search=True, use_adaptive_control=True,
                   runs=30, master_seed=42, verbose=False):
    """
    Ejecuta el experimento para un problema específico con la configuración optimizada.

    Args:
        problem_id: Identificador del problema (1, 2 o 3)
        evaluator: Evaluador específico del problema
        use_local_search: Si se utiliza búsqueda local
        use_adaptive_control: Si se utiliza control adaptativo de parámetros
        runs: Número de ejecuciones independientes
        master_seed: Semilla maestra para reproducibilidad global
        verbose: Verbose del experimento

    Returns:
        Dictionary con resultados estadísticos
    """
    # Inicializar resultados
    exp_results = {
        'best_fitness': [],
        'best_expressions': [],
        'success_count': 0,
        'evals_to_success': [],
        'convergence_data': [],
        'diversity_data': [],
        'mutation_rate_data': [],
        'crossover_rate_data': [],
        'seeds_used': []
    }

    # Configuración base (igual para todos los problemas)
    config = ALGORITHM_CONFIG.copy()

    if problem_id == 1:
        # Configuración optimizada para problema 1
        config['tournament_size'] = 2  # Menor presión selectiva para explorar más
        config['population_size'] = 500  # Mayor población
        config['max_generations'] = 400  # Más generaciones
        config['min_codons'] = 40  # Cromosomas más largos
        config['max_codons'] = 100  # Cromosomas más largos
        config['mutation_rate'] = 0.15  # Mayor tasa de mutación
        config['duplication_rate'] = 0.05  # Tasa de duplicación modificada

        # Para problema 1, usar algoritmo EG-BL si está disponible
        if not use_local_search and not use_adaptive_control:
            print("Advertencia: Para el problema 1 se recomienda usar EG-BL")
    else:
        # No modificar NINGÚN parámetro para problemas 2 y 3
        # Los valores ya están en ALGORITHM_CONFIG
        # Solo mostrar advertencia si no es Memético
        if not (use_local_search and use_adaptive_control):
            print("Advertencia: Para los problemas 2 y 3 se recomienda usar Memético")

    # Fijar la semilla maestra para generar las semillas de cada ejecución
    random.seed(master_seed)
    np.random.seed(master_seed)

    # Generar semillas deterministas pero diferentes para cada ejecución
    run_seeds = [random.randint(1, 100000) for _ in range(runs)]
    exp_results['seeds_used'] = run_seeds

    # Crear directorio para resultados si no existe
    os.makedirs('results', exist_ok=True)

    # Para cada ejecución independiente
    for run in tqdm(range(runs), desc=f"Problema {problem_id}"):
        # Establecer la semilla específica para esta ejecución
        current_seed = run_seeds[run]
        random.seed(current_seed)
        np.random.seed(current_seed)

        # Crear instancia de evolución gramatical
        ge = GrammaticalEvolution(
            grammar=config['grammar'],
            codon_size=8,
            max_wrapping=config['max_wrapping'],
            crossover_rate=config['crossover_rate'] if 'crossover_rate' in config else 0.9,
            mutation_rate=config['mutation_rate'],
            duplication_rate=config['duplication_rate'] if 'duplication_rate' in config else 0.05
        )

        # Si es el problema 1, reemplazar la función de crossover con crossover_with_common_point
        if problem_id == 1:
            ge.crossover = types.MethodType(GrammaticalEvolution.crossover_with_common_point, ge)

        # Inicializar controlador de parámetros si se usa
        param_controller = None
        if use_adaptive_control:
            param_controller = ParameterController(
                init_crossover_rate=config['crossover_rate'] if 'crossover_rate' in config else 0.9,
                init_mutation_rate=config['mutation_rate']
            )

        # Inicializar mecanismo de búsqueda local si se usa
        local_search = None
        if use_local_search:
            local_search = LocalSearch(
                evaluator=evaluator,
                ls_probability=config['ls_probability']
            )
            # Ajustes especiales para el problema 1
            if problem_id == 1:
                # Más intensidad en la búsqueda local
                local_search.max_iterations = 8  # Más iteraciones de BL
                local_search.ls_probability = 0.25  # Mayor probabilidad

        # Inicializar población
        population_size = config['population_size']
        population = ge.initialize_population(population_size)

        # Variables para seguimiento de evolución
        best_fitness = float('inf')
        best_expression = None
        eval_count = 0
        success_found_at = -1
        max_generations = 100  # Valor estándar

        # Datos para gráficos de convergencia
        gen_data = []
        best_fit_data = []
        diversity_data = []
        mutation_data = []
        crossover_data = []

        # Ciclo evolutivo principal
        for generation in range(max_generations):
            # (1) Mapear genotipos a fenotipos
            phenotypes = []
            wrapping_used_count = 0
            for ind in population:
                phenotype, wrapping_used = ge.genotype_to_phenotype(ind)
                if wrapping_used:
                    wrapping_used_count += 1
                # Validar para evitar problemas de sintaxis
                if not validate_expression(phenotype):
                    phenotype = "x"  # Usar expresión simple por defecto
                phenotypes.append(phenotype)

            # Calcular estadísticas de wrapping
            wrapping_percentage = wrapping_used_count / len(population)

            # Ajustar parámetro de wrapping dinámicamente
            action = ge.adjust_wrapping(wrapping_percentage, generation, max_generations, problem_id)

            # Aplicar ajuste de tamaño de cromosoma a los problemas 1 y 2
            if action == "INCREASE_CHROMOSOME_SIZE" and problem_id != 3 and generation < max_generations * 0.7:
                old_min = config['min_codons']
                old_max = config['max_codons']

                # Aumentar tamaños con límites razonables, más agresivamente para problema 1
                increase_factor = 1.2 if problem_id == 1 else 1.1
                config['min_codons'] = min(int(old_min * increase_factor), int(old_max * 0.5))
                config['max_codons'] = min(int(old_max * increase_factor), 300)  # Con límite superior

                if verbose and (config['min_codons'] != old_min or config['max_codons'] != old_max):
                    print(
                        f"Gen {generation}: Ajustando tamaño de cromosoma de [{old_min}, {old_max}] a [{config['min_codons']}, {config['max_codons']}]")

            # Imprimir estadísticas cada 10 generaciones
            if generation % 10 == 0 and verbose:
                print(f"Gen {generation}: {wrapping_percentage * 100:.2f}% wrapping, max_wrapping={ge.max_wrapping}")

            # (2) Calcular diversidad
            diversity = ParameterController.calculate_diversity(population, phenotypes)

            # Intensificar búsqueda local en etapas avanzadas
            if use_local_search and local_search is not None:
                if problem_id == 1:
                    # Para problema 1, intensificar más en etapas avanzadas
                    if generation > max_generations * 0.8:
                        local_search.max_iterations = 15
                        local_search.ls_probability = 0.5
                    else:
                        local_search.max_iterations = 8
                        local_search.ls_probability = 0.25
                else:
                    pass
                    # Para problemas 2 y 3, mantener configuración original
                    # if generation > max_generations * 0.8:
                    #    local_search.max_iterations = 8
                    #    local_search.ls_probability = 0.3
                    # else:
                    #     local_search.max_iterations = 5
                    #     local_search.ls_probability = 0.15

            # (3) Aplicar búsqueda local si procede y queremos que dependa de la diversidad
            if use_local_search and local_search is not None:
                population, phenotypes = local_search.apply(population, phenotypes, diversity=diversity)

            # Evaluar fitness
            fitness_values = []
            for phenotype in phenotypes:
                fitness = evaluator.evaluate(phenotype)
                fitness_values.append(fitness)
                eval_count += 1

            # Actualizar mejor solución
            min_idx = np.argmin(fitness_values)
            if fitness_values[min_idx] < best_fitness:
                best_fitness = fitness_values[min_idx]
                best_expression = phenotypes[min_idx]

                # Verificar si es una solución exitosa (error menor que umbral)
                if best_fitness < 1.0 and success_found_at == -1:
                    success_found_at = eval_count

            # Actualizar parámetros adaptativamente si corresponde
            if use_adaptive_control and param_controller:
                param_controller.update_parameters(best_fitness, diversity)
                params = param_controller.get_parameters()
                ge.crossover_rate = params['crossover_rate']
                ge.mutation_rate = params['mutation_rate']

            # Guardar datos para gráficos
            gen_data.append(generation)
            best_fit_data.append(best_fitness)
            diversity_data.append(diversity)
            mutation_data.append(ge.mutation_rate)
            crossover_data.append(ge.crossover_rate)

            # Crear nueva población
            new_population = []

            # Elitismo: preservar los mejores individuos
            elitism_size = config['elitism_size']
            elite_indices = np.argsort(fitness_values)[:elitism_size]
            for idx in elite_indices:
                new_population.append(population[idx])

            # Generar resto de la población mediante selección, cruce y mutación
            while len(new_population) < population_size:
                # Seleccionar padres
                parent1_idx = ge.selection(population, fitness_values, config['tournament_size'], 0.8)
                parent2_idx = ge.selection(population, fitness_values, config['tournament_size'], 0.8)

                # Aplicar cruce
                child1, child2 = ge.crossover(population[parent1_idx], population[parent2_idx])

                # Aplicar mutación
                child1 = ge.mutation(child1)
                child2 = ge.mutation(child2)

                # Aplicar mutación estructural con tasa específica
                child1 = ge.structural_mutation(child1, rate=config['structural_mutation_rate'])
                child2 = ge.structural_mutation(child2, rate=config['structural_mutation_rate'])

                # Aplicar duplicación
                child1 = ge.duplication(child1)
                child2 = ge.duplication(child2)

                # Añadir a nueva población
                new_population.append(child1)
                if len(new_population) < population_size:
                    new_population.append(child2)

            # Reemplazar población
            population = new_population

        # Guardar resultados de esta ejecución
        exp_results['best_fitness'].append(best_fitness)
        exp_results['best_expressions'].append(best_expression)

        if success_found_at != -1:
            exp_results['success_count'] += 1
            exp_results['evals_to_success'].append(success_found_at)

        exp_results['convergence_data'].append((gen_data, best_fit_data))
        exp_results['diversity_data'].append((gen_data, diversity_data))
        exp_results['mutation_rate_data'].append((gen_data, mutation_data))
        exp_results['crossover_rate_data'].append((gen_data, crossover_data))

    # Calcular estadísticas
    exp_results['mean_best_fitness'] = np.mean(exp_results['best_fitness'])
    exp_results['success_rate'] = exp_results['success_count'] / runs
    if exp_results['evals_to_success']:
        exp_results['mean_evals_to_success'] = np.mean(exp_results['evals_to_success'])
    else:
        exp_results['mean_evals_to_success'] = float('inf')

    return exp_results


def try_(exp, v, sl, y_f):
    """
    Evalúa la expresión 'exp' en x=v, usando 'safe_log'=sl,
    intentando corregir tuplas mal interpretadas.
    Guarda el resultado (o np.nan si falla) en la lista y_f.

    - exp: expresión matemática
    - v: valor de la variable 'x'
    - sl: referencia a la función 'safe_log' (o similar)
    - y_f: lista donde añadir el resultado (o NaN si hay error)
    """

    original_exp = exp  # Guardamos la expresión original para depuración

    try:
        # Corregir patrones conflictivos de tuplas que disparan el SyntaxWarning
        if '(' in exp and ')(' in exp:
            exp = re.sub(r'\)\(', ')*(', exp)
        if '))(' in exp:
            exp = re.sub(r'\)\)\(', '))*((', exp)

        # Evaluación numérica con supresión de warning de numpy
        with np.errstate(all='ignore'):
            result = eval(exp, {"__builtins__": {}},
                          {'x': v, 'np': np, 'safe_log': sl})

        # Verificar si result es infinito o NaN
        if np.isinf(result) or np.isnan(result):
            y_f.append(np.nan)
        else:
            y_f.append(result)

    except (ArithmeticError, ValueError, SyntaxError, NameError, TypeError) as e:
        # Si pasa algo que sugiera 'tuple' object is not callable, mostramos la expresión un % de veces
        if "tuple" in str(e) and random.random() < 0.01:
            print(f"Expresión problemática: {original_exp}")
            print(f"Después de corrección: {exp}")
            print(f"Error: {e}")

        # Guardar NaN para indicar que hubo un fallo en la evaluación
        y_f.append(np.nan)

# Función para comparar solución encontrada con solución exacta
def plot_solution_comparison(problem_id, best_expression, interval, exact_function_str):
    """
    Genera un gráfico comparando la solución encontrada con la solución exacta.
    """
    a, b = interval
    x = np.linspace(a, b, 100)

    # Definir función segura para logaritmo
    def safe_log(var):
        return np.log(var) if var > 0 else np.nan

    # Evaluar la expresión encontrada
    y_found = []
    for val in x:
        expr = best_expression.replace('sen', 'np.sin').replace('cos', 'np.cos')
        expr = expr.replace('exp', 'np.exp').replace('ln', 'safe_log')
        try_(expr, val, safe_log, y_found)

    # Evaluar la función exacta
    y_exact = []
    for val in x:
        try_(exact_function_str, val, safe_log, y_exact)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y_exact, 'b-', label='Solución exacta')
    plt.plot(x, y_found, 'r--', label='Solución encontrada')
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title(f'Comparación de soluciones para el Problema {problem_id}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'results/problema{problem_id}_comparacion.png')
    plt.close()


# Función para generar gráfico de convergencia promedio
def plot_convergence(problem_id, results_dict):
    """
    Genera un gráfico de convergencia comparando las diferentes variantes del algoritmo.
    """
    plt.figure(figsize=(10, 6))

    for current_variant, current_results in results_dict.items():
        # Calcular convergencia promedio
        all_generations = []
        all_fitness = []

        for gen_data, fit_data in current_results['convergence_data']:
            all_generations.append(gen_data)
            all_fitness.append(fit_data)

        # Asegurar que todas las ejecuciones tienen la misma longitud
        min_len = min(len(g) for g in all_generations)
        avg_fitness = np.mean([f[:min_len] for f in all_fitness], axis=0)
        generations = all_generations[0][:min_len]

        plt.plot(generations, avg_fitness, label=current_variant)

    plt.xlabel('Generación')
    plt.ylabel('Mejor fitness')
    plt.title(f'Convergencia para el Problema {problem_id}')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(f'results/problema{problem_id}_convergencia.png')
    plt.close()


# Función para generar gráfico de barras comparando los índices TE, VAMM, PEX
def plot_comparison_metrics(all_results):
    """
    Genera gráficos de barras comparando las métricas TE, VAMM y PEX.
    """
    # Preparar datos para gráficos
    variants = list(all_results[1].keys())
    problems = list(all_results.keys())

    # Gráfico para Tasa de Éxito (TE)
    plt.figure(figsize=(10, 6))
    bar_width = 0.25
    index = np.arange(len(problems))

    for i, current_variant in enumerate(variants):
        te_values = [all_results[p][current_variant]['success_rate'] * 100 for p in problems]
        plt.bar(index + i * bar_width, te_values, bar_width, label=current_variant)

    plt.xlabel('Problema')
    plt.ylabel('Tasa de Éxito (%)')
    plt.title('Comparación de Tasa de Éxito (TE)')
    plt.xticks(index + bar_width, [f'Problema {p}' for p in problems])
    plt.legend()
    plt.savefig('results/comparacion_TE.png')
    plt.close()

    # Gráfico para VAMM
    plt.figure(figsize=(10, 6))

    for i, current_variant in enumerate(variants):
        vamm_values = [all_results[p][current_variant]['mean_best_fitness'] for p in problems]
        plt.bar(index + i * bar_width, vamm_values, bar_width, label=current_variant)

    plt.xlabel('Problema')
    plt.ylabel('VAMM')
    plt.title('Comparación de Valor Adaptación Medio del Mejor (VAMM)')
    plt.xticks(index + bar_width, [f'Problema {p}' for p in problems])
    plt.legend()
    plt.savefig('results/comparacion_VAMM.png')
    plt.close()

    # Gráfico para PEX
    plt.figure(figsize=(10, 6))

    for i, current_variant in enumerate(variants):
        pex_values = []
        for p in problems:
            if all_results[p][current_variant]['mean_evals_to_success'] == float('inf'):
                pex_values.append(0)  # No hubo éxitos
            else:
                pex_values.append(all_results[p][current_variant]['mean_evals_to_success'])

        plt.bar(index + i * bar_width, pex_values, bar_width, label=current_variant)

    plt.xlabel('Problema')
    plt.ylabel('PEX')
    plt.title('Comparación de Promedio de Evaluaciones hasta el Éxito (PEX)')
    plt.xticks(index + bar_width, [f'Problema {p}' for p in problems])
    plt.legend()
    plt.savefig('results/comparacion_PEX.png')
    plt.close()


# Definición de las soluciones exactas
exact_solutions = {
    1: "(1/4)*(x**2+1)*(x-1)",  # F(x) = (1/4)(x^2 + 1)(x - 1)
    2: "x*np.log(1+x)",  # F(x) = x*ln(1 + x)
    3: "np.exp(x)*np.sin(x)"  # F(x) = e^x*sin(x)
}

exact_expr_solutions = {
    1: "(1/4)*(x**2+1)*(x-1)",  # Para evaluación
    2: "x*ln(1+x)",  # Para evaluación
    3: "exp(x)*sen(x)"  # Para evaluación
}

# Configuración para cada problema
problem_configs = {
    1: {
        'evaluator': evaluator1,
        'interval': [-2, 2],
        'target_function': problem1_function
    },
    2: {
        'evaluator': evaluator2,
        'interval': [0, 5],
        'target_function': problem2_function
    },
    3: {
        'evaluator': evaluator3,
        'interval': [-2, 2],
        'target_function': problem3_function
    }
}

# Variantes del algoritmo a comparar
algorithm_variants = {
    'EG-Base': {'use_local_search': False, 'use_adaptive_control': False},
    'EG-BL': {'use_local_search': True, 'use_adaptive_control': False},
    'Memético': {'use_local_search': True, 'use_adaptive_control': True}
}

# Mensaje informativo sobre configuraciones óptimas
print("\nInformación sobre configuraciones óptimas:")
print("- Problema 1: Recomendado usar EG-BL con configuración especializada")
print("- Problema 2: Recomendado usar Memético con configuración original")
print("- Problema 3: Recomendado usar Memético con configuración original")
print()


# Función principal
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evolución Gramatical para Cálculo de Integrales Indefinidas')
    parser.add_argument('--master-seed', type=int, default=42,
                        help='Semilla maestra para reproducibilidad global')
    parser.add_argument('--num-runs', type=int, default=5,
                        help='Número de ejecuciones independientes por experimento')
    parser.add_argument('--test-mode', action='store_true',
                        help='Ejecutar en modo prueba con menos ejecuciones')
    # Nuevos parámetros para seleccionar problema y variante
    parser.add_argument('--problem', type=int, choices=[1, 2, 3], default=None,
                        help='Especificar un problema concreto (1, 2 o 3). Si no se especifica, se ejecutan todos.')
    parser.add_argument('--variant', type=str, choices=['EG-Base', 'EG-BL', 'Memético'], default=None,
                        help='Especificar una variante concreta. Si no se especifica, se ejecutan todas.')
    args = parser.parse_args()

    # Número de ejecuciones (reducido para pruebas si se especifica)
    num_runs = 5 if args.test_mode else args.num_runs

    # Almacenar resultados de todos los experimentos
    all_exp_results = defaultdict(dict)

    # Definir qué problemas ejecutar
    problems_to_run = [args.problem] if args.problem is not None else list(problem_configs.keys())

    # Definir qué variantes ejecutar
    variants_to_run = {args.variant: algorithm_variants[args.variant]} if args.variant else algorithm_variants

    # Ejecutar experimentos para cada problema y variante seleccionada
    for prob_id in problems_to_run:
        global_config = problem_configs[prob_id]
        for variant_name, variant_config in variants_to_run.items():
            print(f"\nEjecutando {variant_name} en Problema {prob_id}...")

            # Ejecutar experimento
            results = run_experiment(
                problem_id=prob_id,
                evaluator=global_config['evaluator'],
                use_local_search=variant_config['use_local_search'],
                use_adaptive_control=variant_config['use_adaptive_control'],
                runs=num_runs,
                master_seed=args.master_seed,
                verbose=True,
            )

            # Guardar resultados
            all_exp_results[prob_id][variant_name] = results

            # Mostrar resultados
            print(f"  Tasa de Éxito (TE): {results['success_rate'] * 100:.1f}%")
            print(f"  VAMM: {results['mean_best_fitness']:.4f}")
            if results['evals_to_success']:
                print(f"  PEX: {results['mean_evals_to_success']:.1f}")
            else:
                print("  PEX: No hubo éxitos")

            # Si es la mejor variante (Memético), generar gráfico comparativo
            if variant_name == 'Memético':
                # Encontrar la mejor expresión
                best_idx = np.argmin(results['best_fitness'])
                best_expr = results['best_expressions'][best_idx]

                if best_expr:
                    print(f"  Mejor expresión encontrada: {best_expr}")

                    # Generar gráfico comparativo
                    plot_solution_comparison(
                        problem_id=prob_id,
                        best_expression=best_expr,
                        interval=global_config['interval'],
                        exact_function_str=exact_solutions[prob_id]
                    )

        # Generar gráfico de convergencia para este problema
        plot_convergence(prob_id, all_exp_results[prob_id])

    # Generar gráficos comparativos de métricas
    plot_comparison_metrics(all_exp_results)

    # Guardar resultados en formato CSV
    results_df = []
    for prob_id in all_exp_results:
        for variant in all_exp_results[prob_id]:
            results = all_exp_results[prob_id][variant]
            row = {
                'Problema': prob_id,
                'Variante': variant,
                'TE': results['success_rate'] * 100,
                'VAMM': results['mean_best_fitness'],
                'PEX': results['mean_evals_to_success'] if results['evals_to_success'] else 'N/A',
                'Mejor_Expresion': results['best_expressions'][np.argmin(results['best_fitness'])]
                if results['best_expressions'] else 'N/A'
            }
            results_df.append(row)

    pd.DataFrame(results_df).to_csv('results/resultados_completos.csv', index=False)

    print("\nExperimentos completados. Resultados guardados en el directorio 'results/'.")