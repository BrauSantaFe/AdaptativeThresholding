import rasterio as rio
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

class Band:
    def __init__(self, banda, ruta):
        self.banda = banda
        self.ruta = ruta
        self.data = None
        self.meta = None
        self.reflectance_max = None
        self.reflectance_min = None
        self.reflectance_mult = None
        self.reflectance_add = None
        
    def load(self):
        ruta_banda = glob.glob('**/*' + self.banda + '*.TIF', root_dir=self.ruta, recursive=True)
        if not ruta_banda:
            raise FileNotFoundError(f"No se encontró el archivo para la banda {self.banda} en la ruta {self.ruta}")
        ruta_banda = os.path.join(self.ruta, ruta_banda[0])
        with rio.open(ruta_banda) as data_src:
            self.data = data_src.read(1).astype(np.float64)
            self.meta = data_src.meta
        return self.data

    def MTL_load(self):
        ruta_mtl = glob.glob('**/*MTL.txt', root_dir=self.ruta, recursive=True)
        if not ruta_mtl:
            raise FileNotFoundError(f"No se encontró el archivo MTL en la ruta {self.ruta}")
        ruta_mtl = os.path.join(self.ruta, ruta_mtl[0])
        band_number = self.banda.replace('B', '')

        with open(ruta_mtl, 'r') as file:
            for line in file:
                if f'REFLECTANCE_MAXIMUM_BAND_{band_number}' in line:
                    self.reflectance_max = float(line.split('=')[1].strip())
                elif f'REFLECTANCE_MINIMUM_BAND_{band_number}' in line:
                    self.reflectance_min = float(line.split('=')[1].strip())
                elif f'REFLECTANCE_MULT_BAND_{band_number}' in line:
                    self.reflectance_mult = float(line.split('=')[1].strip())
                elif f'REFLECTANCE_ADD_BAND_{band_number}' in line:
                    self.reflectance_add = float(line.split('=')[1].strip())
        return (self.reflectance_max, self.reflectance_min, self.reflectance_mult, self.reflectance_add)


class Radiometric_correction:
    def __init__(self, band: Band):
        self.band = band

    def apply_radiometric_correction(self):
        if self.band.data is None:
            raise ValueError('La banda no ha sido cargada previamente')
        if self.band.reflectance_mult is None or self.band.reflectance_add is None:
            raise ValueError('Los metadatos MTL no han sido cargados')
        corrected_data = self.band.reflectance_mult * self.band.data + self.band.reflectance_add
        return corrected_data


class GeneticAlgorithm:
    def __init__(self, pop_size=50, generations=100, pc=0.7, pm=0.2, elite_size=2, sample_fraction=0.01, min_sample=65536):
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc
        self.pm = pm
        self.elite_size = elite_size
        self.sample_fraction = sample_fraction
        self.min_sample = min_sample

        self.LB = np.array([0.5, 0.5, -0.5, 0.1, 0.05, 1.0, -0.8])
        self.UB = np.array([3.5, 3.5, 0.5, 1.0, 1.0, 5.0, 0.2])
        self.n_vars = len(self.LB)
        self.fitness_history = []

    def initialize_population(self):
        return self.LB + (self.UB - self.LB) * np.random.rand(self.pop_size, self.n_vars)

    def evaluate_kapur(self, thresholds, features_sample):
        total_entropy = 0
        n_bins = 256
        eps = 1e-12
        for i in range(self.n_vars):
            data = features_sample[:, i]
            hist, bin_edges = np.histogram(data, bins=n_bins, range=(self.LB[i], self.UB[i]))
            p_i = hist / (np.sum(hist) + eps)
            t = thresholds[i]
            idx = np.searchsorted(bin_edges, t) - 1
            idx = np.clip(idx, 1, n_bins - 1)
            w0 = np.sum(p_i[:idx])
            w1 = np.sum(p_i[idx:])
            if w0 > eps and w1 > eps:
                p0 = p_i[:idx] / w0
                p1 = p_i[idx:] / w1
                h0 = -np.sum(p0[p0 > 0] * np.log(p0[p0 > 0] + eps))
                h1 = -np.sum(p1[p1 > 0] * np.log(p1[p1 > 0] + eps))
                total_entropy += (h0 + h1)
        return total_entropy

    def apply_selection(self, population, fitness):
        new_pop = np.zeros_like(population)
        elite_idx = np.argsort(fitness)[-self.elite_size:]
        new_pop[:self.elite_size] = population[elite_idx]
        for i in range(self.elite_size, self.pop_size):
            idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
            new_pop[i] = population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]
        return new_pop

    def apply_crossover(self, population):
        new_pop = population.copy()
        for i in range(self.elite_size, self.pop_size, 2):
            if i + 1 < self.pop_size and np.random.rand() < self.pc:
                alpha = np.random.rand()
                p1 = population[i].copy()
                p2 = population[i + 1].copy()
                new_pop[i] = alpha * p1 + (1 - alpha) * p2
                new_pop[i + 1] = alpha * p2 + (1 - alpha) * p1
        return new_pop

    def apply_mutation(self, population):
        for i in range(self.elite_size, self.pop_size):
            for j in range(self.n_vars):
                if np.random.rand() < self.pm:
                    scale = 0.1 * (self.UB[j] - self.LB[j])
                    population[i, j] += np.random.normal(0, scale)
                    population[i, j] = np.clip(population[i, j], self.LB[j], self.UB[j])
        return population

    def run(self, features_data):
        N_total = features_data.shape[0]
        N_sample = min(max(self.min_sample, int(N_total * self.sample_fraction)), N_total)
        sample_idx = np.random.choice(N_total, N_sample, replace=False)
        features_sample = features_data[sample_idx]

        population = self.initialize_population()
        best_overall_ind = None
        best_overall_fit = -np.inf

        for gen in range(self.generations):
            fitness = np.array([self.evaluate_kapur(ind, features_sample) for ind in population])
            current_best_idx = np.argmax(fitness)
            self.fitness_history.append(fitness[current_best_idx])
            if fitness[current_best_idx] > best_overall_fit:
                best_overall_fit = fitness[current_best_idx]
                best_overall_ind = population[current_best_idx].copy()
            population = self.apply_selection(population, fitness)
            population = self.apply_crossover(population)
            population = self.apply_mutation(population)
            if gen % 10 == 0:
                print(f"Gen {gen} | Max Entropía: {fitness.max():.4f}")
        return best_overall_ind


class Segmenter:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def segment(self, features_data):
        F = features_data
        T = self.thresholds
        mask = (((F[:, 0] > T[0]) | (F[:, 5] > T[5])) &
                (F[:, 1] > T[1]) &
                (F[:, 2] > T[2]) &
                ((F[:, 3] > 0.8 * T[3]) | (F[:, 4] > 0.5 * T[4])) &
                (F[:, 6] < T[6]))
        return mask.astype(np.uint8)


class Visualizer:
    def __init__(self, segmented_data, features_data, image_shape):
        self.segmented_data = segmented_data
        self.features_data = features_data
        self.image_shape = image_shape

    def visualize(self):
        segmented_image = self.segmented_data.reshape(self.image_shape)
        plt.figure(figsize=(10, 8))
        plt.imshow(segmented_image, cmap='magma')
        plt.title('Detección de fuego')
        plt.axis('off')
        plt.show()

    def save_mask(self, output_path):
        segmented_image = self.segmented_data.reshape(self.image_shape)
        plt.imsave(output_path, segmented_image, cmap='gray')


class Convergence_Visualizer:
    def __init__(self, fitness_history):
        self.fitness_history = fitness_history

    def visualize(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('Convergencia del Algoritmo Genético')
        plt.xlabel('Generación')
        plt.ylabel('Mejor Entropía de Kapur')
        plt.grid()
        plt.show()

    def save_convergence(self, output_path):
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history)
        plt.title('Convergencia del Algoritmo Genético')
        plt.xlabel('Generación')
        plt.ylabel('Mejor Entropía de Kapur')
        plt.grid()
        plt.savefig(output_path)


# ==================================================
# Implementación del modelo completo
# ==================================================
if __name__ == "__main__":
    ruta = '/home/brauliosg/Documents/Mexico/FIRE/update_0/chihuahua/20220513#'
    bandas = ['B4', 'B5', 'B6', 'B7']

    band_data = []
    image_shape = None

    print("1. Cargando y corrigiendo datos TOA...")
    for banda in bandas:
        band = Band(banda, ruta)
        band.load()
        band.MTL_load()
        rc = Radiometric_correction(band)
        data_corr = rc.apply_radiometric_correction()
        if image_shape is None:
            image_shape = data_corr.shape
        band_data.append(data_corr.flatten())

    print("2. Calculando 7 características para el AG...")
    eps_val = 1e-6
    F1 = band_data[3] / (band_data[1] + eps_val)
    F2 = band_data[3] / (band_data[2] + eps_val)
    F3 = band_data[3] - band_data[1]
    F4 = band_data[3]
    F5 = band_data[2]
    F6 = band_data[3] / (band_data[0] + eps_val)
    F7 = (band_data[1] - band_data[3]) / (band_data[1] + band_data[3] + eps_val)

    features_data = np.stack([F1, F2, F3, F4, F5, F6, F7], axis=1)
    features_data = np.nan_to_num(features_data, nan=0.0, posinf=0.0, neginf=0.0)

    print("3. Iniciando Algoritmo Genético...")
    ga = GeneticAlgorithm(pop_size=50, generations=70)
    best_thresholds = ga.run(features_data)  # CORRECCIÓN: se usa run(), no apply_genetic_algorithm

    print("\nProceso finalizado. Umbrales óptimos encontrados:")
    print(best_thresholds)

    print("4. Generando resultados visuales...")
    segmenter = Segmenter(best_thresholds)
    segmented = segmenter.segment(features_data)

    vis = Visualizer(segmented, features_data, image_shape)
    vis.visualize()

    conv_vis = Convergence_Visualizer(ga.fitness_history)
    conv_vis.visualize()

    print("5. Guardando resultados...")
    output_mask = os.path.join(ruta, 'segmented_mask.png')
    output_convergence = os.path.join(ruta, 'convergence.png')
    vis.save_mask(output_mask)
    conv_vis.save_convergence(output_convergence)
