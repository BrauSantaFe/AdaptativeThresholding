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
        self.reflectance_mult = None
        self.reflectance_add = None
        
    def load(self):
        ruta_banda = glob.glob('**/*' + self.banda + '*.TIF',
                               root_dir=self.ruta, recursive=True)
        if not ruta_banda:
            raise FileNotFoundError(f"No se encontró {self.banda}")
        ruta_banda = os.path.join(self.ruta, ruta_banda[0])
        with rio.open(ruta_banda) as src:
            self.data = src.read(1).astype(np.float64)
            self.meta = src.meta
        return self.data

    def MTL_load(self):
        ruta_mtl = glob.glob('**/*MTL.txt',
                             root_dir=self.ruta, recursive=True)
        ruta_mtl = os.path.join(self.ruta, ruta_mtl[0])
        band_number = self.banda.replace('B', '')
        with open(ruta_mtl) as f:
            for line in f:
                if f'REFLECTANCE_MULT_BAND_{band_number}' in line:
                    self.reflectance_mult = float(line.split('=')[1])
                elif f'REFLECTANCE_ADD_BAND_{band_number}' in line:
                    self.reflectance_add = float(line.split('=')[1])


class Radiometric_correction:
    def __init__(self, band):
        self.band = band

    def apply_radiometric_correction(self):
        return self.band.reflectance_mult * self.band.data + self.band.reflectance_add


# ==================================================
# Genetic Algorithm

class GeneticAlgorithm:
    def __init__(self, pop_size=50, generations=100,
                 pc=0.7, pm=0.2, elite_size=2,
                 sample_fraction=0.01, min_sample=65536):

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
        eps = 1e-12
        total_entropy = 0
        for i in range(self.n_vars):
            hist, bins = np.histogram(features_sample[:, i], bins=256,
                                      range=(self.LB[i], self.UB[i]))
            p = hist / (np.sum(hist) + eps)
            idx = np.searchsorted(bins, thresholds[i]) - 1
            idx = np.clip(idx, 1, 255)
            w0, w1 = np.sum(p[:idx]), np.sum(p[idx:])
            if w0 > eps and w1 > eps:
                p0, p1 = p[:idx]/w0, p[idx:]/w1
                h0 = -np.sum(p0[p0 > 0] * np.log(p0[p0 > 0] + eps))
                h1 = -np.sum(p1[p1 > 0] * np.log(p1[p1 > 0] + eps))
                total_entropy += (h0 + h1)
        return total_entropy

    def apply_selection(self, pop, fitness):
        new_pop = np.zeros_like(pop)
        elite = np.argsort(fitness)[-self.elite_size:]
        new_pop[:self.elite_size] = pop[elite]
        for i in range(self.elite_size, self.pop_size):
            a, b = np.random.choice(self.pop_size, 2, replace=False)
            new_pop[i] = pop[a] if fitness[a] > fitness[b] else pop[b]
        return new_pop

    def apply_crossover(self, pop):
        new_pop = pop.copy()
        for i in range(self.elite_size, self.pop_size, 2):
            if i+1 < self.pop_size and np.random.rand() < self.pc:
                a = np.random.rand()
                p1, p2 = pop[i], pop[i+1]
                new_pop[i] = a*p1 + (1-a)*p2
                new_pop[i+1] = a*p2 + (1-a)*p1
        return new_pop

    def apply_mutation(self, pop):
        for i in range(self.elite_size, self.pop_size):
            for j in range(self.n_vars):
                if np.random.rand() < self.pm:
                    pop[i,j] += np.random.normal(0, 0.1*(self.UB[j]-self.LB[j]))
                    pop[i,j] = np.clip(pop[i,j], self.LB[j], self.UB[j])
        return pop

    def run(self, X):
        N = X.shape[0]
        Ns = min(max(self.min_sample, int(N*self.sample_fraction)), N)
        Xs = X[np.random.choice(N, Ns, replace=False)]

        pop = self.initialize_population()
        best, best_fit = None, -np.inf

        for g in range(self.generations):
            fitness = np.array([self.evaluate_kapur(ind, Xs) for ind in pop])
            self.fitness_history.append(fitness.max())
            if fitness.max() > best_fit:
                best_fit = fitness.max()
                best = pop[np.argmax(fitness)].copy()
            pop = self.apply_mutation(
                    self.apply_crossover(
                    self.apply_selection(pop, fitness)))
            if g % 10 == 0:
                print(f"Gen {g} | Entropía máx: {fitness.max():.4f}")
        return best


class Segmenter7d:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def segment(self, features_data):
        F = features_data
        T = self.thresholds
        mask = (((F[:, 0] > T[0]) | (F[:, 5] > T[5]) | (F[:, 6] < T[6])) &
                (F[:, 1] > T[1]) &
                (F[:, 2] > T[2]) &
                ((F[:, 3] > 0.8 * T[3]) | (F[:, 4] > 0.5 * T[4])))
        return mask.astype(np.uint8)
  

# --------------------------------------------------------
# MAIN

if __name__ == "__main__":

    root = "/home/brauliosg/Documents/Mexico/FIRE/previos/20220411_FQL36_FVZ96_FED15"
    bandas = ['B4', 'B5', 'B6', 'B7']

    for subdir in os.listdir(root):
        ruta = os.path.join(root, subdir)

        if not os.path.isdir(ruta):
            continue

        ID = ruta[-5:]

        palabras_clave = ["Active_fire_detection", "False_alarm_correction"]

        if any(
            any(p in archivo for p in palabras_clave)
            for archivo in os.listdir(ruta)
        ):
            print(f"Archivo de salida ya existe para el ID {ID}, saltando...")
            continue   # ← ESTE sí salta al siguiente subdir

        # SOLO entra aquí si NO encontró archivos
        print('-' * 50)
        print(f"1. Cargando y corrigiendo bandas... ID: {ID}")

        band_data = []
        image_shape = None

        for b in bandas:
            band = Band(b, ruta)
            band.load()
            band.MTL_load()
            rc = Radiometric_correction(band)
            data = rc.apply_radiometric_correction()

            if image_shape is None:
                image_shape = data.shape

            band_data.append(data.flatten())

        print('-' * 50)
        print("2. Extrayendo características...")

        eps = 1e-6
        F1 = band_data[3] / (band_data[1] + eps)
        F2 = band_data[3] / (band_data[2] + eps)
        F3 = band_data[3] - band_data[1]
        F4 = band_data[3]
        F5 = band_data[2]
        F6 = band_data[3] / (band_data[0] + eps)
        F7 = (band_data[1] - band_data[3]) / (band_data[1] + band_data[3] + eps)

        F7d = np.nan_to_num(np.stack([F1, F2, F3, F4, F5, F6, F7], axis=1))
        F5d = np.nan_to_num(np.stack([F1, F2, F3, F6, F7], axis=1))
        F3d = np.nan_to_num(np.stack([F1, F2, F3], axis=1))

        # límites del espacio de búsqueda
        LB = np.array([0.5, 0.5, -0.5, 0.1, 0.05, 1.0, -0.8])
        UB = np.array([3.5, 3.5,  0.5, 1.0, 1.0, 5.0,  0.2])

        # -------- 7D --------
        print('-' * 50)
        print("3. Ejecutando algoritmo genético...")

        ga7 = GeneticAlgorithm(generations=50)
        ga7.LB = LB
        ga7.UB = UB
        ga7.n_vars = 7

        T7 = ga7.run(F7d)
        mask7 = Segmenter7d(T7).segment(F7d).reshape(image_shape)

        print(f"Los umbrales obtenidos para el ID {ID}: {T7}")

        # guardamos la máscara
        print('-' * 50)
        print("4. Guardando máscara de detección...")

        meta = band.meta.copy()
        meta.update(dtype=rio.uint8, count=1)

        # convertimos 0/1 a 0/255
        mask7_uint8 = (mask7 * 255).astype(np.uint8)

        if 'previos' in ruta:
            out_path = os.path.join(ruta, "False_alarm_correction.tif")
        else:
            out_path = os.path.join(ruta, "Active_fire_detection.tif")

        with rio.open(out_path, "w", **meta) as dst:
            dst.write(mask7_uint8, 1)

        print(f"Máscara guardada correctamente en {out_path}")
        print('-' * 50)
        print(f"Proceso completado para el ID: {ID}")
