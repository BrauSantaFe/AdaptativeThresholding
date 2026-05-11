import rasterio as rio
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import gc
from typing import Optional, List, Tuple


class Band:
    def __init__(self, banda: str, ruta: str) -> None:
        self.banda: str = banda
        self.ruta: str = ruta
        self.data: Optional[np.ndarray] = None
        self.meta: Optional[dict] = None
        self.reflectance_mult: Optional[float] = None
        self.reflectance_add: Optional[float] = None
        self.sun_elevation: Optional[float] = None
        
    def load(self) -> np.ndarray:
        ruta_banda: List[str] = glob.glob('**/*' + self.banda + '*.TIF',
                                          root_dir=self.ruta, recursive=True)
        if not ruta_banda:
            raise FileNotFoundError(f"No se encontró {self.banda}")
        ruta_banda = os.path.join(self.ruta, ruta_banda[0])
        with rio.open(ruta_banda) as src:
            self.data = src.read(1).astype(np.float32)
            self.meta = src.meta
        return self.data

    def MTL_load(self) -> None:
        ruta_mtl: List[str] = glob.glob('**/*MTL.txt',
                                        root_dir=self.ruta, recursive=True)
        if not ruta_mtl:
            raise FileNotFoundError("No se encontró archivo MTL")
        ruta_mtl = os.path.join(self.ruta, ruta_mtl[0])
        band_number: str = self.banda.replace('B', '')
        with open(ruta_mtl) as f:
            for line in f:
                line = line.strip()
                if f'REFLECTANCE_MULT_BAND_{band_number}' in line:
                    self.reflectance_mult = float(line.split('=')[1])
                elif f'REFLECTANCE_ADD_BAND_{band_number}' in line:
                    self.reflectance_add = float(line.split('=')[1])
                elif 'SUN_ELEVATION' in line:
                    self.sun_elevation = float(line.split('=')[1])
        if self.reflectance_mult is None or self.reflectance_add is None:
            raise ValueError("Faltan coeficientes de reflectancia")
        if self.sun_elevation is None:
            raise ValueError("Falta SUN_ELEVATION")


class Radiometric_correction:
    def __init__(self, band: Band) -> None:
        self.band: Band = band

    def apply_radiometric_correction(self) -> np.ndarray:
        rho: np.ndarray = self.band.reflectance_mult * self.band.data + self.band.reflectance_add
        return rho / np.sin(np.deg2rad(self.band.sun_elevation))


class GeneticAlgorithm:
    def __init__(self, pop_size: int = 50, generations: int = 100,
                 pc: float = 0.8, pm: float = 0.2, elite_size: int = 4,
                 sample_fraction: float = 0.01, min_sample: int = 65536) -> None:

        self.pop_size: int = pop_size
        self.generations: int = generations
        self.pc: float = pc
        self.pm: float = pm
        self.elite_size: int = elite_size
        self.sample_fraction: float = sample_fraction
        self.min_sample: int = min_sample

        self.LB: np.ndarray = np.array([0.5, 0.5,  0.1, 0.05, 1.0])
        self.UB: np.ndarray = np.array([3.5, 3.5,  1.0, 1.0, 5.0])
        self.n_vars: int = len(self.LB)

        self.fitness_history: List[float] = []

    def initialize_population(self) -> np.ndarray:
        return self.LB + (self.UB - self.LB) * np.random.rand(self.pop_size, self.n_vars)

    def evaluate_kapur(self, thresholds: np.ndarray, features_sample: np.ndarray) -> float:
        eps: float = 1e-12
        total_entropy: float = 0.0
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

    def apply_selection(self, pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        new_pop = np.zeros_like(pop)
        elite = np.argsort(fitness)[-self.elite_size:]
        new_pop[:self.elite_size] = pop[elite]
        for i in range(self.elite_size, self.pop_size):
            a, b = np.random.choice(self.pop_size, 2, replace=False)
            new_pop[i] = pop[a] if fitness[a] > fitness[b] else pop[b]
        return new_pop

    def apply_crossover(self, pop: np.ndarray) -> np.ndarray:
        new_pop = pop.copy()
        for i in range(self.elite_size, self.pop_size, 2):
            if i+1 < self.pop_size and np.random.rand() < self.pc:
                a = np.random.rand()
                p1, p2 = pop[i], pop[i+1]
                new_pop[i] = a*p1 + (1-a)*p2
                new_pop[i+1] = a*p2 + (1-a)*p1
        return new_pop

    def apply_mutation(self, pop: np.ndarray) -> np.ndarray:
        for i in range(self.elite_size, self.pop_size):
            for j in range(self.n_vars):
                if np.random.rand() < self.pm:
                    pop[i,j] += np.random.normal(0, 0.2*(self.UB[j]-self.LB[j]))
                    pop[i,j] = np.clip(pop[i,j], self.LB[j], self.UB[j])
        return pop

    def run(self, X: np.ndarray) -> np.ndarray:
        N: int = X.shape[0]
        Ns: int = min(max(self.min_sample, int(N*self.sample_fraction)), N)
        Xs: np.ndarray = X[np.random.choice(N, Ns, replace=False)]

        pop: np.ndarray = self.initialize_population()
        best: Optional[np.ndarray] = None
        best_fit: float = -np.inf

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


class Segmenter5d:
    def __init__(self, thresholds: np.ndarray) -> None:
        self.thresholds: np.ndarray = thresholds

    def segment(self, features_data: np.ndarray) -> np.ndarray:
        F = features_data
        T = self.thresholds
        mask = (((F[:, 0] > T[0]) | (F[:, 4] > T[4])) &
                (F[:, 1] > T[1]) &
                ((F[:, 2] > 0.8 * T[2]) | (F[:, 3] > 0.5 * T[3])))
        return mask.astype(np.uint8)


# ------------------------------------------------3-------
# MAIN

if __name__ == "__main__":
    # Apuntamos a la raíz de la actualización
    root_update = "/mnt/wwn-0x5000c500fad8a04f-part2/Mexico/FIRE/previos"
    bandas = ['B4', 'B5', 'B6', 'B7']

    # 1. Iterar sobre cada estado (aguascalientes, chihuahua, etc.)
    for estado in os.listdir(root_update):
        ruta_estado = os.path.join(root_update, estado)
        
        if not os.path.isdir(ruta_estado):
            continue
            
        print(f"\n" + "="*60)
        print(f"PROCESANDO ESTADO: {estado.upper()}")
        print("="*60)
        ID = ruta_estado.split('_')[-1]  # Extraemos el ID del estado

        # 2. Iterar sobre cada carpeta de fecha/ID (ej: 20200613_FXX62)
        for subdir in os.listdir(ruta_estado):
            ruta = os.path.join(ruta_estado, subdir)

            if not os.path.isdir(ruta):
                continue

            # Extraemos el ID dinámicamente (lo que esté después del último '_')
            # Para '20200613_FXX62' devolverá 'FXX62'
            #ID = subdir.split('_')[-1] if '_' in subdir else subdir

            palabras_clave = []

            # Verificar si ya se procesó esta carpeta
            if any(
                any(p in archivo for p in palabras_clave)
                for archivo in os.listdir(ruta)
            ):
                print(f"--- Salto: Resultados ya existen para ID {ID} en {estado}")
                continue

            print(f"\n>> Iniciando proceso - Estado: {estado} | ID: {ID}")
            
            try:
                # --- CARGA Y CORRECCIÓN ---
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

                # --- EXTRACCIÓN DE CARACTERÍSTICAS (5D) ---
                eps = 1e-6
                F1 = band_data[3] / (band_data[1] + eps) # B7/B5
                F2 = band_data[3] / (band_data[2] + eps) # B7/B6
                F3 = band_data[3]                        # B7
                F4 = band_data[2]                        # B6
                F5 = band_data[3] / (band_data[0] + eps) # B7/B4

                F5d = np.nan_to_num(np.stack([F1, F2, F3, F4, F5], axis=1))

                # Límites del espacio de búsqueda (Asegúrate que coincidan con tus 5 variables)
                LB = np.array([0.5, 0.5, 0.1, 0.05, 1.0])
                UB = np.array([3.5, 3.5, 1.0, 1.0, 5.0])

                # --- ALGORITMO GENÉTICO ---
                print(f"Ejecutando GA para {ID}...")
                ga5 = GeneticAlgorithm(generations=100)
                ga5.LB = LB
                ga5.UB = UB
                ga5.n_vars = 5

                T5 = ga5.run(F5d)
                mask5 = Segmenter5d(T5).segment(F5d).reshape(image_shape)

                # --- GUARDADO ---
                meta = band.meta.copy()
                meta.update(dtype=rio.uint8, count=1)
                mask5_uint8 = (mask5 * 255).astype(np.uint8)

                # Definir nombre de salida según el contexto
                prefix = "False_alarm_correction" if 'previos' in ruta else "Active_fire_detection"
                out_path = os.path.join(ruta, f"{prefix}_5d_2_{ID}.tif")

                with rio.open(out_path, "w", **meta) as dst:
                    dst.write(mask5_uint8, 1)

                print(f"ÉXITO: Máscara guardada en {out_path}")

            except Exception as e:
                print(f"ERROR procesando {ID} en {estado}: {e}")

            finally:
                # Limpieza agresiva de memoria incluso si hay error
                if 'band_data' in locals(): del band_data
                if 'F5d' in locals(): del F5d
                if 'mask5' in locals(): del mask5
                gc.collect()

    print("\nPROCESAMIENTO GLOBAL FINALIZADO.")