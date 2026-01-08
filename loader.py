import rasterio as rio
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

class Band:
    """
    Representa una banda de un archivo raster
    Tiene dos principales atributos:
    Data: es el array numpy que contiene los datos de la banda raster.
    Meta: es un diccionario que contiene los metadatos para realizar la correccion radiométrica, asi como el max y min de la banda.
    Tiene dos principales metodos:
    Load: carga la banda raster desde un archivo dado.
    MTL_load : carga los metadatos desde un archivo MTL asociado al raster.
    """
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
        """
        Carga la banda raster desde un archivo dado.
        """
        ruta_banda = glob.glob(
            '**/*' + self.banda + '*.TIF',
            root_dir=self.ruta,
            recursive=True
        )
        if not ruta_banda:
            raise FileNotFoundError(
                f"No se encontró el archivo para la banda {self.banda} en la ruta {self.ruta}"
            )
        ruta_banda = os.path.join(self.ruta, ruta_banda[0])
        with rio.open(ruta_banda) as data_src:
            # Usamos float64 para igualar la precisión de MATLAB
            self.data = data_src.read(1).astype(np.float64)
            self.meta = data_src.meta
        return self.data

    def MTL_load(self):
        """
        Carga los metadatos desde un archivo MTL asociado al raster.
        """
        ruta_mtl = glob.glob('**/*MTL.txt', root_dir=self.ruta, recursive=True)
        if not ruta_mtl:
            raise FileNotFoundError(
                f"No se encontró el archivo MTL en la ruta {self.ruta}"
            )

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

        return (
            self.reflectance_max,
            self.reflectance_min,
            self.reflectance_mult,
            self.reflectance_add
        )


class Radiometric_correction:
    """
    Esta clase es independiente de Band y solo realiza la corrección radiométrica
    de acuerdo al manual Handbook para Landsat 8.

    Aplica la siguiente fórmula:
    Reflectance = Reflectance_mult * DN + Reflectance_add

    Retorna un array numpy con los valores corregidos radiométricamente.
    """
    def __init__(self, band: Band):
        self.band = band

    def apply_radiometric_correction(self):
        if self.band.data is None:
            raise ValueError('La banda no ha sido cargada previamente')

        if self.band.reflectance_mult is None or self.band.reflectance_add is None:
            raise ValueError('Los metadatos MTL no han sido cargados')

        corrected_data = (
            self.band.reflectance_mult * self.band.data
            + self.band.reflectance_add
        )
        return corrected_data


class GeneticAlgorithm:
    """
    Implementación optimizada de un Algoritmo Genético para optimizar umbrales
    utilizando la entropía de Kapur.
    """
    def __init__(
        self,
        pop_size=50,
        generations=100,
        pc=0.7,
        pm=0.2,
        elite_size=2,
        sample_fraction=0.01,
        min_sample=65536
    ):
        self.pop_size = pop_size
        self.generations = generations
        self.pc = pc
        self.pm = pm
        self.elite_size = elite_size
        self.sample_fraction = sample_fraction
        self.min_sample = min_sample

        # Límites corregidos: LB debe ser menor que UB
        # F1, F2, F3, F4, F5, F6, F7(NBR)
        self.LB = np.array([2.0, 2.0, 0.0, 0.2, 0.3, 2.0,-0.8])
        self.UB = np.array([4.0, 4.0, 0.5, 1.0, 1.0, 3.0, -0.4])

        self.n_vars = len(self.LB)
        self.fitness_history = []

    def initialize_population(self):
        return self.LB + (self.UB - self.LB) * np.random.rand(
            self.pop_size, self.n_vars
        )

    def calculate_kapur_fitness(self, thresholds, features_sample):
        """
        Calcula la entropía de Kapur 6D para un individuo.
        """
        total_entropy = 0
        n_bins = 256
        eps = 1e-12

        for i in range(self.n_vars):
            data = features_sample[:, i]
            hist, bin_edges = np.histogram(
                data, bins=n_bins, range=(self.LB[i], self.UB[i])
            )
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

    def apply_genetic_algorithm(self, features_data):
        """
        Aplica el algoritmo genético para encontrar los mejores umbrales.
        """
        N_total = features_data.shape[0]
        N_sample = min(
            max(self.min_sample, int(N_total * self.sample_fraction)),
            N_total
        )

        sample_idx = np.random.choice(N_total, N_sample, replace=False)
        features_sample = features_data[sample_idx]

        population = self.initialize_population()
        best_thresholds = None
        best_fitness = -np.inf

        for gen in range(self.generations):
            fitness = np.array([
                self.calculate_kapur_fitness(ind, features_sample)
                for ind in population
            ])

            idx_sorted = np.argsort(fitness)[::-1]

            if fitness[idx_sorted[0]] > best_fitness:
                best_fitness = fitness[idx_sorted[0]]
                best_thresholds = population[idx_sorted[0]].copy()

            new_population = np.zeros_like(population)
            new_population[:self.elite_size] = population[
                idx_sorted[:self.elite_size]
            ]

            for i in range(self.elite_size, self.pop_size, 2):
                def tournament():
                    c = np.random.randint(0, self.pop_size, 3)
                    return population[c[np.argmax(fitness[c])]]

                p1, p2 = tournament(), tournament()

                if np.random.rand() < self.pc:
                    cp = np.random.randint(1, self.n_vars)
                    c1 = np.concatenate([p1[:cp], p2[cp:]])
                    c2 = np.concatenate([p2[:cp], p1[cp:]])
                else:
                    c1, c2 = p1.copy(), p2.copy()

                for child in (c1, c2):
                    for j in range(self.n_vars):
                        if np.random.rand() < self.pm:
                            child[j] += np.random.normal(
                                0, 0.05 * (self.UB[j] - self.LB[j])
                            )
                    child[:] = np.clip(child, self.LB, self.UB)

                new_population[i] = c1
                if i + 1 < self.pop_size:
                    new_population[i + 1] = c2

            population = new_population
            self.fitness_history.append(best_fitness)

            if gen % 10 == 0:
                print(f"  Gen {gen} | Max Entropía: {best_fitness:.4f}")

        return best_thresholds



class Segmenter:
    """
    Realiza la segmentación final usando los umbrales optimizados.
    """
    def __init__(self, thresholds):
        self.thresholds = thresholds

        # Dentro de class Segmenter:
    def segment(self, features_data):
        F = features_data
        T = self.thresholds

        mask = (
            ((F[:, 0] > T[0]) | (F[:, 5] > T[5])) &   
            (F[:, 1] > T[1]) &
            (F[:, 2] > T[2]) &
            ((F[:, 3] > 0.8 * T[3]) | (F[:, 4] > 0.5 * T[4])) &
            (F[:, 6] < T[6]) # Esta condición capturará lo menor al umbral (ej. < -0.55)
        )
        return mask.astype(np.uint8)




class Visualizer:
    """
    Esta clase visualiza la máscara final de la segmentación, las características individuales
    y las características binarizadas con el umbral óptimo.
    """
    def __init__(self, segmented_data, features_data, image_shape, thresholds):
        self.segmented_data = segmented_data
        self.features_data = features_data
        self.image_shape = image_shape
        self.thresholds = thresholds
        self.feature_names = ['F1 (B7/B5)', 'F2 (B7/B6)', 'F3 (B7-B5)', 
                              'F4 (B7)', 'F5 (B6)', 'F6 (B7/B4)', 'F7 (NBR)']

    def visualize_thresholded_features(self):
        """
        Muestra cada feature aplicando su umbral óptimo.
        Permite ver qué está aportando cada característica a la decisión final.
        """
        for i in range(self.features_data.shape[1]):
            # Creamos una figura nueva para cada feature
            plt.figure(figsize=(14, 6))
            
            # --- Lado Izquierdo: Feature Original (Relieve mejorado) ---
            plt.subplot(1, 2, 1)
            feat_img = self.features_data[:, i].reshape(self.image_shape)
            
            # Normalización robusta para distinguir relieve en F1, F2 y F7
            vmin, vmax = np.nanpercentile(feat_img, [2, 98])
            
            im = plt.imshow(feat_img, cmap='viridis', vmin=vmin, vmax=vmax)
            plt.title(f'Original: {self.feature_names[i]}\n(Contraste 2%-98%)')
            plt.colorbar(im, fraction=0.046, pad=0.04, label='Valor de reflectancia/ratio')

            # --- Lado Derecho: Feature con Umbral Aplicado ---
            plt.subplot(1, 2, 2)
            
            # Lógica corregida: F7 (NBR) busca valores menores (<)
            # El resto de los features usualmente buscan valores mayores (>) para fuego
            if i == 6: # F7 es NBR
                thresh_img = (feat_img < self.thresholds[i]).astype(float)
                op = "<"
            else:
                thresh_img = (feat_img > self.thresholds[i]).astype(float)
                op = ">"
            
            plt.imshow(thresh_img, cmap='gray')
            plt.title(f'Umbral Aplicado: {self.feature_names[i]} {op} {self.thresholds[i]:.4f}')
            plt.axis('on') # Ejes activos para ver coordenadas de píxeles

            plt.tight_layout()
            print(f"Graficando {self.feature_names[i]}... Mueva el mouse sobre la imagen para ver info de píxeles.")
            plt.show()

    def visualize(self):
        """
        Visualiza la máscara final combinada de la segmentación.
        """
        segmented_image = self.segmented_data.reshape(self.image_shape)
        plt.figure(figsize=(10, 8))
        # Usamos magma para resaltar las zonas de fuego (blanco/amarillo sobre negro)
        im = plt.imshow(segmented_image, cmap='magma')
        plt.title('Máscara Final de Detección de Fuego\n(Combinación de todas las reglas)')
        plt.colorbar(im, label='0: No Fuego | 1: Fuego')
        plt.axis('off')
        plt.show()

    def save_mask(self, output_path):
        """
        Guarda la máscara final en formato imagen.
        """
        segmented_image = self.segmented_data.reshape(self.image_shape)
        plt.imsave(output_path, segmented_image, cmap='gray')

class Convergence_Visualizer:
    """
    Esta clase visualiza la convergencia del AG.
    """
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

    # 🔥 TODO ESTO VA DENTRO DEL MAIN
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
    features_data = np.nan_to_num(
        features_data, nan=0.0, posinf=0.0, neginf=0.0
    )

    print("3. Iniciando Algoritmo Genético...")
    ga = GeneticAlgorithm(
        pop_size=50,
        generations=70,
        LB=[2.0, 2.0, 0.0, 0.2, 0.3, 2.0, -0.8],
        UB=[4.0, 4.0, 0.5, 1.0, 1.0, 3.0, -0.4]
    )

    best_thresholds = ga.apply_genetic_algorithm(features_data)

    print("4. Generando máscara final")
    segmenter = Segmenter(best_thresholds)
    segmented = segmenter.segment(features_data)

    print("5. Visualizando resultados")
    vis = Visualizer(segmented, features_data, image_shape, best_thresholds)
    vis.visualize_thresholded_features()
    vis.visualize()

    conv_vis = Convergence_Visualizer(ga.fitness_history)
    conv_vis.visualize()

    print("6. Guardando resultados")
    output_mask = os.path.join(ruta, 'segmented_mask.png')
    output_convergence = os.path.join(ruta, 'convergence.png')
    vis.save_mask(output_mask)
    conv_vis.save_convergence(output_convergence)
