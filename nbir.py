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
        # Buscamos el archivo TIF de la banda específica
        ruta_banda = glob.glob(f'**/*{self.banda}*.TIF',
                               root_dir=self.ruta, recursive=True)
        if not ruta_banda:
            raise FileNotFoundError(f"No se encontró {self.banda} en {self.ruta}")
        
        ruta_completa = os.path.join(self.ruta, ruta_banda[0])
        with rio.open(ruta_completa) as src:
            self.data = src.read(1).astype(np.float64)
            self.meta = src.meta
        return self.data

    def MTL_load(self):
        # Buscamos el archivo de metadatos MTL
        ruta_mtl = glob.glob('**/*MTL.txt',
                             root_dir=self.ruta, recursive=True)
        if not ruta_mtl:
            raise FileNotFoundError(f"No se encontró archivo MTL en {self.ruta}")
        
        ruta_completa_mtl = os.path.join(self.ruta, ruta_mtl[0])
        band_number = self.banda.replace('B', '')
        
        with open(ruta_completa_mtl) as f:
            for line in f:
                if f'REFLECTANCE_MULT_BAND_{band_number}' in line:
                    self.reflectance_mult = float(line.split('=')[1].strip())
                elif f'REFLECTANCE_ADD_BAND_{band_number}' in line:
                    self.reflectance_add = float(line.split('=')[1].strip())

class Radiometric_correction:
    def __init__(self, band):
        self.band = band

    def apply_radiometric_correction(self):
        if self.band.reflectance_mult is None or self.band.reflectance_add is None:
            raise ValueError(f"Band {self.band.banda} no tiene parámetros de reflectancia")
        
        # Fórmula: REFLECTANCE = MULT * DN + ADD
        corrected = self.band.reflectance_mult * self.band.data + self.band.reflectance_add
        return corrected

def normalize_to_8bit(arr):
    """
    Convierte datos de reflectancia a 0-255 con estiramiento de contraste (2%-98%).
    """
    # 1. Limpiar valores no válidos (NaN o Infinitos)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    
    # 2. Estiramiento de contraste (Contrast Stretch)
    # Ignoramos el 2% de los píxeles más oscuros y el 2% de los más brillantes
    p2, p98 = np.percentile(arr[arr > 0], [2, 98]) 
    
    # 3. Escalar a 0 - 255
    arr_rescaled = np.clip(arr, p2, p98)
    arr_rescaled = ((arr_rescaled - p2) / (p98 - p2 + 1e-6)) * 255
    
    return arr_rescaled.astype(np.uint8)

# ==================================================
import tifffile

# ==================================================
# PROCESAMIENTO PRINCIPAL
if __name__ == "__main__":

    root = '/home/brauliosg/Documents/Mexico/FIRE/update_0'

    if not os.path.exists(root):
        print(f"Error: La ruta {root} no existe.")
    else:
        estados = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])[::]

        for estado in estados:
            ruta_estado = os.path.join(root, estado)
            print(f"\nProcesando estado: {estado}")

            dias = sorted([
                d for d in os.listdir(ruta_estado)
                if os.path.isdir(os.path.join(ruta_estado, d))
            ])

            for dia in dias:
                ruta_dia = os.path.join(ruta_estado, dia)
                print(f"  Día: {dia}")

                try:
                    # Instanciar bandas necesarias para Falso Color (B7, B5, B4)
                    band_r = Band('B7', ruta_dia)
                    band_g = Band('B5', ruta_dia)
                    band_b = Band('B4', ruta_dia)

                    # Cargar datos y metadatos MTL
                    for b in [band_r, band_g, band_b]:
                        b.load()
                        b.MTL_load()

                    # Aplicar corrección radiométrica (TOA Reflectance)
                    R = Radiometric_correction(band_r).apply_radiometric_correction()
                    G = Radiometric_correction(band_g).apply_radiometric_correction()
                    B = Radiometric_correction(band_b).apply_radiometric_correction()

                    # Normalizar a 8 bits para visualización
                    R_img = normalize_to_8bit(R)
                    G_img = normalize_to_8bit(G)
                    B_img = normalize_to_8bit(B)

                    # Crear array RGB interleaved (height, width, 3)
                    rgb_array = np.stack([R_img, G_img, B_img], axis=-1)

                    # Guardar con tifffile como RGB
                    output_file = os.path.join(ruta_dia, "RGB_composite.tif")
                    tifffile.imwrite(output_file, rgb_array, photometric='rgb')
                    
                    print(f"    ✅ Imagen guardada: {output_file}")

                except Exception as e:
                    print(f"    ❌ Error procesando {dia}: {e}")
