import os
import glob
import numpy as np
import rasterio as rio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import csv

# ---------------------------
# CONFIGURACIÓN
# ---------------------------

root = "/mnt/wwn-0x5000c500fad8a04f-part2/Mexico/FIRE/prueba"
root_prev = "/mnt/wwn-0x5000c500fad8a04f-part2/Mexico/FIRE/previos"

# Umbral en reflectancia real 
UMBRAL_B7 = 0.37

# Dilataciones independientes
DILATACION_B7 = 5
DILATACION_MASK = 7



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
        if not ruta_mtl:
            raise FileNotFoundError("No se encontró archivo MTL")
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


# ---------------------------
# FUNCIONES
# ---------------------------

def cargar_mask(dia_path):

    mask = None
    transform = None
    crs = None

    mask_paths = (
        glob.glob(os.path.join(dia_path, "*detection*.tif")) +
        glob.glob(os.path.join(dia_path, "*correction*.tif"))
    )

    if mask_paths:
        with rio.open(mask_paths[0]) as src:
            mask = src.read(1)
            transform = src.transform
            crs = src.crs

    return mask, transform, crs


def cargar_band7_corregida(dia_path):

    band = Band("B7", dia_path)
    band.load()
    band.MTL_load()

    corrected = Radiometric_correction(band).apply_radiometric_correction()

    return corrected, band.meta["transform"], band.meta["crs"]


def reproyectar(src_array, src_transform, src_crs,
                dst_shape, dst_transform, dst_crs,
                resampling):

    dst = np.zeros(dst_shape, dtype=src_array.dtype)

    reproject(
        source=src_array,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=resampling
    )

    return dst




#Main

# empezamos con root 

for estado in os.listdir(root):
    estado_path = os.path.join(root, estado)
    if not os.path.isdir(estado_path):
        continue

    for dia in os.listdir(estado_path):
        dia_path = os.path.join(estado_path, dia)
        if not os.path.isdir(dia_path):
            continue

        ID = dia_path[-5:]
        print(f"Procesando {estado} - {ID}...")
        
        # Cargamos las macara actual y la banda 7 
        mask_actual, mask_transform, mask_crs = cargar_mask(dia_path)
        if mask_actual is None:
            print(f"  No se encontró máscara para {ID}, saltando...")
            continue

        band7_corr, band7_transform, band7_crs = cargar_band7_corregida(dia_path)

        
        print(' Banda 7 actual cargada y corregida, mascara actual cargada.')
        
        # ahora cargamos la mascara previa y la banda 7 previa para el mismo ID
        for carpeta_prev in os.listdir(root_prev):
            if ID in carpeta_prev: 
                dia_prev_path = os.path.join(root_prev, carpeta_prev)
                if not os.path.isdir(dia_prev_path):
                    continue

                bands_prev = []
                masks_prev = []

                for dia_prev in os.listdir(dia_prev_path):
                    dia_prev_path_full = os.path.join(dia_prev_path, dia_prev)
                    if not os.path.isdir(dia_prev_path_full):
                        continue

                    # cargamos la banda 7 previa y la mascara previa
                    print(f"--- Cargando el historico de banda7 de {dia_prev}...")
                    band7_prev, band7_prev_transform, band7_prev_crs = cargar_band7_corregida(dia_prev_path_full)
                    # las reproyectamos a la misma resolución y sistema de coordenadas que la banda 7 actual
                    band7_prev_reproj = reproyectar(
                        band7_prev, band7_prev_transform, band7_prev_crs,
                        band7_corr.shape, band7_transform, band7_crs,
                        Resampling.bilinear
                    )
                    bands_prev.append((band7_prev_reproj, band7_transform, band7_crs))


                    print(f"--- Cargamos las mascaras previas de {dia_prev}...")
                    mask_prev, mask_prev_transform, mask_prev_crs = cargar_mask(dia_prev_path_full)
                    # las reproyectamos a la misma resolución y sistema de coordenadas que la mascara actual
                    mask_prev_reproj = reproyectar(
                        mask_prev, mask_prev_transform, mask_prev_crs,
                        mask_actual.shape, mask_transform, mask_crs,
                        Resampling.nearest
                    )   
                    masks_prev.append((mask_prev_reproj, mask_transform, mask_crs))

                


                if not bands_prev or not masks_prev:
                    print(f"  No hay históricos suficientes para el ID {ID}.")
                    continue

                # Sumamos todas las mascaras previas reproyectadas para obtener una mascara combinada
                mask_combinada = np.zeros_like(mask_actual, dtype=np.uint8)
                for mask_prev_reproj, _, _ in masks_prev:
                    mask_combinada = np.logical_or(mask_combinada, mask_prev_reproj > 0)
                mask_combinada = mask_combinada.astype(np.uint8)

                # Calculamos la mediana de las bandas 7 previas
                band7_prev_median = np.median(
                    np.array([band for band, _, _ in bands_prev]),
                    axis=0
                )

                # plt.figure(figsize=(15,15))
                # plt.imshow(band7_prev_median, cmap='seismic')
                # plt.colorbar()
                # plt.show()


                # Aplicamos la dilatación a la mascara combinada
                mask_dilatada = binary_dilation(mask_combinada, iterations=DILATACION_MASK).astype(np.uint8)

                # dilatamos la banda 7 previa mediana para cubrir posibles desplazamientos
                band7_prev_median_dilatada = binary_dilation(band7_prev_median > UMBRAL_B7, iterations=DILATACION_B7).astype(np.uint8)

                # con la mascara dilatada hacemos un and con la mascara actual, si ambos son 1 se elimina el pixel de la mascara actual
                
                
                mascara_final = np.logical_and(
                    mask_actual > 0,
                    np.logical_not(mask_dilatada)
                )

                # con la banda 7 previa mediana dilatada hacemos un and con la mascara actual, si ambos son 1 se elimina el pixel de la mascara actual
                mascara_final = np.logical_and(
                    mascara_final,
                    np.logical_not(band7_prev_median_dilatada)
                )

                # # Superposición simple: banda 7 corregida + máscara dilatada
                # plt.figure(figsize=(10,10))
                # plt.imshow(band7_corr, cmap='gray')  # imagen base
                # plt.imshow(band7_prev_median_dilatada, cmap='Reds', alpha=0.5)  # máscara semitransparente
                # plt.title('Band7 Corr + Band7 Prev Dilatada')
                # plt.axis('off')
                # plt.show()

                # # Comparación de máscaras
                # fig, axes = plt.subplots(1,3, figsize=(18,6))

                # # Máscara dilatada
                # axes[0].imshow(band7_corr, cmap='gray')  # base
                # axes[0].imshow(mask_dilatada, cmap='Blues', alpha=0.5)  # máscara semitransparente
                # axes[0].set_title('Mask Dilatada')
                # axes[0].axis('off')

                # # Band7 prev median dilatada
                # axes[1].imshow(band7_corr, cmap='gray')  # base
                # axes[1].imshow(band7_prev_median_dilatada, cmap='Reds', alpha=0.5)
                # axes[1].set_title('Band7 Prev Median Dilatada')
                # axes[1].axis('off')

                # # Comparación de todas las máscaras
                # axes[2].imshow(band7_corr, cmap='gray')  # base
                # axes[2].imshow(mask_dilatada, cmap='Blues', alpha=0.4)
                # axes[2].imshow(band7_prev_median_dilatada, cmap='Reds', alpha=0.4)
                # axes[2].imshow(mask_actual, cmap='Greens', alpha=0.4)
                # axes[2].set_title('Comparación de máscaras')
                # axes[2].axis('off')

                # plt.tight_layout()
                # plt.show()
                # # guardamos la mascara final corregida en la misma carpeta del dia actual con el nombre "Mask_corrrected_ID.tif"
                

                print('-' * 50)
                print("4. Guardando máscara de detección corregida ...")

                # Convertimos 0/1 a 0/255
                mascara_final = (mascara_final * 255).astype(np.uint8)
                
                # Pixeles originales
                total_mask_actual = np.sum(mask_actual > 0)

                # Pixeles eliminados por mascaras previas (solo donde mask_actual es 1)
                total_eliminados_prev = np.sum(np.logical_and(mask_actual > 0, mask_dilatada > 0))

                # Pixeles eliminados por banda 7 (solo donde mask_actual sigue siendo 1 después de previas)
                total_eliminados_b7 = np.sum(np.logical_and(mask_actual > 0, np.logical_and(np.logical_not(mask_dilatada), band7_prev_median_dilatada > 0)))

                # Pixeles que quedan en la mascara final
                total_final = np.sum(mascara_final > 0)

                comision_error = total_eliminados_b7/(np.sum(mask_actual>0) - np.sum(mascara_final>0))

                


                # Guardamos el csv 
                print(' --- Guardando datos en el CSV ')

                estadisticas = ["ID", "Pixeles detectados", "Pixeles corregido por mascaras (persistencia temporal)", "Pixeles corregidos por brillo en banda 7(falsa alarma)","pixeles finales", "comission Error"]
                row = [ID, total_mask_actual,total_eliminados_prev, total_eliminados_b7, total_final, comision_error]

                estadisticas_path = os.path.join(root,'estadisticas.csv')

                file_exists = os.path.isfile(estadisticas_path)

                with open(estadisticas_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists: 
                        writer.writerow(estadisticas)
                    writer.writerow(row)


                # Definimos ruta de salida
                output_path = os.path.join(dia_path, f"Mask_corrected_{ID}.tif")

                # Creamos metadata basada en la máscara actual
                out_meta = {
                    "driver": "GTiff",
                    "height": mascara_final.shape[0],
                    "width": mascara_final.shape[1],
                    "count": 1,
                    "dtype": rio.uint8,
                    "crs": mask_crs,
                    "transform": mask_transform,
                    "compress": "lzw"
                }

                with rio.open(output_path, "w", **out_meta) as dst:
                    dst.write(mascara_final, 1)

                print(f"Máscara guardada en: {output_path}")

