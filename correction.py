import os
import numpy as np
import rasterio as rio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_dilation # Para dar tolerancia al filtrado

# ---------------------------
# CONFIGURACIÓN
# ---------------------------
PATHS = {
    "mask_actual": '/home/brauliosg/Documents/Mexico/FIRE/update_0/sinaloa/20220507_FWW3/ActiveFire_detection.tif',
    "prev_folder": '/home/brauliosg/Documents/Mexico/FIRE/previos/20220413_FWW3_FFV18_FCW35/band7',
    "output": '/home/brauliosg/Documents/Mexico/FIRE/mask_actual_filtered.tif'
}

REFLECTANCE_MULT = 1.210700
REFLECTANCE_ADD = -0.099980
UMBRAL_CORTE = 15000

def main():
    # 1. CARGAR MÁSCARA ACTUAL (Referencia Maestra)
    with rio.open(PATHS["mask_actual"]) as src_master:
        mask_actual = src_master.read(1)
        profile_master = src_master.profile
        # Forzamos estos parámetros para la reproyección de las históricas
        master_dims = (src_master.height, src_master.width)
        master_transform = src_master.transform
        master_crs = src_master.crs

    tif_files = sorted([os.path.join(PATHS["prev_folder"], f) 
                       for f in os.listdir(PATHS["prev_folder"]) 
                       if f.lower().endswith('.tif')])

    # 2. PROCESAR HISTÓRICAS ASEGURANDO COINCIDENCIA DE PÍXEL
    reprojected_images = []
    print(f"Alineando {len(tif_files)} imágenes a la máscara maestra...")

    for ruta in tif_files:
        with rio.open(ruta) as src:
            data = src.read(1).astype(np.float32)
            data_corr = (data * REFLECTANCE_MULT) + REFLECTANCE_ADD
            
            # Matriz vacía con el tamaño EXACTO de la máscara actual
            img_aligned = np.zeros(master_dims, dtype=np.float32)
            
            reproject(
                source=data_corr,
                destination=img_aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=master_transform,
                dst_crs=master_crs,
                resampling=Resampling.nearest, # Nearest mantiene mejor los valores de corte
                dst_nodata=0
            )
            reprojected_images.append(img_aligned)

    # 3. CÁLCULO DE MEDIANA
    median_prev = np.median(np.stack(reprojected_images), axis=0)

    # 4. CREAR MÁSCARA DE FALSAS ALARMAS (Donde siempre brilla > 2.2)
    # Si sospechas que hay un pequeño desplazamiento, dilatamos la zona de rechazo
    falsas_alarmas_zonas = (median_prev > UMBRAL_CORTE)
    
    # OPCIONAL: Dilatar 1 píxel para asegurar que atrape el borde de la falsa alarma
    # falsas_alarmas_zonas = binary_dilation(falsas_alarmas_zonas, structure=np.ones((3,3)))

    # 5. FILTRADO FINAL
    mask_actual_filtered = np.copy(mask_actual)
    # Aplicamos el filtro: Si la zona es detectada como fuego pero es una falsa alarma histórica -> 0
    mask_actual_filtered[falsas_alarmas_zonas] = 0

    # 6. GUARDAR
    with rio.open(PATHS["output"], 'w', **profile_master) as dst:
        dst.write(mask_actual_filtered.astype(profile_master['dtype']), 1)

    # Estadísticas de validación
    print(f"Píxeles en máscara original: {np.sum(mask_actual > 0)}")
    print(f"Píxeles eliminados por coincidencia temporal: {np.sum((mask_actual > 0) & falsas_alarmas_zonas)}")
    print(f"Píxeles restantes: {np.sum(mask_actual_filtered > 0)}")

if __name__ == "__main__":
    main()