# ---------------------------
import cv2

# Reproyectar / redimensionar la máscara previa
mask_prev_resized = cv2.resize(mask_prev.astype(np.uint8), 
                                (image_shape_actual[1], image_shape_actual[0]), 
                                interpolation=cv2.INTER_NEAREST)

# ---------------------------
# Corrección de falsas alarmas
mask_corrected = mask_actual.copy()
mask_corrected[mask_prev_resized==1] = 0

print(f"🔥 Píxeles de incendio originales: {mask_actual.sum()}")
print(f"🚫 Píxeles eliminados (falsas alarmas): {(mask_prev_resized==1).sum()}")
print(f"🔥 Píxeles de incendio finales: {mask_corrected.sum()}")

# ---------------------------
# Visualización
plt.imshow(mask_corrected, cmap="magma")
plt.title("Incendios corregidos"); plt.axis("off"); plt.show()


# guardamos el resultado
meta = band.meta
meta.update(dtype=rio.uint8, count=1)
with rio.open("incendios_corregidos.tif", "w", **meta) as dst:
    dst.write(mask_corrected.astype(rio.uint8), 1)



print("✔ Proceso finalizado correctamente.")

