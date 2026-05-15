import os
import numpy as np
import pandas as pd
import rasterio as rio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import cv2  # Para redimensionar si los tamaños no coinciden

def calcular_metricas_escena():
    root_dir = '/mnt/wwn-0x5000c500fad8a04f-part2/Mexico/FIRE/prueba'
    csv_file = os.path.join(root_dir, 'metricas_final_fire_ROC.csv')
    
    # Preparar lista para almacenar resultados y crear el DataFrame al final
    resultados = []

    # 1. Obtener Estados
    estados = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    for estado in estados:
        path_estado = os.path.join(root_dir, estado)
        fechas = [d for d in os.listdir(path_estado) if os.path.isdir(os.path.join(path_estado, d))]
        
        for folder_name in fechas:
            path_fecha = os.path.join(path_estado, folder_name)
            
            # Extraer ID y Fecha (Ejemplo: 20240501_F45)
            parts = folder_name.split('_')
            if len(parts) < 2: continue
            fecha, id_escena = parts[0], parts[1]
            
            # Rutas de archivos
            path_heat = os.path.join(path_fecha, f'HeatMap_5d_{id_escena}.tif')
            path_det  = os.path.join(path_fecha, f'Mask_corrected_relaxed_{id_escena}.tif')
            path_true = os.path.join(path_fecha, f'TrueMask_{id_escena}.tif')
            
            if not all(os.path.isfile(p) for p in [path_heat, path_det, path_true]):
                print(f"Faltan archivos para ID {id_escena}, saltando...")
                continue

            # --- Carga de Datos ---
            with rio.open(path_heat) as src: heat_map = src.read(1).astype(float)
            with rio.open(path_det) as src: det_mask = (src.read(1) > 127).astype(np.uint8)
            with rio.open(path_true) as src: true_mask = (src.read(1) > 0).astype(np.uint8)

            # Redimensionar si es necesario (equivalente a imresize de MATLAB)
            if heat_map.shape != true_mask.shape:
                heat_map = cv2.resize(heat_map, (true_mask.shape[1], true_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                det_mask = cv2.resize(det_mask, (true_mask.shape[1], true_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            # --- 1. Cálculo de la Curva ROC ---
            y_true = true_mask.flatten()
            y_scores = heat_map.flatten()
            y_pred = det_mask.flatten()

            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)

            # --- 2. Matriz de Confusión (Punto GA) ---
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            eps = 1e-8
            precision = tp / (tp + fp + eps)
            recall = tp / (tp + fn + eps)
            f1 = 2 * (precision * recall) / (precision + recall + eps)
            iou = tp / (tp + fp + fn + eps)
            fpr_point = fp / (fp + tn + eps)

            # Guardar datos en la lista
            resultados.append([estado, fecha, id_escena, roc_auc, precision, recall, f1, iou, tp, fp, fn, tn])

            # --- 3. Generar Gráfica ---
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], color='black', linestyle='--')
            plt.plot(fpr_point, recall, 'ro', markersize=10, label='Punto Final (GA + Filtros)')
            
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate (FPR)')
            plt.ylabel('True Positive Rate (TPR / Recall)')
            plt.title(f'Detección de Incendios - {estado} (ID: {id_escena})')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Guardar gráfica
            plt.savefig(os.path.join(path_fecha, f'ROC_Plot_{id_escena}.png'))
            plt.close()
            print(f'Escena {id_escena} procesada. AUC: {roc_auc:.4f}')

    # --- 4. Guardar CSV Final ---
    df = pd.DataFrame(resultados, columns=['estado','fecha','id','AUC','precision','recall','f1','iou','TP','FP','FN','TN'])
    df.to_csv(csv_file, index=False)
    print(f"\n✅ PROCESO COMPLETADO.\nArchivo de métricas: {csv_file}")

if __name__ == "__main__":
    calcular_metricas_escena()