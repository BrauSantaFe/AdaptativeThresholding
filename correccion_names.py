import os
import glob 

if __name__ == "__main__":

    root = "/home/brauliosg/Documents/Mexico/FIRE/update_0"

    for estado in os.listdir(root):
        estado_path = os.path.join(root, estado)
        for fecha in os.listdir(estado_path):
            fecha_path = os.path.join(estado_path, fecha)
            ID = fecha_path[-5:]
            for documento in os.listdir(fecha_path):
                if str(ID) in documento:
                    continue  # Si el ID ya está en el nombre, saltamos

                if ("ActiveFire_detection.tif" in documento) or ("Active_fire_detection" in documento) \
                    or ("RGB_composite" in documento) or ("trueMask" in documento) or ("TrueMask" in documento):
                    # renombramos el archivo con el ID al final
                    new_name = documento.replace(".tif", f"_{ID}.tif")
                    old_path = os.path.join(fecha_path, documento)
                    new_path = os.path.join(fecha_path, new_name)
                    os.rename(old_path, new_path)