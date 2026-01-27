# Descarga, recorte, reproyección y guardado automático de productos GOES

import s3fs
import os
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.geometry import mapping

# ==========================================================
# 1. PARÁMETROS GENERALES
# ==========================================================

bucket_name = "noaa-goes16"   # Cambiar a noaa-goes18 / noaa-goes19 si se requiere

# Productos GOES y su variable principal
products = {
    "ABI-L1b-RadF": "Rad",
    "ABI-L2-FDCF": "Mask",
    "ABI-L2-ACHA2KMF": "HT",
    "ABI-L2-ACHAF": "HT",
    "ABI-L2-ACHP2KMF": "PRES",
    "ABI-L2-ACHTF": "TEMP",
    "ABI-L2-ACMF": "Cloud_Probabilities",
    "ABI-L2-ACTPF": "Phase",
    "ABI-L2-ADPF": "Smoke",
    "ABI-L2-CCLF": "TCF",
    "ABI-L2-CMIPF": "CMI",
    "ABI-L2-COD2KMF": "COD",
    "ABI-L2-CPSF": "CPS",
    "ABI-L2-CTPF": "PRES",
    "ABI-L2-DSIF": "CAPE",
    "ABI-L2-LST2KMF": "LST",
    "ABI-L2-LSTF": "LST",
}
years = [2024]
days_of_year = [8]
hours = [17]

# Directorio base
base_directory = r"C:\Users\User\Desktop\Datos\Output"

# Shapefile
shape_path = (
    r"C:\Users\User\Desktop\articulo"
    r"\Recorte de imagenes satelitales\Grilla reproyectada.shp"
)

# ==========================================================
# 2. CONEXIÓN A AWS
# ==========================================================
fs = s3fs.S3FileSystem(anon=True)

# ==========================================================
# 3. CRS GEOSTACIONARIO GOES
# ==========================================================
goes_crs = (
    "+proj=geos +h=35786023 +a=6378137 +b=6356752.31414 "
    "+lon_0=-75 +sweep=x +no_defs"
)

# ==========================================================
# 4. LEER SHAPEFILE
# ==========================================================
bogota_shape = gpd.read_file(shape_path)
bogota_shape = bogota_shape.to_crs(goes_crs)

# ==========================================================
# 5. BUCLE PRINCIPAL
# ==========================================================
for product_name, var in products.items():

    print("\n========================================")
    print(f"PROCESANDO PRODUCTO: {product_name}")
    print("========================================")

    product_dir = os.path.join(base_directory, product_name)
    netcdf_dir = os.path.join(product_dir, "netCDF")
    image_dir = os.path.join(product_dir, "imagenes")

    os.makedirs(netcdf_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    for year, day_of_year, hour in zip(years, days_of_year, hours):

        print(f"\nFecha: {year} | Día: {day_of_year} | Hora: {hour}")

        prefix = f"{bucket_name}/{product_name}/{year}/{day_of_year:03d}/{hour:02d}/"

        try:
            files = fs.ls(prefix)
        except FileNotFoundError:
            print("⚠️ No hay datos disponibles.")
            continue

        if not files:
            print("⚠️ Carpeta vacía.")
            continue

        # --------------------------------------------------
        # --------------------------------------------------
        first_file = files[0]
        local_nc = os.path.join(netcdf_dir, os.path.basename(first_file))
        
        if os.path.exists(local_nc) and os.path.getsize(local_nc) < 10_000:
            print("⚠️ NetCDF corrupto detectado, se elimina.")
            os.remove(local_nc)
        
        if not os.path.exists(local_nc):
            fs.download(first_file, local_nc)
        else:
            print("ℹ️ NetCDF válido ya existe.")
        
        try:
            ds = xr.open_dataset(local_nc, engine="netcdf4")
        except Exception as e:
            print(f"❌ Error al abrir NetCDF: {e}")
            continue
        
                # --------------------------------------------------
        # ABRIR DATASET
        # --------------------------------------------------
        ds = xr.open_dataset(local_nc)
        print("Variable utilizada:", var)


        # Escalar coordenadas
        h = 35786023
        ds = ds.assign_coords({"x": ds.x * h, "y": ds.y * h})

        da = ds[var].rio.write_crs(goes_crs)

        # --------------------------------------------------
        # CLIP
        # --------------------------------------------------
        clipped = da.rio.clip(
            bogota_shape.geometry.apply(mapping),
            bogota_shape.crs,
            drop=True
        )

        nx = clipped.sizes.get("x", 0)
        ny = clipped.sizes.get("y", 0)

        if nx < 2 or ny < 2:
            print(f"⚠️ {product_name} sin cobertura efectiva.")
            continue

        # --------------------------------------------------
        # REPROYECCIÓN
        # --------------------------------------------------
        clipped_lonlat = clipped.rio.reproject("EPSG:4326")

        # --------------------------------------------------
        # GUARDAR IMAGEN
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        clipped_lonlat.plot(ax=ax, cmap="inferno")

        ax.set_title(
            f"GOES {product_name}\n"
            f"{year} - Day {day_of_year:03d} - {hour:02d}:00 UTC"
        )

        image_name = (
            f"{product_name}_{year}_"
            f"DOY{day_of_year:03d}_H{hour:02d}.png"
        )

        image_path = os.path.join( r"C:\Users\User\Desktop\Articulo\Imagenes satelitales", image_name)

        plt.savefig(image_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"✅ Imagen guardada: {image_name}")
