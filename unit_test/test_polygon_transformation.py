import time

from src.data.base_dataset import geometry_to_crs_pyproj, geometry_to_crs
import geopandas as gpd
from shapely.geometry import MultiPolygon


def test_polygon_transformation():
    # Edge case; rectangle < patch
    polys = gpd.read_file("./unit_test/sample_labels/training_polygons_example.gpkg")
    from_crs = polys.crs
    to_crs = 'EPSG:6933'
    l_polys = polys['geometry'].tolist()

    print(len(polys))

    t0 = time.time()

    m1_polys = [geometry_to_crs(l, from_crs, to_crs) for l in l_polys]
    t1 = time.time()
    m2_polys = [geometry_to_crs_pyproj(l, from_crs, to_crs) for l in l_polys]
    t2 = time.time()
    m3_polys = geometry_to_crs(MultiPolygon(l_polys), from_crs, to_crs)
    t3 = time.time()

    print(f"Time for geopandas method: {t1 - t0}")
    print(f"Time for pyproj method: {t2 - t1}")
    print(f"Time for geopandas combined method: {t3 - t2}")

    assert m1_polys == m2_polys
    assert list(m3_polys.geoms) == m2_polys
