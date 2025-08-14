import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

from src.data.image_matching import score_cutlines


# Write test for score cutlines
def test_score_cutlines():
    r_inp_img = np.zeros((100, 100), dtype=np.float32)
    r_inp_img[20:60, 20:60] = 100
    r_inp_img[:, :20] = 33
    r_inp_img[:20, :] = 23

    r_inp_img[90:, 90:] = 5

    polygons = [Polygon([(20, 20), (60, 20), (60, 60), (20, 60), (20, 20)]), # 100
                Polygon([(-10, -10), (20, -10), (20, 100), (-10, 100), (-10, -10)]),  # 23
                Polygon([(20, -10), (20, 20), (100, 20), (100, -10), (20, -10)]), # 33
                Polygon([(90, 90), (100, 90), (100, 100), (90, 100), (90,90)]), # 5
                Polygon([(20, 60), (20, 100), (90, 100), (90, 90), (100, 90), (100, 20), (60, 20), (60, 60), (20, 60)])] # 0

    cutlines = gpd.GeoDataFrame({'geometry': polygons})
    cutlines = score_cutlines(r_inp_img, cutlines)
    # print(cutlines)
    assert cutlines.loc[0, 'score'] == max(cutlines['score'].tolist())
    assert cutlines.loc[1, 'score'] < cutlines.loc[4, 'score'] # 33
    assert cutlines.loc[2, 'score'] < cutlines.loc[1, 'score']  # 23
    assert cutlines.loc[3, 'score'] == min(cutlines['score'].tolist())
    assert cutlines.loc[4, 'score'] < cutlines.loc[0, 'score']

def test_score_cutlines2():
    r_inp_img = np.zeros((100, 100), dtype=np.float32)
    r_inp_img[:50, :50] = 0
    r_inp_img[50:, :50] = 1
    r_inp_img[:50, 50:] = 4
    r_inp_img[50:, 50:] = 9

    polygons = [Polygon([(0, 0), (50, 0), (50, 50), (0, 50), (0, 0)]), # 0
                Polygon([(50, 0), (100, 0), (100, 50), (50, 50), (50, 0)]), # 1
                Polygon([(0, 50), (50, 50), (50, 100), (0, 100), (0, 50)]), # 4
                Polygon([(50, 50), (100, 50), (100, 100), (50, 100), (50, 50)])] # 9




    cutlines = gpd.GeoDataFrame({'geometry': polygons})
    cutlines = score_cutlines(r_inp_img, cutlines)
    assert cutlines.loc[0, 'score'] == min(cutlines['score'].tolist())
    assert cutlines.loc[1, 'score'] == cutlines.loc[2, 'score']  # 1,4
    assert cutlines.loc[3, 'score'] == max(cutlines['score'].tolist())

if __name__ == '__main__':
    test_score_cutlines()
    test_score_cutlines2()