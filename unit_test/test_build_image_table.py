import typing
from pathlib import Path

import src.utils.data_utils
from src.data.image_table import build_images_table
from src.modelling import helper


def test_build_idf():
    data_dir = Path("./unit_test/sample_labels")
    n_images = 3
    file_names = [data_dir / f"idf_img_sample_{i}.tif" for i in range(n_images)]
    filelist_path = data_dir / "file_list"
    filelist_path_2 = data_dir / "file_list_2"
    with open(filelist_path, "w") as file:
        for file_name in file_names:
            file.write(str(file_name) + "\n")
    with open(filelist_path_2, "w") as file:
        for file_name in file_names[:-1]:  # write one less
            file.write(str(file_name) + "\n")

    # create dummy images
    [src.utils.data_utils.write_geotiff_to_file(3, 50, 50, (0, 0, 50, 50), file_name) for file_name in file_names]

    image_srcs: typing.Dict = {
        "random_5m": {
            "base_path": data_dir,
            "image_file_type": ".tif",
            "image_file_prefix": "idf_",
            "image_file_postfix": "",
        }
    }

    idf = build_images_table(image_srcs, "random_5m", None)
    assert len(idf) == n_images

    # with set project_crs
    crs = "EPSG:6933"
    idf_crs = build_images_table(image_srcs, "random_5m", "EPSG:6933")
    assert len(idf_crs) == n_images
    assert idf_crs.crs == crs

    # create with multiple cpus and check if the idf has the same amount of images (note: might in different order)
    idf_mthreat = build_images_table(image_srcs, "random_5m", None, n_jobs=2)
    assert len(idf_mthreat.merge(idf, on="path")) == n_images

    # test file list path function
    image_srcs["random_5m"].update({"filelist_path": filelist_path})
    idf = build_images_table(image_srcs, "random_5m", None)
    assert len(idf) == n_images

    image_srcs["random_5m"].update({"filelist_path": filelist_path_2})
    idf = build_images_table(image_srcs, "random_5m", None)
    assert len(idf) == n_images - 1


if __name__ == "__main__":
    test_build_idf()
