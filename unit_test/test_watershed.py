import os
import typing
from pathlib import Path
from shutil import rmtree

import geopandas as gpd

import src.utils.data_utils
from config import base
from config.watershed import WatershedConfig
from predict.watershed import WatershedPredictor
from train.watershed import WatershedTrainer


class WatershedSegmentationTestConfig(WatershedConfig):
    """
    Over-write the BaseSegmentationConfig
    """
    pipeline_test = True
    data_dir: str = Path("./unit_test/sample_labels")
    target_features_file: str = "training_polygons_example.gpkg"
    target_col: str = None
    target_classes: str = None
    dataset_split_by: str = "patches"  # areas or patches
    split_col: str = "split"  # When splitting by areas. The split is written into this col

    labeled_areas_file: str = "training_areas_example.gpkg"
    allow_partial_patches: bool = True
    reference_source: str = "random_5m"
    patch_size = (64, 64)

    image_srcs: typing.Dict = {
        "random_5m": {
            "base_path": data_dir,
            "image_file_type": ".tif",
            "image_file_prefix": "instance_segmentation_img_sample",
            "image_file_postfix": "",
        }
    }

    log_dir: str = "./unit_test/runs"
    processed_dir: str = None
    epochs: int = 2
    backbone = "efficientnet_v2_s"

    def __init__(self) -> None:
        super().__init__()
        areas = gpd.read_file(os.path.join(self.data_dir, self.labeled_areas_file))
        project_crs = self.project_crs if self.project_crs is not None else areas.crs
        areas.to_crs(project_crs, inplace=True)
        src.utils.data_utils.write_geotiff_to_file(
            3, 5000, 5000, areas.total_bounds, self.data_dir / "instance_segmentation_img_sample.tif"
            # , crs=project_crs
        )
        self.__doc__ = base.extend_docstring(self, base.BaseSegmentationConfig, base.BaseConfig)


def train_segmentation():
    config = WatershedSegmentationTestConfig()
    trainer = WatershedTrainer(config)
    trainer.train_net()

    assert True
    return config, trainer.run_dir


def predict_segmentation(config, run_dir, save_dir):
    config.load = f"{run_dir}/model/BestModel.pth"
    predictor = WatershedPredictor(config, WatershedTrainer)
    predictor.predict(save_dir)

    assert True

    return predictor.run_dir


def test_watershed_segmentation():
    train_segmentation()
    config, run_dir = train_segmentation()
    save_dir = "unit_test/test_wsegmentation"
    prun_dir = predict_segmentation(config, run_dir, save_dir)

    rmtree(save_dir)
    rmtree(run_dir)
    rmtree(prun_dir)


if __name__ == "__main__":
    test_watershed_segmentation()
