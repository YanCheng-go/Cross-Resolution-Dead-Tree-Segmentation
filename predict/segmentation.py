import logging
import os
import warnings
from functools import partial
from pathlib import Path

import torch
from torch import nn
from tqdm import tqdm

from config.base import BaseSegmentationConfig
from src.data.base_dataset import get_dataloader
from src.data.base_segmentation_dataset import BaseSegmentationDataset
from src.data.collate import default_collate_with_shapely_support
from src.data.image_table import build_images_table
from src.data.raster_prediction_writer import RasterPredictionWriter
from src.modelling import helper
from src.utils import config_parser
from src.visualization.visualize import image_for_display
from train.base import init_config
from train.segmentation import SegmentationTrainer

warnings.filterwarnings("ignore")

normalize = partial(image_for_display, per_instance=True)


class SegmentationPredictor(nn.Module):
    raster_writer_cls = RasterPredictionWriter
    dataset_cls: BaseSegmentationDataset = BaseSegmentationDataset

    def predict(self, out_path: [str, Path]):

        out_path = Path(out_path)
        out_path.mkdir(exist_ok=True)
        if os.listdir(out_path):
            logging.warning(f"Directory is not empty! {out_path}: {os.listdir(out_path)}")

        # go through images
        areas = self.idf.query(f"src == @self.config.reference_source")[['geometry']].copy()
        # predict each image individually
        for i in tqdm(areas.index.values, f"Predicting tile ..."):
            image_df = self.idf.query(f"src != @self.config.reference_source or index == @i").copy()
            area = areas.loc[[i]].copy()
            project_crs = self.idf.loc[i]["ori_crs"]
            if image_df.crs != project_crs:
                image_df = image_df.to_crs(project_crs)
                area = area.to_crs(project_crs)
            test_dataset = self.dataset_cls(
                labeled_areas=area, images_df=image_df,
                reference_source=self.config.reference_source,
                target_features=None,
                target_col=None,
                target_classes=None,
                patch_size=self.config.patch_size,
                sequential_stride=self.config.sequential_stride,
                allow_empty_patches=True,
                allow_partial_patches=self.config.allow_partial_patches,
                processed_dir=self.config.processed_dir, save_samples=self.config.save_samples,
                save_patch_df=self.config.save_patch_df,
                save_labeled_areas_df=self.config.save_labeled_areas_df,
                project_crs=project_crs
            )

            test_image_loader = get_dataloader(
                test_dataset, self.config.batch_size_val, self.config.num_workers,
                collate_fn=default_collate_with_shapely_support, train=False
            )

            """Evaluation of the network"""
            self.predict_image(out_path, test_image_loader)

    def predict_image(self, out_path, test_image_loader):
        output_class_labels = self.config.get("output_class_labels", True)
        pred_writer = self.raster_writer_cls(
            test_image_loader.dataset, out_path, 'det_', output_class_labels, self.config.get("compression", None),
            self.config.get("n_edge_removal", 0)
        )
        n_batches = len(test_image_loader)
        self.model.eval()
        self.aug_transform.eval()
        with tqdm(total=n_batches, desc=f'Evaluating ...', leave=False) as pbar:
            with torch.no_grad():
                for batch in test_image_loader:
                    x = self.aug_transform.get_input(batch, self.device)  # Normalize
                    pred = self.model(x)
                    if output_class_labels:
                        if self.config.n_classes > 1:
                            # Since the written image must have at least one channel we keep all dims
                            pred = torch.argmax(pred, dim=1, keepdim=True)
                        else:
                            pred = (pred > 0).int()
                    else:
                        if self.config.n_classes > 1:
                            pred = torch.softmax(pred, dim=1)
                        else:
                            pred = pred.sigmoid()
                    prb = {
                        'patch_id': batch['patch_id'].cpu().numpy(),
                        'predictions': pred.cpu().numpy(),
                    }
                    pred_writer(prb)
                    pbar.update()
        pred_writer.dump_cache()

    # The code is structured along some basics building blocks.
    # The idea is that a user reuses them or adapts them as per the task.
    def __init__(self, config, trainer_cls=SegmentationTrainer):
        super(SegmentationPredictor, self).__init__()
        self.trainer_cls = trainer_cls

        self.config = config
        parser = config_parser.config_to_argparser(config)
        args = parser.parse_args()
        config_parser.update_conf_with_parsed_args(config, args)

        params = "prediction"#model_params_as_str(config)
        self.run_dir = helper.initialize_rundir(config.log_dir, params)
        helper.save_config_to_rundir(self.run_dir, config)

        assert config.load is not None

        self.device = helper.get_device()
        logging.info(f'Using device {self.device}')

        self.idf = build_images_table(
            config.image_srcs, config.reference_source, project_crs=config.project_crs,
            processed_dir=config.processed_dir, save_idf=config.save_idf
        )

        # Normalization is a part of the network
        config.use_augmentation = False
        self.aug_transform = trainer_cls.init_aug_transform(config)

        self.model, _ = trainer_cls.init_model(config, self.device)
        logging.info('Network: %s', self.model)

        self.to(self.device)


if __name__ == '__main__':
    config = init_config(BaseSegmentationConfig())
    predictor = SegmentationPredictor(config, SegmentationTrainer)
    predictor.predict("output")
