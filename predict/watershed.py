import numpy as np
import rasterio
import torch
from skimage.segmentation import watershed
from tqdm import tqdm

from predict.segmentation import SegmentationPredictor


class WatershedPredictor(SegmentationPredictor):
    def predict_image(self, out_path, test_image_loader):
        output_class_labels = self.config.get("output_class_labels", True)
        if output_class_labels:
            print("Warning: this script only outputs probabilities and watershed output, no direct class labels")

        pred_writer = self.raster_writer_cls(
            test_image_loader.dataset, out_path, 'det_', False, self.config.get("compression", None),
            self.config.n_edge_removal
        )
        n_batches = len(test_image_loader)
        self.model.eval()
        self.aug_transform.eval()
        with tqdm(total=n_batches, desc=f'Evaluating ...', leave=False) as pbar:
            with torch.no_grad():
                for batch in test_image_loader:
                    x = self.aug_transform.get_input(batch, self.device)  # Normalize
                    pred_seg, pred_sobel, pred_energy, pred_density = self.model(x)

                    if output_class_labels:
                        if self.config.n_classes > 1:
                            pred_seg = torch.argmax(pred_seg, dim=1, keepdim=True)
                        else:
                            pred_seg = (pred_seg > 0).int()

                        pred = torch.cat([pred_seg, torch.argmax(pred_energy, dim=1, keepdim=True)], 1)

                    else:
                        if self.config.n_classes > 1:
                            pred_seg = torch.softmax(pred_seg, dim=1)
                        else:
                            pred_seg = pred_seg.sigmoid()

                        pred = torch.cat([pred_seg, torch.softmax(pred_energy, dim=1), pred_density], 1)
                    prb = {
                        'patch_id': batch['patch_id'].cpu().numpy(),
                        'predictions': pred.cpu().numpy(),
                    }
                    pred_writer(prb)
                    pbar.update()
        if output_class_labels:
            pred_writer.cache_image[np.isnan(pred_writer.cache_image)] = -1
            energy = pred_writer.cache_image[1]
            mask = pred_writer.cache_image[1] > 0
            pred_watershed = watershed(-energy, connectivity=2, mask=mask)

            pred_writer.cache_image = np.append(pred_writer.cache_image, pred_watershed[None], axis=0)
            pred_writer.cache_image = pred_writer.cache_image.astype(int)
            pred_writer.dtype = rasterio.int32

        pred_writer.dump_cache()
