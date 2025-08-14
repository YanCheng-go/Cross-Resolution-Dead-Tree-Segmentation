import unittest
import torch

from src.data.watershed_dataset import calculate_size_weights
import rasterio


def read_tif_file(patch_fp='/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/aug20240607_data20240522_256_5c_countWeights_edgeWeights/extracted_pts/train_6279.tif'):
    with rasterio.open(patch_fp) as src:
        arr = src.read()
    return arr


class TestCalculateEdgeWeights(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        patch_fp = '/mnt/raid5/DL_TreeHealth_Aerial/Merged/processed_dir/aug20240612_data20240522_256_5c_countWeights_edgeWeights/extracted_pts/train_6279.tif'
        arr = rasterio.open(patch_fp).read()
        self.mask = torch.tensor(arr[7], dtype=torch.int)
        self.energy_layer = torch.tensor(arr[4], dtype=torch.int)
        self.expected_output = 69

    def test_size_weight(self):
        result = calculate_size_weights(self.mask, device=self.device)

        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.mask.cpu().numpy())
        axs[0].set_title('Mask')
        axs[1].imshow(result.cpu().numpy())
        axs[1].set_title('Size weights')
        # add colorbars to each subplot
        fig.colorbar(axs[0].imshow(self.mask.cpu().numpy()), ax=axs[0])
        fig.colorbar(axs[1].imshow(result.cpu().numpy()), ax=axs[1])
        plt.show()


if __name__ == '__main__':
    unittest.main()