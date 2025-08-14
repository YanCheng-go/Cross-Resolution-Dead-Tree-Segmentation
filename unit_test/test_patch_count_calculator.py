from src.data.base_dataset import BaseDataset

patches_count_calculator = BaseDataset.patches_count_calculator


def test_patches_count_calculator():
    # Edge case; rectangle < patch
    rectangle_shape = 10
    patch_size = 20
    stride_size = 5
    assert patches_count_calculator(rectangle_shape, patch_size, stride_size, False) == 0
    assert patches_count_calculator(rectangle_shape, patch_size, stride_size, True) == 1

    # Normal case; rectangle >> patch
    rectangle_shape = 30
    patch_size = 10
    stride_size = 5
    assert patches_count_calculator(rectangle_shape, patch_size, stride_size, False) == 5
    assert patches_count_calculator(rectangle_shape, patch_size, stride_size, True) == 5

    rectangle_shape = 51
    patch_size = 10
    stride_size = 5
    assert patches_count_calculator(rectangle_shape, patch_size, stride_size, False) == 9
    assert patches_count_calculator(rectangle_shape, patch_size, stride_size, True) == 10

    rectangle_shape = 21
    patch_size = 10
    stride_size = 2
    assert patches_count_calculator(rectangle_shape, patch_size, stride_size, False) == 6
    assert patches_count_calculator(rectangle_shape, patch_size, stride_size, True) == 7

    # Build up testing;
    cmx = patch_size
    c = 1
    while cmx < rectangle_shape:
        cmx += stride_size
        c += 1
        assert c == patches_count_calculator(cmx, patch_size, stride_size, False)
        assert c + 1 == patches_count_calculator(cmx + 2, patch_size, stride_size, True)

    rectangle_shape = 21
    patch_size = 10
    stride_size = 2

    cmx = patch_size
    c = 1
    while cmx < rectangle_shape:
        cmx += stride_size
        c += 1
        assert c == patches_count_calculator(cmx, patch_size, stride_size, False)
        assert c + 1 == patches_count_calculator(cmx + 1, patch_size, stride_size, True)
