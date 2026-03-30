import pytest
import torch
import numpy as np
from PIL import Image

from fewpy.util.data import d2_tensor_transform, PreprocessingMethod, FSLDataset, fsl_collate

@pytest.fixture
def dummy_images():
    return [Image.new("RGB", (100, 100), color=(i * 10, 255, 255)) for i in range(3)]

@pytest.fixture
def dummy_bbox_labels():
    return [{"bboxes": [[10.0, 10.0, 50.0, 50.0]], "cls": [0]} for _ in range(3)]

@pytest.fixture
def dummy_mask_labels():
    return [torch.zeros((1, 100, 100)) for _ in range(3)]

@pytest.fixture
def sample_data():
    imgs = [Image.new("RGB", (100, 200)) for _ in range(2)] # W=100, H=200
    labels = [
        {"bboxes": [[10, 20, 50, 80]], "cls": [1]}, # xmin, ymin, xmax, ymax
        {"bboxes": [[0, 0, 10, 10]], "cls": [2]}
    ]
    return imgs, labels

def test_d2_tensor_transform():
    np_img = np.zeros((100, 100, 3), dtype=np.uint8)
    tensor = d2_tensor_transform(np_img)
    
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 100, 100)
    assert tensor.dtype == torch.float32

def test_dataset_initialization_no_support(dummy_images):
    dataset = FSLDataset(x=dummy_images, img_size=(224, 224))
    
    assert len(dataset) == 3
    assert not dataset.support_set
    
    item = dataset[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (3, 224, 224)

def test_dataset_standard_preprocessing(dummy_images, dummy_bbox_labels):
    dataset = FSLDataset(
        x=dummy_images,
        s_x=dummy_images,
        s_y=dummy_bbox_labels,
        labels=dummy_bbox_labels,
        img_size=(224, 224),
        support_set_preprocessing_method=PreprocessingMethod.STANDARD
    )
    
    assert dataset.support_set
    
    xi, yi, s_x, s_y = dataset[0]
    
    assert isinstance(xi, torch.Tensor)
    assert xi.shape == (3, 224, 224)
    
    assert isinstance(dataset.s_x, torch.Tensor)
    assert dataset.s_x.shape == (3, 3, 224, 224)
    
    assert "bboxes" in yi
    assert isinstance(yi["bboxes"], list)

def test_dataset_resize_support_gt(dummy_images, dummy_mask_labels):
    dataset = FSLDataset(
        x=dummy_images,
        s_x=dummy_images,
        s_y=dummy_mask_labels,
        img_size=(50, 50),
        support_set_preprocessing_method=PreprocessingMethod.RESIZE_SUPPORT_GT
    )
    
    assert isinstance(dataset.s_y, torch.Tensor)
    assert dataset.s_y.shape == (3, 50, 50)

def test_dataset_detection_crop(dummy_images, dummy_bbox_labels):
    dataset = FSLDataset(
        x=dummy_images,
        s_x=dummy_images,
        s_y=dummy_bbox_labels,
        img_size=(64, 64), # Cropped size
        support_set_preprocessing_method=PreprocessingMethod.DETECTION_CROP
    )
    
    assert isinstance(dataset.s_x, torch.Tensor)
    assert dataset.s_x.shape == (3, 3, 64, 64)
    
    assert isinstance(dataset.s_y[0]["bboxes"], torch.Tensor)
    assert "cls" in dataset.s_y[0]

def test_dataset_augment_support_images(dummy_images, dummy_bbox_labels):
    dummy_tensor_labels = [torch.tensor([1]) for _ in range(3)]
    
    epochs = 2
    dataset = FSLDataset(
        x=dummy_images,
        s_x=dummy_images,
        s_y=dummy_tensor_labels,
        support_set_preprocessing_method=PreprocessingMethod.AUGMENT_SUPPORT_IMAGES,
        epochs=epochs
    )
    
    assert isinstance(dataset.s_x, torch.Tensor)
    assert dataset.s_x.shape == (6, 3, 224, 224) 

def test_fsl_collate_with_query_labels(dummy_images, dummy_bbox_labels):
    dataset = FSLDataset(
        x=dummy_images,
        s_x=dummy_images,
        s_y=dummy_bbox_labels,
        labels=dummy_bbox_labels,
        img_size=(224, 224)
    )
    
    batch = [dataset[0], dataset[1]]
    
    batch_xi, batch_yi, s_x, s_y = fsl_collate(batch)
    
    assert isinstance(batch_xi, torch.Tensor)
    assert batch_xi.shape == (2, 3, 224, 224)
    assert isinstance(batch_yi, list)
    assert len(batch_yi) == 2
    
    assert s_x.shape == (3, 3, 224, 224)

def test_fsl_collate_without_query_labels(dummy_images, dummy_bbox_labels):
    dataset = FSLDataset(
        x=dummy_images,
        s_x=dummy_images,
        s_y=dummy_bbox_labels,
        img_size=(224, 224)
    )
    
    batch = [dataset[0], dataset[1]]
    
    batch_xi, s_x, s_y = fsl_collate(batch)
    
    assert isinstance(batch_xi, torch.Tensor)
    assert batch_xi.shape == (2, 3, 224, 224)
    assert s_x.shape == (3, 3, 224, 224)

def test_normalize_support_gt(sample_data):
    imgs, labels = sample_data
    dataset = FSLDataset(
        x=imgs,
        s_x=imgs,
        s_y=labels,
        support_set_preprocessing_method=PreprocessingMethod.NORMALIZE_SUPPORT_GT
    )
    
    first_bbox = dataset.s_y[0]["bboxes"][0]
    assert first_bbox == [0.1, 0.1, 0.5, 0.4]

def test_norm_detection_crop(sample_data):
    imgs, labels = sample_data
    pixel_norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    dataset = FSLDataset(
        x=imgs,
        s_x=imgs,
        s_y=labels,
        img_size=(128, 128),
        pixel_norm=pixel_norm,
        support_set_preprocessing_method=PreprocessingMethod.NORM_DETECTION_CROP
    )
    
    assert isinstance(dataset.s_x, torch.Tensor)
    assert dataset.s_x.shape == (2, 3, 128, 128)
    assert dataset.s_x.min() < 0 or dataset.s_x.max() > 1

def test_preprocessing_none(sample_data):
    imgs, labels = sample_data
    dataset = FSLDataset(
        x=imgs,
        s_x=imgs,
        s_y=labels,
        support_set_preprocessing_method=PreprocessingMethod.NONE
    )

    assert isinstance(dataset.s_x[0], Image.Image)
    assert len(dataset.s_x) == 2

def test_unsupported_method_error(sample_data):
    imgs, labels = sample_data
    with pytest.raises(ValueError, match="Unsuported support_set_preprocessing_method"):
        FSLDataset(
            x=imgs,
            s_x=imgs,
            s_y=labels,
            support_set_preprocessing_method="invalid_method_name"
        )

def test_padded_crop_logic(sample_data):
    imgs, labels = sample_data
    dataset = FSLDataset(x=imgs)

    patch, bbox = dataset._padded_crop(imgs[0], labels[0]["bboxes"][0], output_size=(64, 64))
    
    assert patch.shape == (3, 64, 64)
    assert len(bbox) == 4
