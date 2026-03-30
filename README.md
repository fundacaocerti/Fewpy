# Fewpy

An Open Source project for Few-Shot Learning (FSL). 

Fewpy implements state-of-art Image Segmentation FSL models. It allows the user to easily load and use the following models for inference:

## Use cases:

### Proof of Concept

Fewpy can be used to verify feasability of tasks in SOTA FSL models and support proof of concept.

### Dataset Annotation

Fewpy enables faster data annotation by simplifying the use of cutting-edge AI models.

### Benchmarking

Fewpy makes it easy to load and run highly performant models on benchmarks to experiment or compare with other model's results.

## FSLDataset

`FSLDataset` is a PyTorch-compatible dataset for Few-Shot Learning (FSL). It manages a **Query Set** (input data) and a **Support Set** (reference data). It features **Lazy Processing**, which triggers the transformation of the support set only once when the first item is accessed, based on a specific preprocessing strategy.

### Class: FSLDataset

#### Initialization Parameters
* **x**: (list[Image]) Query images.
* **s_x**: (list[Image]) Support images.
* **s_y**: (list/Tensor) Support labels, masks, or bounding box dictionaries.
* **labels**: (list[dict]) Optional annotations (bboxes/classes) for the query images.
* **img_size**: (tuple/int) Target (H, W) for resizing.
* **max_size**: (int) Optional maximum edge length constraint for resizing.
* **antialias**: (bool) If set True Resizing uses antialias.
* **interpolation**: (torchvision.transforms.InterpolationMode) Interpolation mode used in the resizing operation.
* **pixel_norm**: (tuple) Mean and Std for normalization.
* **center_crop**: (int) Size of the center crop.
* **support_set_preprocessing_method**: (PreprocessingMethod) Strategy for support set transformation. Options include:
    * `PreprocessingMethod.STANDARD`: Basic transform and stacking.
    * `PreprocessingMethod.RESIZE_SUPPORT_GT`: Resizes images and mask-style `s_y`.
    * `PreprocessingMethod.NORMALIZE_SUPPORT_GT`: Scales `s_y` bboxes to [0, 1].
    * `PreprocessingMethod.DETECTION_CROP`: Crops `s_x` around `s_y` bboxes with padding.
    * `PreprocessingMethod.NORM_DETECTION_CROP`: Padded crop with pixel normalization.
    * `PreprocessingMethod.AUGMENT_SUPPORT_IMAGES`: Creates more support images through augmentation of the support set.
    * `PreprocessingMethod.NONE`: Skips preprocessing.
* **transform_datapoints**: (bool) If True, applies the transform pipeline to Query images and scales their associated `labels`.

#### Key Logic
1. **Transform Pipeline**: A `torchvision.transforms.Compose` pipeline is built using `Resize`, `ToTensor`, and `Normalize` based on initialization arguments.
2. **Lazy Support Set Preprocessing**:
   - On initialization `__init__` call, the dataset executes a specific method determined by `support_set_preprocessing_method`.
   - **Stacking**: Images are stacked into a 4D tensor if shapes are uniform; otherwise, they remain a list of tensors.
   - **Detection Cropping**: Uses `_padded_crop` to zoom into support objects with a 0.5 padding ratio before resizing.
3. **Query Label Scaling**: If `transform_datapoints` is enabled and `labels` are provided, bounding boxes and segmentation maps for query images are automatically rescaled to match the new `img_size`.

#### Utility: fsl_collate
Standard collators stack every element, which can lead to redundant memory usage in FSL because the support set is usually identical for every query in a batch. `fsl_collate` optimizes this by:
- Stacking Query images (`x`) and their labels (`yi`) into a batch.
- Returning only the **first** instance of the processed support set (`s_x`, `s_y`) for the entire batch.

### Usage Example

```python
from torch.utils.data import DataLoader, PreprocessingMethod

# Initialize with detection-based cropping for the support set
ds = FSLDataset(
    x=q_imgs, 
    s_x=s_imgs, 
    s_y=s_annots, 
    img_size=(320, 320),
    support_set_preprocessing_method=PreprocessingMethod.DETECTION_CROP
)

# Use custom collate to prevent support set duplication in memory
loader = DataLoader(ds, batch_size=4, collate_fn=fsl_collate)

for queries, support_x, support_y in loader:
    # queries: [Batch, C, 320, 320]
    # support_x: [Support_Size, C, 320, 320] (Processed/Cropped)
    pass
```

## Models Documentation

Fewpy provides implementations for several few-shot learning architectures. You can run inference using the unified FewShotModel class and a configuration dictionary.

### Quick Start Example
To initialize a model, use the following structure:
```python
from fewpy.util.inference import FewShotModel

args = {
    "datasetname": 'voc_bottle_n_sofa',
    "classnames": ["bottle", "sofa"],
    "confidence_threshold": 0.5,
    "mapping_to_contiguous_ids": {"bottle": 0, "sofa": 1}
}

model = FewShotModel(model="AirShot", config=args)
```

### Supported Models

#### 1. AnomalyCLIP
* **Task:** Segmentation
* **Description:** Implementation of [AnomalyCLIP](https://arxiv.org/abs/2310.18961). Optimized for zero-shot or few-shot anomaly detection and segmentation.

* **Detailed Documentation:** [examples/models/anomalyclip.md](./examples/models/anomalyclip.md)

#### 2. FPTRANS
* **Task:** Segmentation
* **Description:** Adapted from [Feature-Proxy Transformer](https://arxiv.org/abs/2210.06908). Focuses on few-shot segmentation and supports SAHI for high-resolution processing.

* **Detailed Documentation:** [examples/models/fptrans.md](./examples/models/fptrans.md)

#### 3. AirShot (FSOD RCNN)
* **Task:** Detection
* **Description:** Based on [AirShot](https://arxiv.org/abs/2404.05069). An efficient Few-Shot Object Detection (FSOD) RCNN framework built with Detectron2 and FewX.

* **Detailed Documentation:** [examples/models/airshot.md](./examples/models/airshot.md)

#### 4. Qwen
* **Task:** Detection
* **Description:** A wrapper for [Qwen3-VL-8B-Instruct](https://arxiv.org/abs/2505.09388) that enables few-shot prompting for object detection via Vision-Language Models.

* **Detailed Documentation:** [examples/models/qwen.md](./examples/models/qwen.md)

#### 5. TipAdapter
* **Task:** Classification
* **Description:** Based on [Tip-Adapter](https://arxiv.org/abs/2207.09519). A training-free Adaption of CLIP for Few-shot Classification.

* **Detailed Documentation:** [examples/models/qwen.md](./examples/models/tipadapter.md)

## CVAT

Fewpy can be used with CVAT for data annotation. For that purpose inference outputs a dictionary and fewpy implements CVATAdapter. For examples of integration with cvat-cli and CVAT Community refer to [CVAT Examples](./examples/cvat/)

### Legal & Acknowledgments

**AirShot License (Copyright (c) 2024, Zihan Wang):**
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

