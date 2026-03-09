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

`FSLDataset` is a PyTorch-compatible dataset for Few-Shot Learning. It handles a **Query Set** (input data) and a **Support Set** (reference data). It uses **Lazy Processing** to transform the support set only once when the first item is accessed.

### Class: FSLDataset

#### Initialization Parameters
* **x**: (list[Image]) Query images.
* **s_x**: (list[Image]) Support images.
* **s_y**: (list/Tensor) Support labels or bounding box dicts.
* **img_size**: (tuple) Target (H, W).
* **max_size**: (int) Max edge length for resizing.
* **pixel_norm**: (tuple) Mean and Std for normalization.
* **norm_annot**: (bool) If True, scales bboxes to [0, 1].
* **resize_annot**: (bool) If True, resizes mask tensors to img_size.
* **transform_datapoints**: (bool) If True, applies transforms to Query images.

#### Key Logic
1. **Transform Pipeline**: Combines Resize, ToTensor, and Normalize based on input parameters.
2. **Lazy Support Set Preprocessing**:
   - On the first `__getitem__` call, `s_x` and `s_y` are processed.
   - Images are transformed and stacked into a 4D tensor if shapes match.
   - Bounding boxes are normalized if `norm_annot` is True (requires "bboxes" key).
   - Masks are resized using Nearest Neighbor if `resize_annot` is True.
3. **Caching**: Once processed, `self.support_set_preproc` is set to True to bypass reprocessing for subsequent indices.

#### Utility: fsl_collate
Standard collators stack every element. In Few-Shot Learning, the support set is usually identical for every query in a batch. `fsl_collate` prevents memory bloat by:
- Stacking Query images (`x`) into a batch tensor.
- Returning only the **first** instance of the support set (`s_x`, `s_y`) for the entire batch.

### Usage Example

```python
from torch.utils.data import DataLoader

# Initialize
ds = FSLDataset(x=q_imgs, s_x=s_imgs, s_y=s_annots, img_size=(224, 224))

# Use custom collate
loader = DataLoader(ds, batch_size=4, collate_fn=fsl_collate)

# Iterate
for queries, support_x, support_y in loader:
    # queries: [Batch, C, H, W]
    # support_x: [Support_Size, C, H, W]
    pass
```

## Models Documentation

Fewpy provides implementations for several few-shot learning architectures. You can run inference using the unified FewShotModel class and a configuration dictionary.

### Quick Start Example
To initialize a model, use the following structure:

from fewpy.util.inference.FewShotModel import FewShotModel

args = {
    "datasetname": 'voc_bottle_n_sofa',
    "classnames": ["bottle", "sofa"],
    "confidence_threshold": 0.5,
    "mapping_to_contiguous_ids": {"bottle": 0, "sofa": 1}
}

model = FewShotModel(model="AirShot", config=args)

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

## CVAT

Fewpy can be used with CVAT for data annotation. For that purpose inference outputs a dictionary and fewpy implements CVATAdapter. For examples of integration with cvat-cli and CVAT Community refer to [CVAT Examples](./examples/cvat/)

### Legal & Acknowledgments

**AirShot License (Copyright (c) 2024, Zihan Wang):**
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

