# Running AirShot with fewpy

In order to run the AirShot implementation within fewpy you will need:
    - to import fewpy's FewShotModel class
    - a checkpoint (weight) file for the model named "airshot.pth"
    - to import fewpy's FSLDataset (optional)

## Obtaining the models weights

The weights were made available through the paper [Official Github Repository](https://github.com/Z1hanW/AirShot). There is download link for the file in the repository's README.md file under **USAGE**.

The easiest way to make the weights available to fewpy is to download them, rename them and move them to a "weights" folder in your projects' root directory.

## Importing necessary fewpy tooling

In order to instatiate the model FewShotModel is necessary, so import it as in the following snippet.

```python
from fewpy.util.inference.FewShotModel import FewShotModel
```

You may or may not use fewpy's FSLDataset class. Notice that it is a recommendation to use torch's DataLoader with FSLDataset but any compatible implementation should work. Fewpy also contains a collate function that should be used with torch.util.data.DataLoader.

```python
from fewpy.util.data.dataset import FSLDataset, fsl_collate
from torch.utils.data import DataLoader
```

## Instatiating AirShot

Now that you imported FewShotModel all you will need to instatiate the model is a configuration dictionary, name it whatever works best for as it will be passed to the FewShotModel contructor.

The configuration parameters the model expects are:

```python
args = {
    "datasetname": 'voc_bottle_n_sofa', # the name of your dataset (optional)
    "classnames": ["bottle", "sofa"],   # a list containing the name of each class in your dataset
    "confidence_threshold": 0.5,    # lower bound of confidence for accepted proposals (optional)
    "mapping_to_contiguous_ids": {"bottle": 0, "sofa": 1}   # id mapper in case dataset does not have contiguous ids
}
```

## Preparing your input

AirShot instance acessible through FewShotModel expects an input of a certain format. The query (x) may be batched into a list of tensors list[torch.Tensor]. The support images (s_x) is the same for all batched inputs. It may be a list of tensors or optionally a list of paths to the image files list[str]. Since the list of support images is the same for all batched queries the same applies to the ground truth list. It is a list of dictionaries, each containing annotations for a single support images. 

```python
annotation = {
    "cls": cls_name,    # name of the objects class
    "bboxes": boxes,    # torchTensor containing the bounding boxes for all N_i detections of that class contained in the image
    # format (N_i, 4)
}
```

As previouly mentioned you may or may not use fewpy's dataset class and collate function. A full example of AirShot inference using fewpy is availble in [Fewpy/examples/models/airshot.py]().
