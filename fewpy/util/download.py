import hashlib
import warnings
import sys

from pathlib import Path
from urllib.request import urlopen

import torch

from typing import List, Union

from tqdm import tqdm


model2url = {
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

def download(
    model: str,
    cache_dir: Union[str, None] = None,
    new_name: str="",
): 
    
    if model not in model2url.keys():
        valid_models = ";\n".join(model2url.keys()) + "."
        raise ValueError(f"Tried to download invalid model {url}\nValid Models:\n{valid_models}")
    
    url = model2url[model]

    if not cache_dir:
        current_dir = Path(sys.path[0])
        cache_dir = current_dir / "weights"
    else:
        cache_dir = Path(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name

    if "openaipublic" in url:
        expected_sha256 = url.split("/")[-2]
    elif "mlfoundations" in url:
        expected_sha256 = Path(filename).stem.split("-")[-1]
    else:
        expected_sha256 = ""

    download_target = cache_dir / filename

    if download_target.exists() and not download_target.is_file():
        raise RuntimeError(
            f"{download_target} exists and is not a regular file"
        )

    if download_target.is_file():
        if expected_sha256:
            if (
                hashlib.sha256(download_target.read_bytes())
                .hexdigest()
                .startswith(expected_sha256)
            ):
                return str(download_target)
            else:
                warnings.warn(
                    f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
                )
        else:
            return str(download_target)

    with (
        urlopen(url) as source,
        download_target.open("wb") as output,
    ):
        with tqdm(
            total=int(source.headers.get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,    
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(
        download_target.read_bytes()
    ).hexdigest().startswith(expected_sha256):
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not not match"
        )
    
    if new_name != "":
        download_target = download_target.rename(f"{download_target.parent / new_name}{download_target.suffix}")

    return str(download_target)

import torch
import torchvision.models as models

class BackboneFactory:
    STANDARD_MODELS = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "mobilenet_v3": models.mobilenet_v3_small,
        "vit_b_16": models.vit_b_16,
    }

    @staticmethod
    def extract_visual_encoder(model, device, keep_avg_pool=False):
        
        extracted = False
        if hasattr(model, "fc"): 
            model.fc = torch.nn.Identity()
            extracted = True
        if hasattr(model, "head"): 
            model.head = torch.nn.Identity()
            extracted = True
        if hasattr(model, "avgpool") and not keep_avg_pool:
            model.avgpool = torch.nn.Identity()
            extracted = True

        if extracted: return model

        visual = None
        if hasattr(model, "visual"): 
            visual = model.visual
        elif hasattr(model, "visual_tower"): 
            visual = model.visual_tower
        else:
            return model

        if not isinstance(model, torch.jit.ScriptModule):
            visual = torch.jit.trace(visual, torch.randn(1, 3, 224, 224).to(device))
        backbone = torch.jit.freeze(visual.eval())

        del model
        torch.cuda.empty_cache()

        return backbone

    @classmethod
    def get_backbone(cls, model_identifier: str, keep_avg_pool: bool = False, cache_dir: str = "./cache"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_identifier.lower() in cls.STANDARD_MODELS:
            print(f"Fewpy: Loading standard backbone '{model_identifier}' with ImageNet weights...")
            
            constructor = cls.STANDARD_MODELS[model_identifier.lower()]
            model = constructor(weights="DEFAULT")
            
            model = cls.extract_visual_encoder(model, device, keep_avg_pool=keep_avg_pool)
            
            return model.to(device)
        elif model_identifier in model2url:
            
            print(f"Fewpy: Downloading and loading backbone '{model_identifier}' from OpenAI's CLIP repository...")
            model_path = Path(download(model_identifier, cache_dir=cache_dir))
            model = torch.jit.load(model_path).eval().to(device)
            backbone = cls.extract_visual_encoder(model, device, keep_avg_pool=keep_avg_pool)

            return backbone
    
        raise ValueError(f"Identifier '{model_identifier}' is not a standard model.")