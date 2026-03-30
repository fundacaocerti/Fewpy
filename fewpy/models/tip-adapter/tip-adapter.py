from fewpy.models.register import register_constructor
from fewpy.util.download import download
from .base.clip import CLIP

from typing import Union, List
from .config import TipAdapterConfig
from .base.simple_tokenizer import SimpleTokenizer

import warnings

import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch
import sys

from torch import nn

from pathlib import Path


_tokenizer = SimpleTokenizer()
def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class TipAdapter(nn.Module):

    def __init__(self, config):
        super(TipAdapter, self).__init__()

        self.clip = None
        self.config = config

        self.adapter = None
        self.first = not self.config.load_adapter

        self.alpha = self.config.alpha
        self.beta = self.config.beta

        self._tranform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

    def init_adapter(self, in_dim, out_dim):

        if self.adapter is None:
            self.adapter = nn.Linear(in_dim, out_dim, bias=False)

    def train(self, mode=True):

        if self.adapter is not None:
            self.adapter.train(mode)

        return super().train(mode)
    
    def parameters(self, recurse=True):

        if self.adapter is not None:
            yield from self.adapter.parameters(recurse=recurse)

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
  
        if self.adapter is not None:
            yield from self.adapter.named_parameters(
                prefix=prefix + 'adapter.', recurse=recurse, remove_duplicate=remove_duplicate
            )

    def construct_kv(self, images, gt, num_classes):
        if self.config.cache_kv:
            cache_dir = Path(self.config.cache_dir)
            cache_keys = cache_dir / "keys" / f"{self.config.kshot}_shots.pt"
            cache_values = cache_dir / "values" / f"{self.config.kshot}_shots.pt"

            if cache_keys.exists() and cache_values.exists():
                return torch.load(cache_keys), torch.load(cache_values)
            
        with torch.no_grad():
            image_features = self.encode_image(images)

        num_samples = gt.shape[0]
        num_aug = self.config.augment_epoch

        mismatch_msg = "Tensor size mismatch between augmented support images tensor and gt tensor"
        assert image_features.shape[0] == num_aug * num_samples, mismatch_msg

        k = image_features.view(num_aug, num_samples, -1).mean(dim=0)
        k /= k.norm(dim=-1, keepdim=True)
        k = k.permute(1, 0)
        k = k.contiguous()
        v = F.one_hot(gt, num_classes=num_classes).half()

        if self.config.cache_kv:
            cache_keys.parent.mkdir(parents=True, exist_ok=True)
            cache_values.parent.mkdir(parents=True, exist_ok=True)

            torch.save(k, cache_keys)
            torch.save(v, cache_values)

        return k, v
    
    def encode_text(self, prompts, classnames):
        with torch.no_grad():
            clip_weights = []
            target_device = next(self.clip.parameters()).device

            for classname in classnames:
                # Tokenize the prompts
                classname = classname.replace('_', ' ')
                texts = [t.format(classname) for t in prompts]
                texts = tokenize(texts).to(target_device)
                # prompt ensemble for ImageNet
                class_embeddings = self.clip.encode_text(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                clip_weights.append(class_embedding)

            clip_weights = torch.stack(clip_weights, dim=1).to(target_device)

        return clip_weights
    
    def encode_image(self, x):

        with torch.no_grad():
            x = self.clip.encode_image(x)
            x /= x.norm(dim=-1, keepdim=True)

        return x
    
    def combine_logits(self, x, clip_weights, affinity, values):

        clip_logits = 100. * x @ clip_weights
        cache_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ values
        final_logits = cache_logits + clip_logits * self.alpha

        return final_logits
    

    def forward(self, inputs):

        x, s_x, s_y, prompts, classnames = zip(*inputs)

        # construct cache model
        K, V = self.construct_kv(s_x, s_y, len(classnames))

        # extracting textual features with clip
        clip_weights = self.encode_text(prompts, classnames)

        # extract image features
        x = self.encode_image(x)

        affinity = x @ K
        final_logits = self.combine_logits(x, clip_weights, affinity, V)

        return final_logits
    
    def _initialize_adapter(self, k):

        target_device = next(self.clip.parameters()).device
        self.adapter.to(device=target_device)
        k = k.to(target_device)
        self.adapter.half()
        assert self.adapter.weight.shape == k.t().shape
        with torch.no_grad():
            self.adapter.weight.copy_(k.t())
        self.adapter.train()

        self.first = False

    def predict(
            self, 
            x: torch.Tensor, 
            s_x: torch.Tensor, 
            s_y: torch.Tensor, 
            prompts: list[str]=[],
            classnames: list[str]=[],
        ):
        """
        self.model.predict:
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * x: Tensor, batch of images in (B, C, H, W) format.
                * s_x: Tensor, batch of support images in (B, C, H, W) format.
                * s_y: Tensor, batch of ground truth classes in (B) format.
                * prompts: list[str], prompt templates for the classnames to be formatted into.
                * classnames: list[str], name of all novel classes. 
        Returns:
                list[dict]:
                    Each dict is corresponds to the output of a single input image.
                    The dict contains a string "classification" under the key "task" to specify the task type and
                    a "data" class prediction, Tensor of format (1)
        """
        # construct cache model
        K, V = self.construct_kv(s_x, s_y, len(classnames))

        # extracting textual features with clip
        clip_weights = self.encode_text(prompts, classnames)

        # extract image features
        x = self.encode_image(x)

        if self.training:
            if self.first:
                self._initialize_adapter(K)
            affinity = self.adapter(x)

            return self.combine_logits(x, clip_weights, affinity, V)

        affinity = x @ K
        final_logits = self.combine_logits(x, clip_weights, affinity, V)

        final_predictions = torch.argmax(final_logits, dim=1)

        results = []
        for prediction in final_predictions:
            results.append({
                "data": prediction,
                "task": "classification"
            })

            if self.config.show_logits:
                results[-1]["logits"] = final_logits
        
        return results

@register_constructor(name="TipAdapter", config_cls=TipAdapterConfig)
class constructor_TipAdapter():
    def __init__(self, args):
            
        self.args = args

    def instantiate_model(self):

        model = TipAdapter(self.args)
        return self.load_weights(model)
    
    @staticmethod
    def load_from_state_dict(state_dict):

        def _convert_weights_to_fp16(l):
            if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                l.weight.data = l.weight.data.half()
                if l.bias is not None:
                    l.bias.data = l.bias.data.half()

            if isinstance(l, nn.MultiheadAttention):
                for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.half()

            for name in ["text_projection", "proj"]:
                if hasattr(l, name):
                    attr = getattr(l, name)
                    if attr is not None:
                        attr.data = attr.data.half()

        vit = "visual.proj" in state_dict

        if vit:
            vision_width = state_dict["visual.conv1.weight"].shape[0]
            vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
            vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
            grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
            image_resolution = vision_patch_size * grid_size
        else:
            counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
            vision_layers = tuple(counts)
            vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
            output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
            vision_patch_size = None
            assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
            image_resolution = output_width * 32

        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        model = CLIP(
            embed_dim,
            image_resolution, vision_layers, vision_width, vision_patch_size,
            context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
        )

        for key in ["input_resolution", "context_length", "vocab_size"]:
            if key in state_dict:
                del state_dict[key]

        model.apply(_convert_weights_to_fp16)
        model.load_state_dict(state_dict)
        return model

    def load_weights(self, model):

        def load_to_cpu():
            warnings.warn(
                "CUDA was not found or jit.load failed. Inference/Training will run on CPU and may be significantly slower.",
                category=RuntimeWarning,
                stacklevel=2
            )
            state_dict = torch.jit.load(model_path, map_location=device).state_dict()
            clip = constructor_TipAdapter.load_from_state_dict(state_dict)

            return clip, "cpu"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        current_dir = Path(__file__).resolve().parent
        model_path = current_dir / "weights" / f"tip_adapter.pt"
        if not model_path.exists():
            main_dir = Path(sys.path[0])
            model_path = main_dir / "weights" / f"tip_adapter.pt"
        if not model_path.exists():
            model_path = Path(download(self.args.clip_model, new_name="tip_adapter"))

        if torch.cuda.is_available():
            try:
                device = 'cuda'
                clip = torch.jit.load(model_path, map_location=device)
            except RuntimeError:
                clip, device = load_to_cpu()
        else:
            clip, device = load_to_cpu()

        model.clip = clip
        model.clip.to(device)
        model.clip.eval()

        if self.args.load_adapter:
            checkpoint_path = model_path.parent / "tip_adapter.pth"
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Adapter weights not found at {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path)
            k1, k2 = "adapter.weight", "weight"
            if k1 in checkpoint.keys():
                adapter_weights = checkpoint[k1]
            elif k2 in checkpoint.keys():
                adapter_weights = checkpoint[k2]
            else:
                e = f"Could not find adapter weights in state_dict! Weights must be under {k1} or {k2}"
                raise KeyError(e)
            
            out_dim, in_dim = adapter_weights.shape
            model.adapter = nn.Linear(in_dim, out_dim, bias=False)
            model.adapter.weight = nn.Parameter(adapter_weights)
            target_device = next(model.clip.parameters()).device
            target_dtype = next(model.clip.parameters()).dtype

            model.adapter.to(device=target_device, dtype=target_dtype)
        else:
            model.init_adapter(
                in_dim=1024,
                out_dim=self.args.kshot * self.args.n_novel_classes,
            )

        if self.args.training:
        
            return model.train(), device

        return model.eval(), device
