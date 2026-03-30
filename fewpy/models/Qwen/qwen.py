from fewpy.models.Qwen.config import QwenConfig
from fewpy.models.register import register_constructor

from transformers import BitsAndBytesConfig, AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

import PIL
from typing import List, Dict

import supervision as sv
import torch


class QwenWrapper:

    def __init__(self, cfg, model: torch.nn.Module, processor):
        self.cfg = cfg
        self.model = model
        self.processor = processor

        classnames = ", ".join(self.cfg.classnames)
        classnames = " and ".join(classnames.rsplit(", ", 1))

        self.prompt = f"Outline the position of objects of classes: {classnames}. Then output all the coordinates and classes of these objects in JSON format."
        
    def __str__(self):

        return str(self.model)

    def train(self, mode: bool=True):

        return self.model.train(mode)

    def eval(self):

        return self.train(False)

    def parameters(self, recurse=True):

        yield from self.model.parameters(recurse)

    def named_parameters(self, prefix="", recurse=True, remove_duplicate=True):

        yield from self.model.named_parameters(prefix, recurse, remove_duplicate)

    def predict(self, x: List[PIL.Image], s_x: List[PIL.Image]=None, s_y: List[Dict]=None, single_cls: str=None):

        prompt = self.prompt
        if single_cls is not None:
            prompt = f"Outline the position of objects of class={single_cls}. Then output all the coordinates and classes of these objects in JSON format."

        messages = []
        if (s_x is not None) and (s_y is not None):
            for example, annot in zip(s_x, s_y):
                usr_msg = {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": example},
                        {"type": "text", "text": prompt}
                    ],
                }
                target_msg = {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text", 
                            "text": "{\"bbox_2d\":" + f"{annot["bboxes"]}" + ", \"label\": \"" + f"{annot["cls"]}" + "\"}"
                        }
                    ]
                }
                messages.extend([usr_msg, target_msg])

        batched_msgs = []
        for xi in x:
            task_msg = {
                "role": "user",
                "content": [
                    {"type": "image", "image": xi},
                    {"type": "text", "text": prompt}
                ],
            }
            batched_msgs.append(messages + [task_msg])

        image_inputs, video_inputs = process_vision_info(batched_msgs)

        texts = [self.processor.apply_chat_template(
            msgs, 
            tokenize=False, 
            add_generation_prompt=True
        ) for msgs in batched_msgs]

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        if self.model.training:

            inputs["labels"] = inputs["input_ids"].clone()
            outputs = self.model(**inputs)

            return outputs.loss

        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=256,
        )

        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
        decoded_texts = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        results = []
        for xi, text in zip(x, decoded_texts):
            image_results = []
            text = text.replace("```json", "").replace("```", "").strip()
            try:
                datections = sv.Detections.from_vlm(
                    vlm=sv.VLM.QWEN_3_VL,
                    result=text,
                    resolution_wh=xi.size
                )
                for bbox, _, conf, label, _, _ in datections:
                    result = {
                        "task": "detection",
                        "conf": float(conf),
                        "data": bbox.tolist(),
                    }
                    if label is not None:
                        result["label"] = label
                    image_results.append(result)
                
            except Exception as e:
                print("Output is not json compatible!", e)

            results.append(image_results)

        return results
        

@register_constructor(name="Qwen", config_cls=QwenConfig)
class contructor_Qwen:

    model_cls_mame = "QwenWrapper"
 
    def __init__(self, config: QwenConfig):
        
        self.config = config

    def instantiate_model(self):

        model_id = "Qwen/Qwen3-VL-8B-Instruct"

        if self.config.quantization:
            if self.config.quantization_bits == 4:

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self.config.torch_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                )
            
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map={"": 0}, # Forces model to GPU 0, avoiding the "meta" device bug
                torch_dtype=self.config.torch_dtype,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_id, 
                device_map="cuda"
            )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.image_processor.max_pixels = self.config.max_pixels
        processor.image_processor.min_pixels = self.config.min_pixels
    
        if self.config.lora:
            
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                target_modules=self.config.target_modules,
                lora_dropout=self.config.lora_dropout,
                bias=self.config.lora_bias,
                task_type="CAUSAL_LM",
            )
            
            model = get_peft_model(model, lora_config)

        if self.config.gradient_checkpointing:

            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        
        model_wrapper = QwenWrapper(
            cfg=self.config,
            model=model,
            processor=processor
        )

        return model_wrapper, "cuda"
