from fewpy.models.Qwen.config import QwenConfig
from fewpy.util.inference.register import register_constructor

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

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

    def predict(self, x, s_x=None, s_y=None, single_cls: str=None):

        prompt = self.prompt
        if single_cls is not None:
            prompt = f"Outline the position of objects of class={single_cls}. Then output all the coordinates and classes of these objects in JSON format."

        messages = []
        if (not s_x is None) and (s_y is None):
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
                            "text": "{\"bbox_2d\":" + f"{annot["bboxes"]}" + ", \"label\": \"" + f"{cls}" + "\"}"
                        }
                    ]
                }

                messages.append(usr_msg)
                messages.append(target_msg)

        task_msg = {
            "role": "user",
            "content": [
                {"type": "image", "image": x},
                {"type": "text", "text": prompt}
            ],
        }
        messages.append(task_msg)

        image_inputs, video_inputs = process_vision_info(messages)

        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.inference_mode():
            gen = self.model.generate(
                **inputs,
                max_new_tokens=256,
        )

        trimmed = [g[len(i):] for i, g in zip(inputs.input_ids, gen)]
        text = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        text = text.replace("```json", "").replace("```", "").strip()
        results = []
        try:
            C, H, W = x.shape
            datections = sv.Detections.from_vlm(
                vlm=sv.VLM.QWEN_3_VL,
                result=output,
                resolution_wh=[H, W]
            )
            for bbox, _, conf, label, _, _ in datections:
                result = {
                    "task": "detection",
                    "conf": float(conf),
                    "data": bbox.tolist(),
                }
                if label is not None:
                    result["label"] = label
                results.append(result)
        except Exception as e:
            print("output is not json compatible")
            output = {"out": text, "error": e}

        return results
        

@register_constructor(name="Qwen", config_cls=QwenConfig)
class contructor_Qwen:

    model_cls_mame = "QwenWrapper"
 
    def __init__(self, config: QwenConfig):
        
        self.config = config

    def instantiate_model(self):

        model_id = "Qwen/Qwen3-VL-8B-Instruct"

        model = AutoModelForImageTextToText.from_pretrained(
            model_id, 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)

        model_wrapper = QwenWrapper(
            cfg=self.config,
            model=model,
            processor=processor
        )

        return model_wrapper, "auto"
