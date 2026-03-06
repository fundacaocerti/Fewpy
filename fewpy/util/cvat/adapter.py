import cvat_sdk.auto_annotation as cvataa
import numpy as np


class CVATAdapter:
    @staticmethod
    def to_cvat(standard_output, label_map):

        results = []
        
        for item in standard_output:
            label_name = item.get("label")
            label_id = item.get("label_id")
            if label_id is None and label_name is not None:
                if label_name in label_map:
                    label_id = label_map[label_name]
                else:
                    label_id = list(label_map.values())[0] if label_map else 0

            match item:
                case {"task": "segmentation", "data": mask_tensor}:
                    results.append(
                        cvataa.mask(
                            label_id=label_id,
                            points=CVATAdapter.tensor_to_cvat_mask(mask_tensor)
                        )
                    )
                
                case {"task": "detection", "data": bbox}:
                    results.append(
                        cvataa.rectangle(
                            label_id=label_id,
                            points=list(bbox)
                        )
                    )
                case _:
                    print(f"Warning: Unknown task type or malformed data in {item}")

        return results
    
    @staticmethod
    def tensor_to_cvat_mask(mask_tensor):
        mask = mask_tensor.detach().cpu().numpy().squeeze().astype(np.uint8)
        
        pos = np.where(mask > 0)
        if len(pos[0]) == 0:
            print("empty mask")
            return None
        
        ymin, xmin = np.min(pos[0]), np.min(pos[1])
        ymax, xmax = np.max(pos[0]), np.max(pos[1])
        
        mask_crop = mask[ymin:ymax+1, xmin:xmax+1]
        
        mask_data = mask_crop.flatten().tolist()
        
        return [float(xmin), float(ymin), float(xmax), float(ymax)] + mask_data
