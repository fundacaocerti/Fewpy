from pathlib import Path
import json


def list_pretty_print(items, num=10, per_line=5):
    for i, item in enumerate(items[:num]):
        print(f"{item},", end=" ")
        if (i + 1) % per_line == 0:
            print()

CUB_LENGTH = 11788

def get_data(dataset: str, K: int, W: int, Q: int=0, E: int=0, TRAIN: int=0):
    match dataset.lower():
        case "cub":
            dataset_dir = Path("./datasets").expanduser() / "CUB_200_2011"
            images_dir = dataset_dir / "images"
            cls_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()], key=lambda x: x.name)
            
            CLASSES = [cls_dir.name.split('.')[1] for cls_dir in cls_dirs]
            cls_ids = [i for i in range(len(cls_dirs))]
            map_to_contiguous_ids = {cls_id: idx for idx, cls_id in enumerate(cls_ids)}
            map_back = {idx: cls_id for idx, cls_id in enumerate(cls_ids)}
            cls2int = {k: i for i, k in enumerate(CLASSES)}

            split_path = dataset_dir / "test.json"
            img_id_path = dataset_dir / "images.txt"

            with open(split_path, "r") as f:
                test_split = json.load(f)

            images = [{"path": [], "cls": []}, {"path": [], "cls": []}]
            images[0]["cls"] = test_split["image_labels"]
            for name in test_split["image_names"]:
                name = name.split("/")
                path = dataset_dir
                for p in name[4:]:
                    path = path / p
                images[0]["path"].append(path)

            used_classes = set(test_split["image_labels"])
            
            for cls in images_dir.glob("*"):
                if not cls.is_dir():
                    continue

                for img_path in cls.glob("*"):
                    # if img_path not in used_images:
                    label = int(cls.name.split(".")[0]) - 1

                    if label in used_classes: continue
                    
                    images[1]["path"].append(img_path)
                    images[1]["cls"].append(label)
                
            return {"classes": CLASSES, "data": images[0], "train": images[1], "cls2int": cls2int}

        case "miniimagenet":
            mini_image_net = Path("./datasets").expanduser() / "miniImageNet" 
            mini_image_net = mini_image_net / "1"
            train_split = mini_image_net / "train.pkl"
            
            cache_dir = Path("/home/Fewpy/cache").expanduser()
            
            CLASSES = [
                "ostrich", "robin", "triceratops", "green mamba", 
                "Gila monster", "ruffed grouse", "jellyfish", "Great apes", 
                "Walker hound", "Saluki", "Gordon setter", "Yorkshire terrier", 
                "boxer", "Tibetan mastiff", "French bulldog", "Newfoundland", 
                "miniature poodle", "hyena", "ladybug", "fitch", 
                "rock beauty", "electric ray", "ashcan", "ballpoint pen", 
                "beer glass", "castle", "chime", "coffee mug", 
                "comb", "dining table", "dishrag", "file", 
                "fireboat", "fur coat", "hair slide", "hoopskirt", 
                "loudspeaker", "oboe", "organ", "pajamas", 
                "panpipe", "photocopier", "prayer rug", "reel", 
                "slot machine", "snorkel", "solar dish", "spider web", 
                "stopwatch", "tank", "turtle", "turnstile", 
                "unicycle", "vacuum", "velvet", "worm fence", 
                "yawl", "street sign", "consomme", "hotdog", 
                "orange", "cliff", "bolete", "ear of corn", 
            ]
            
            with open(mini_image_net / "base.json") as f:
                base_split = json.load(f)
            images = [{"path": [], "cls": []}, {"path": [], "cls": []}]
            
            for img, label in zip(base_split["image_names"], base_split["image_labels"]):
                img = Path(img)
                images[1]["path"].append(mini_image_net / img.parent.name / img.name)
                images[1]["cls"].append(label)

            with open(mini_image_net / "val.json") as f:
                val_split = json.load(f)
            
            for img, label in zip(val_split["image_names"], val_split["image_labels"]):
                img = Path(img)
                images[0]["path"].append(mini_image_net / img.parent.name / img.name)
                images[0]["cls"].append(label)
            
            cls_ids = sorted(list(set(images[0]["cls"])))
            assert len(cls_ids) == len(CLASSES) 
            map_to_contiguous_id = {cls: i for i, cls in enumerate(cls_ids)}
            map_back = {v: k for k, v in map_to_contiguous_id.items()}
            cls2int = {k: i for i, k in enumerate(CLASSES)}
                
        case "eurosat":
            import csv
            
            dataset_dir = Path("./datasets").expanduser() / "eurosat" / "EuroSAT"
            with open(dataset_dir / "test.csv", "r") as f:
                reader = csv.reader(f)
                
            next(reader)
            CLASSES = set()
            cls2id = dict()
            images = [{"path": [], "cls": []}, {"path": [], "cls": []}]
            for img_id, filename, label, classname in reader:
                CLASSES.add(classname)
                cls2id[classname] = label
                parent, filename = filename.split("/")
                filename = dataset_dir / parent / filename

                images[0]["path"].append(filename)
                images[0]["cls"].append(label)

            with open(dataset_dir / "train.csv", "r") as f:
                reader = csv.reader(f)
                
            next(reader)
            CLASSES = set()
            cls2id = dict()
            images = [{"path": [], "cls": []}, {"path": [], "cls": []}]
            for img_id, filename, label, classname in reader:
                CLASSES.add(classname)
                cls2id[classname] = label
                parent, filename = filename.split("/")
                filename = dataset_dir / parent / filename

                images[1]["path"].append(filename)
                images[1]["cls"].append(label)
                
            CLASSES = sorted(list(CLASSES))
            cls_ids = [cls2id[cls] for cls in CLASSES]
            map_to_contiguous_ids = {id_: i for i, id_ in enumerate(cls_ids)}
            map_back = {i: id_ for i, id_ in enumerate(cls_ids)}
            cls2int = {cls: i for i, cls in enumerate(CLASSES)}

                
    return {"classes": CLASSES, "data": images[0], "train": images[1], "cls2int": cls2int}