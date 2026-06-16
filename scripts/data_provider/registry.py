DATASET_REGISTRY = {}


def register_dataset(name: str):
    def deco(cls):
        if name in DATASET_REGISTRY:
            raise KeyError(f"Dataset {name} already registered")
        DATASET_REGISTRY[name] = cls
        return cls
    return deco


def build_dataset(name: str, **kwargs):
    if name not in DATASET_REGISTRY:
        raise KeyError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[name](**kwargs)
