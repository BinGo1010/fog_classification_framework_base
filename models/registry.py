MODEL_REGISTRY = {}


def register_model(name: str):
    def deco(cls):
        if name in MODEL_REGISTRY:
            raise KeyError(f"Model {name} already registered")
        MODEL_REGISTRY[name] = cls
        return cls
    return deco


def build_model(cfg):
    name = cfg["model"]["name"]
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    kwargs = {k: v for k, v in cfg["model"].items() if k != "name"}
    return MODEL_REGISTRY[name](**kwargs)
