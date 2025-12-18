# loader.py
import yaml
from . import CLASS_REGISTRY


def build_from_config(cfg):
    """
    递归地根据配置实例化对象。
    - cfg 是字典，必须包含 'class' 或只有基本字段
    """
    if isinstance(cfg, dict) and "class" in cfg:
        cls_name = cfg["class"]
        params = {k: build_from_config(v) for k, v in cfg.items() if k != "class"}
        cls = CLASS_REGISTRY.get(cls_name)
        if cls is None:
            raise ValueError(f"Class {cls_name} not found in registry")
        return cls(**params)
    elif isinstance(cfg, dict):
        # 普通字典，递归解析其值
        return {k: build_from_config(v) for k, v in cfg.items()}
    elif isinstance(cfg, list):
        return [build_from_config(item) for item in cfg]
    else:
        return cfg


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
