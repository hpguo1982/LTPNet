# registry.py
from torch.nn import Module

CLASS_REGISTRY = {}


def register_class(cls):
    """装饰器：把类注册到全局字典"""
    CLASS_REGISTRY[cls.__name__] = cls
    return cls


# 元类
class AutoRegisterMeta(type):
    def __new__(mcs, name, bases, attrs):
        cls = super().__new__(mcs, name, bases, attrs)
        if name != "BaseModel":  # 避免注册基类
            register_class(cls)
        return cls


class AIITModel(metaclass=AutoRegisterMeta):
    pass
