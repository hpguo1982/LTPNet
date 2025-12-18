import importlib
import pkgutil
import types
from typing import Dict, Any


def import_symbols(package_name: str) -> Dict[str, Any]:
    """
    递归导入 package 及其子包，只获取 __all__ 中声明的对象
    返回一个字典 {name: object}
    """
    symbols = {}

    # 先导入包本身
    pkg = importlib.import_module(package_name)

    # 取出包本身 __all__ 中声明的对象
    if hasattr(pkg, "__all__"):
        for name in pkg.__all__:
            try:
                symbols[name] = getattr(pkg, name)
            except AttributeError:
                pass

    # 遍历子模块/子包
    if hasattr(pkg, "__path__"):  # 有 __path__ 表示是包
        for _, subname, ispkg in pkgutil.iter_modules(pkg.__path__):
            full_name = f"{package_name}.{subname}"
            if ispkg:
                # 递归导入子包
                symbols.update(import_symbols(full_name))
            else:
                # 普通模块：先导入，再看模块自身的 __all__
                mod = importlib.import_module(full_name)
                if hasattr(mod, "__all__"):
                    for name in mod.__all__:
                        symbols[name] = getattr(mod, name, None)

    return symbols


if __name__ == "__main__":
    result = import_symbols("models")
    for k, v in result.items():
        print(f"{k}: {v}")
