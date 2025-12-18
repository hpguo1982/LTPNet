# components.py
from . import register_class, AIITModel


#@register_class
class Database(AIITModel):
    def __init__(self, host, port):
        self.host = host
        self.port = port

    def __repr__(self):
        return f"<Database host={self.host} port={self.port}>"

#@register_class
class Cache(AIITModel):
    def __init__(self, backend, size):
        self.backend = backend
        self.size = size

    def __repr__(self):
        return f"<Cache backend={self.backend} size={self.size}>"

#@register_class
class Service(AIITModel):
    def __init__(self, name, database, cache):
        self.name = name
        self.database = database
        self.cache = cache

    def __repr__(self):
        return f"<Service name={self.name} db={self.database} cache={self.cache}>"

# main.py
from . import load_config, build_from_config

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    obj = build_from_config(cfg)  # 返回 Service 实例
    print(obj)

