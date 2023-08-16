import json

class Config():
  def __init__(self, config_path) -> None:
    with open(config_path,encoding="utf-8") as f:
      self.config = json.load(f)
    print(self.config)
  
  def get(self, key):
    return self.config.get(key, None)
  
  def set(self, key, value):
    self.config[key] = value
  
  def get_all_keys(self):
    return list(self.config.keys())
