import json

class ConfigParser:
    def __init__(self, filepath):
        self.filepath = filepath
        self.parse_config()

    def parse_config(self):
        with open(self.filepath, 'r') as file:
            config = json.load(file)
            self.dict = config
        
        for key, value in config.items():
            setattr(self, key, value)
    
    def get_dict(self):
        return self.dict


# Usage example
config_parser = ConfigParser('/home/hari/Documents/CMU_masters/targetless_lidar_camera_calibration/lidar-camera-alignment/src/cfg/config.json')
# print(config_parser.rot_param_type)
# print(config_parser.use_depth)
# print(config_parser.lr)
# Access other variables in a similar way