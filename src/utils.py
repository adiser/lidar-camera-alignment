import yaml


class LabelConfig:
    # REL_LABELS_COUNT = [
    #     (40, 34228),
    #     (70, 27123),
    #     (48, 26360),
    #     (50, 18268),
    #     (0, 6506),
    #     (44, 3268),
    #     (72, 2964),
    #     (52, 1471),
    #     (99, 1227),
    #     (71, 1192),
    # ]

    #

    LABEL_MAP = {
        'car': 10,
        'road': 40,
        'sidewalk': 48
    }

    data_source = '../data/Kitti/dataset/semantic-kitti.yaml'
    yaml_dict = yaml.safe_load(open(data_source))

    @classmethod
    def labels(cls):
        return cls.LABEL_MAP.values()

    @classmethod
    def label_map(cls):
        return cls.LABEL_MAP

    @classmethod
    def label_names(cls):
        return cls.LABEL_MAP.keys()

    @classmethod
    def color(cls, label: int):
        # BGR to RGB.
        return cls.yaml_dict['color_map'][label][::-1]


DOWNSCALING_FACTOR = 1000


def load_semkitti_config(data_source: str = '../data/semantic-kitti.yaml'):
    yaml_dict = yaml.safe_load(open(data_source))
    return yaml_dict


