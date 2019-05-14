import yaml


class Config:
    def __init__(self, config_path):
        self.config = self.load(config_path)

    @staticmethod
    def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=dict):
        class OrderedLoader(Loader):
            pass

        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return object_pairs_hook(loader.construct_pairs(node))

        OrderedLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)

        return yaml.load(stream, OrderedLoader)


    def load(self, config_path):
        with open(config_path, 'r') as f:
            cfg = self.ordered_load(f, yaml.SafeLoader)
        return cfg