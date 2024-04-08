import json

from config import DataArgs, ModelArgs, TrainArgs, InferenceArgs, DeepspeedArgs


def get_args():
    """ parse all config args into one place """

    data_args = DataArgs()
    model_args = ModelArgs()
    train_args = TrainArgs()
    inference_args = InferenceArgs()
    deepspeed_args = DeepspeedArgs()

    args = {}

    for config_args in [model_args, train_args, data_args, inference_args, deepspeed_args]:
        args.update(vars(config_args))

    # load deepspeed config
    with open(args['deepspeed_config'], 'r') as cfg:
        deepspeed_cfg = json.load(cfg)
    args.update(deepspeed_cfg)

    return args
