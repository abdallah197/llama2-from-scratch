from config import DataArgs, ModelArgs, TrainArgs, InferenceArgs


def get_args():
    """ parse all config args into one place """
    data_args = DataArgs()
    model_args = ModelArgs()
    train_args = TrainArgs()
    inference_args = InferenceArgs()
    deepspeed_args = DataArgs()

    args = {}

    for args in [model_args, train_args, data_args, inference_args, deepspeed_args]:
        args.update(vars(args))
    return args
