from utils.Experiment import Experiment
from utils.functions import fix_random_seed
from utils.ExperimentArgs import ExperimentArgs
from utils.logger import logger_wrapper


def main() -> None:
    exp_args = ExperimentArgs()
    exp_args.save_args()
    logger_wrapper.set_file_log(exp_args.get_save_path())
    random_seed = exp_args['random_seed']
    fix_random_seed(random_seed)
    experiment = Experiment(exp_args)
    if exp_args['train_test']:
        experiment.train()
        experiment.test()
    else:
        model_save_path = exp_args['model_save_path']
        experiment.load_model(model_save_path)
        experiment.test()


if __name__ == '__main__':
    main()