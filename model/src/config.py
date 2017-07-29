import os
import utils
import tensorflow as tf

class Config(object):
    def __init__(self, args):
        self.codebase_root_path = args.path
        self.folder_suffix = args.folder_suffix
        self.project_name = args.project
        self.dataset_name = args.dataset
        self.batch_size = args.batch_size
        self.max_epochs = args.max_epochs
        self.hyperparams = args.hyperparams
        class Solver(object):
            def __init__(self, t_args):
                self.learning_rate = t_args.lr
                self.dropout = t_args.dropout
                if t_args.opt.lower() not in ["adam", "rmsprop", "sgd"]: 
                    raise ValueError('Undefined type of optmizer')
                else:  
                    self.optimizer = {"adam": tf.train.AdamOptimizer, "rmsprop": tf.train.RMSPropOptimizer, "sgd": tf.train.GradientDescentOptimizer, "normal": tf.train.Optimizer}[t_args.opt.lower()]
        
        self.solver = Solver(args)
        self.project_path, self.project_prefix_path, self.dataset_path, self.train_path, self.test_path, self.ckptdir_path = self.set_paths()

    def set_paths(self):
        project_path = utils.path_exists(self.codebase_root_path)
        project_prefix_path = "" #utils.path_exists(os.path.join(self.codebase_root_path, self.project_name, self.folder_suffix))
        dataset_path = utils.path_exists(os.path.join(self.codebase_root_path, "../data", self.dataset_name))
        ckptdir_path = utils.path_exists(os.path.join(self.codebase_root_path, "checkpoint"))
        train_path = os.path.join(dataset_path, "data_batch_")
        test_path = os.path.join(dataset_path, "test_batch")

        return project_path, project_prefix_path, dataset_path, train_path, test_path, ckptdir_path
