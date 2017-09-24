import argparse

class Parser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--path", default="../", help="Base Path for the Folder")
        parser.add_argument("--project", default="NASuRL", help="Project Folder")
        parser.add_argument("--folder_suffix", default="Default", help="Folder Name Suffix")
        parser.add_argument("--dataset", default="cifar-10", help="Name of the Dataset")
        parser.add_argument("--num_classes", default=10, type=int, help="Number of Classes")
        parser.add_argument("--opt", default="sgd", help="Optimizer : adam, rmsprop, sgd, normal")
        parser.add_argument("--hyperparams", default=10, help="Number of Hyperparameters to search")
        parser.add_argument("--lr", default=0.1, help="Learning Rate", type=float)
        parser.add_argument("--batch_size", default=75, help="Batch Size", type=int)
        parser.add_argument("--dropout", default=0.5, help="Dropout Probab. for Pre-Final Layer", type=float)
        parser.add_argument("--max_epochs", default=100, help="Maximum Number of Epochs", type=int)
        parser.add_argument("--debug", default=False, type=self.str_to_bool, help="Debug Mode")
        parser.add_argument("--load", default=False, type=self.str_to_bool, help="Load Model to calculate accuracy")
        self.parser=parser

    def str_to_bool(self, string):
        if string.lower() == "true":
            return True
        elif string.lower() == "false":
            return False
        else :
            return argparse.ArgumentTypeError("Boolean Value Expected")

    def get_parser(self):
        return self.parser