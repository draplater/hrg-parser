import importlib
from argparse import Namespace
from collections import OrderedDict

from logging import FileHandler

from multiprocessing import Process, Lock

import os

from logger import logger


def dict_to_commandline(dic, prefix=()):
    option_cmd = list(prefix)
    for k, v in dic.items():
        assert isinstance(k, str)
        if v is True:
            option_cmd.append("--" + k)
        elif v is False:
            continue
        else:
            option_cmd.append("--" + k)
            if isinstance(v, list):
                option_cmd.extend(str(i) for i in v)
            else:
                option_cmd.append(str(v))

    return option_cmd


def parse_cmd_multistage(dep_parser_class, cmd):
    namespace = Namespace()
    arg_parser = dep_parser_class.get_arg_parser()
    _, rest_cmd = arg_parser.parse_known_args(cmd, namespace)
    stage = 1
    while True:
        next_arg_parser = dep_parser_class.get_next_arg_parser(stage, namespace)
        if next_arg_parser is None:
            if rest_cmd:
                try:
                    from gettext import gettext as _
                except ImportError:
                    def _(message):
                        return message
                msg = _('unrecognized arguments: %s')
                arg_parser.error(msg % ' '.join(rest_cmd))
            else:
                return namespace
        stage += 1
        _, rest_cmd = next_arg_parser.parse_known_args(rest_cmd, namespace)
        arg_parser = next_arg_parser


def parse_dict_multistage(dep_parser_class, dic, prefix=()):
    return parse_cmd_multistage(dep_parser_class, dict_to_commandline(dic, prefix))


class TrainingScheduler(object):
    """
    Run multiple instance of trainer.
    """

    def __init__(self, train_func, dep_parser_class, train=None, dev=None, test=None):
        self.train = train
        self.dev = dev
        self.test = test
        self.train_func = train_func
        self.dep_parser_class = dep_parser_class
        self.all_options = OrderedDict()

    def add_options(self, title, options_dict, outdir_prefix=""):
        options_dict["title"] = title
        options_dict["outdir"] = os.path.join(outdir_prefix, "model-" + title)
        options = parse_dict_multistage(self.dep_parser_class, options_dict, ["train"])
        self.train_func = options.func
        self.all_options[title] = options

    def run_parallel(self):
        if len(self.all_options) == 1:
            self.run()
            return

        processes = {}
        for title, options in self.all_options.items():
            print("Training " + title)
            processes[title] = Process(target=self.train_func,
                                       args=(options, self.train, self.dev, self.test))

        try:
            for index, process in processes.items():
                process.start()
            for index, process in processes.items():
                process.join()
        except KeyboardInterrupt:
            for index, process in processes.items():
                process.terminate()

    def run(self):
        for title, options in self.all_options.items():
            logger.info("Training " + title)
            self.train_func(options, self.train, self.dev, self.test)
            for handler in logger.handlers:
                if isinstance(handler, FileHandler):
                    logger.removeHandler(handler)


def lazy_run_parser(module_name, class_name, title, options_dict, outdir_prefix,
                    initializer_lock, mode="train", initializer=None):
    if mode == "train":
        options_dict["title"] = title
        options_dict["outdir"] = os.path.join(outdir_prefix, "model-" + title)

    if initializer is not None:
        with initializer_lock:
            initializer(options_dict)

    dep_parser_class = getattr(importlib.import_module(module_name), class_name)
    options = parse_dict_multistage(dep_parser_class, options_dict, [mode])
    options.func(options)


class LazyLoadTrainingScheduler(object):
    """
    Run multiple instance of trainer.
    """

    def __init__(self, module_name, class_name, initializer=None):
        self.module_name = module_name
        self.class_name = class_name
        self.all_options_and_outdirs = OrderedDict()
        self.initializer = initializer

    @classmethod
    def of(cls, parser_class, initializer=None):
        return cls(parser_class.__module__, parser_class.__name__, initializer)

    def add_options(self, title, options_dict, outdir_prefix="", mode="train"):
        self.all_options_and_outdirs[title, outdir_prefix, mode] = dict(options_dict)

    def run_parallel(self):
        initializer_lock = Lock()
        if len(self.all_options_and_outdirs) == 1:
            self.run()
            return

        processes = {}
        for (title, outdir_prefix, mode), options_dict in self.all_options_and_outdirs.items():
            print("Training " + title)
            processes[title, outdir_prefix] = Process(target=lazy_run_parser,
                                       args=(self.module_name, self.class_name, title,
                                             options_dict, outdir_prefix, initializer_lock,
                                             mode, self.initializer)
                                       )

        try:
            for index, process in processes.items():
                process.start()
            for index, process in processes.items():
                process.join()
        except KeyboardInterrupt:
            for index, process in processes.items():
                process.terminate()

    def run(self):
        for (title, outdir_prefix, mode), options_dict in self.all_options_and_outdirs.items():
            logger.info("Training " + title)
            if self.initializer is not None:
                self.initializer(options_dict)
            lazy_run_parser(self.module_name, self.class_name, title,
                            options_dict, outdir_prefix, None, mode)
            for handler in logger.handlers:
                if isinstance(handler, FileHandler):
                    logger.removeHandler(handler)

