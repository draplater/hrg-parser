import os
import re

import shutil

from common_utils import ensure_dir
from logger import logger
from select_best import select_best, formats


def k_fold_validation(train_file, dev_file, op, FormatClass,
                      project_name, outdir_prefix, scheduler, k=5,
                      prevent_redundant_preparation=True,
                      header=None):
    train_file_basename = os.path.basename(train_file)
    train_file_prefix, _, ext = train_file_basename.rpartition(".")
    train_sents = FormatClass.from_file(train_file)
    project_dir = os.path.join(outdir_prefix, project_name)
    ensure_dir(project_dir)
    train_file_i = os.path.join(project_dir, train_file_prefix + ".{}." + ext)
    train_file_except_i = os.path.join(project_dir, train_file_prefix + ".except-{}." + ext)
    data_preparation_done_file = os.path.join(
        project_dir, "." + train_file_basename + ".done")

    # do data preparation
    if not prevent_redundant_preparation or not os.path.exists(data_preparation_done_file):
        train_sents_splitted = []
        for i in range(k):
            start = int(i * len(train_sents) / k)
            end = int((i + 1) * len(train_sents) / k)
            train_sents_splitted.append(train_sents[start:end])

        f_train_list = [open(train_file_i.format(i), "w") for i in range(k)]
        f_train_except_list = [open(train_file_except_i.format(i), "w") for i in range(k)]
        if header is not None:
            for f_i in f_train_list:
                f_i.write(header + "\n")
            for f_i in f_train_except_list:
                f_i.write(header + "\n")
        for i, train_sents_i in enumerate(train_sents_splitted):
            for sent in train_sents_i:
                for j in range(k):
                    if j == i:
                        f_train_list[j].write(sent.to_string())
                    else:
                        f_train_except_list[j].write(sent.to_string())
        for f_i in f_train_list:
            f_i.close()
        for f_i in f_train_except_list:
            f_i.close()
        with open(data_preparation_done_file, "w") as f:
            f.write("Done!")
        logger.info("{}-fold data preparation done!".format(k))
    else:
        logger.info("No need to prepare {}-fold data.".format(k))

    # create training tasks
    for i in range(k):
        op_i = dict(op)
        op_i["train"] = train_file_except_i.format(i)
        op_i["dev"] = dev_file
        scheduler.add_options("except-{}".format(i), op_i, project_dir)


def k_fold_select_best(project_dir, data_format, key, scheduler,
                       header=None, delete_poor_model=False):
    # detect k
    k = len([i for i in os.listdir(project_dir)
             if i.startswith("model-")])
    # detect train file prefix
    train_file_prefix, train_file_ext = [
        i for i in os.listdir(project_dir)
        if i.find(".except-0.") >= 0][0].split(".except-0.")

    # select best model
    best_models = {}
    for i in range(k):
        task_dir = os.path.join(project_dir, "model-except-{}".format(i))
        score_files = [os.path.join(task_dir, j) for j in os.listdir(task_dir)
                       if j.endswith(".txt") and not j.endswith(".txt.txt")]
        best_performance, best_epoch = select_best(
            data_format, score_files, key, 1)
        logger.info("Select epoch {} of task {} with {}={}".format(
            best_epoch, i, key, best_performance))
        best_model = best_models[i] = os.path.join(task_dir, "model.{}".format(best_epoch))
        # use dev file of part 0 as parsed dev file
        if i == 0:
            dev_file = [os.path.join(task_dir, j) for j in os.listdir(task_dir)
                        if re.search(r"_epoch_{}.{}$".format(best_epoch, train_file_ext), j)][0]
            dev_final_file = os.path.join(project_dir, "dev.parsed." + train_file_ext)
            shutil.copyfile(dev_file, dev_final_file)
        if delete_poor_model:
            model_files = sorted(os.path.join(task_dir, j)
                                 for j in os.listdir(task_dir)
                                 if re.match(r"^model\.\d+(\.data)?$", j))
            for model_file in model_files:
                if model_file != best_model:
                    os.remove(model_file)

    # predict rest part
    output_file = os.path.join(
        project_dir,
        train_file_prefix + ".parsed.{}." + train_file_ext)
    for i, best_model_i in best_models.items():
        train_part = os.path.join(
            project_dir,
            train_file_prefix + ".{}.".format(i) + train_file_ext)
        op = {"model": best_model_i, "test": train_part,
              "output": output_file.format(i)}
        scheduler.add_options(str(i), op, "", "predict")
    scheduler.run_parallel()

    # merging output files
    logger.info("Merging files...")
    merged_file = os.path.join(
        project_dir,
        train_file_prefix + ".merged." + train_file_ext)
    with open(merged_file, "w") as f:
        if header:
            f.write(header + "\n")
        for i in range(k):
            sents = formats[data_format].from_file(output_file.format(i))
            for sent in sents:
                f.write(sent.to_string())
