#!/usr/bin/env python3
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime
import yaml
import time

from jinja2 import Environment, FileSystemLoader

DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
TEMPLATE_FILE = "hparams/kube.jinja2"

EXPERIMENT_DIR = "varying/labeled/"


def start_config(args, run_info, model_info, yaml, dry_run=True):
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    template = Environment(loader=FileSystemLoader(template_path)).get_template(TEMPLATE_FILE)

    n_examples = run_info["experiment"]["n_labeled_examples"]
    run_groups = run_info["experiment"]["run_groups"]

    datamodules = model_info["datamodules"]

    job_name_base_string = yaml["job_name"]
    for run_group in run_groups:
        # select the specified run group as some models have to be trained separately
        if args.run_group == run_group["name"]:
            for model in run_group["models"]:
                for datamodule in datamodules:
                    for n in n_examples:
                        job_name = f"{job_name_base_string}-{model}-{datamodule['name']}-{n}"
                        yaml["job_name"] = job_name

                        experiment_filename = "_".join([model, datamodule["name"], str(n)])
                        args_str = f"experiment={EXPERIMENT_DIR + experiment_filename}"

                        output_text = template.render(args_str=args_str, datamodule_name=datamodule["name"], **yaml)
                        print(output_text)
                        if not dry_run:
                            try:
                                command = "kubectl -n studerhard create -f -"
                                p = subprocess.Popen(command.split(), stdin=subprocess.PIPE)
                                p.communicate(output_text.encode())
                                time.sleep(30)
                            except:
                                print(f'Could not start job for {model["name"]} - {datamodule["name"]}')

if __name__ == '__main__':
    parser = ArgumentParser()
    # `--round` is needed because 3 models (AE+R, VAE+detR, VAE+probR) 
    # depend on fully trained components (AE, VAE) before the remaining 
    # component can be trained
    #parser.add_argument("--experiment_name")
    parser.add_argument("--run_group")
    args = parser.parse_args()


    date_time = datetime.now().strftime(DATETIME_FORMAT)

    with open("scripts/yamls/run_info.yaml") as file:
        run_info = yaml.safe_load(file)

    with open("scripts/yamls/exp_info.yaml") as file:
        model_info = yaml.safe_load(file)

    #info_dict = {"hyper": hyper_info_dict, "exp": exp_info_dict}

    # Arguments which will be passed to the python script. Boolean flags will be automatically set to "--key" (if True)
    yaml_dict = {
        "job_name": "{}".format("vary-labeled"),
        "image": "ls6-stud-registry.informatik.uni-wuerzburg.de/studerhard/ssdgm-pytorch:0.0.5",
        "cpus": 12,
        "cpus_big_datasets": 24,
        "memory": 12,
        "memory_big_datasets": 32,
        "use_gpu": False,
        "script_path": "/workspace/ssdgm/train.py"
    }


    start_config(args, run_info, model_info, yaml_dict, dry_run=True)