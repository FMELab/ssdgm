#!/usr/bin/env python3
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime
import yaml

from jinja2 import Environment, FileSystemLoader

DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
TEMPLATE_FILE = "hparams_exp_kube.jinja2"




def start_config(current_round, info, yaml, dry_run=True):
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    template = Environment(loader=FileSystemLoader(template_path)).get_template(TEMPLATE_FILE)

    for round in info["hyperparameter_tuning"]:
        if current_round == round["round"]:
            for model in round["models"]:
                for datamodule in info["datamodules"]:
                    args_str = "-m hparams_search=optuna_" + model["name"] + " experiment=hyperparameter_search/" + model["name"] + "_" + datamodule["name"]
                    output_text = template.render(args_str=args_str, **yaml)
                    print(output_text)
                    if not dry_run:
                        command = "kubectl -n studerhard create -f -"
                        p = subprocess.Popen(command.split(), stdin=subprocess.PIPE)
                        p.communicate(output_text.encode())
                
                break
        break

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--round")
    args = parser.parse_args()

    current_round = int(args.round)

    date_time = datetime.now().strftime(DATETIME_FORMAT)

    with open("scripts/yamls/info.yaml") as file:
        info_dict = yaml.safe_load(file)

    # Arguments which will be passed to the python script. Boolean flags will be automatically set to "--key" (if True)
    yaml_dict = {
        "job_name": "{}-{}d".format("hyperparameter_search", date_time),
        "image": "ls6-stud-registry.informatik.uni-wuerzburg.de/studerhard/ssdgm-pytorch:0.0.1",
        "cpus": 8,
        "memory": 4,
        "use_gpu": False,
        "script_path": "/ssdgm/train.py"
    }


    start_config(current_round, info_dict, yaml_dict, dry_run=True)