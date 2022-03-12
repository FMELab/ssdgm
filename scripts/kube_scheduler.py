#!/usr/bin/env python3
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime
import yaml

from jinja2 import Environment, FileSystemLoader

DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
TEMPLATE_FILE = "hparams/kube.jinja2"




def start_config(args, info, yaml, dry_run=True):
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    template = Environment(loader=FileSystemLoader(template_path)).get_template(TEMPLATE_FILE)

    job_name_base_string = yaml["job_name"]
    for round in info["hyper"]["hyperparameter_tuning"]:
        if args.round == round["round"]:
            for model in round["models"]:
                for datamodule in info["exp"]["datamodules"]:  
                    yaml["job_name"] = f"{job_name_base_string}|{model['name']}-{datamodule['name']}"
                    args_str = "-m" + ", hparams_search=optuna_" + model["name"] + ", experiment=hyper/" + args.experiment_name + "/" + model["name"] + "_" + datamodule["name"]
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
    # `--round` is needed because 3 models (AE+R, VAE+detR, VAE+probR) 
    # depend on fully trained components (AE, VAE) before the remaining 
    # component can be trained
    parser.add_argument("--round")
    parser.add_argument("--experiment_name")
    args = parser.parse_args()


    date_time = datetime.now().strftime(DATETIME_FORMAT)

    with open("scripts/yamls/hyper_info.yaml") as file:
        hyper_info_dict = yaml.safe_load(file)

    with open("scripts/yamls/exp_info.yaml") as file:
        exp_info_dict = yaml.safe_load(file)

    info_dict = {"hyper": hyper_info_dict, "exp": exp_info_dict}

    # Arguments which will be passed to the python script. Boolean flags will be automatically set to "--key" (if True)
    yaml_dict = {
        "job_name": "{}|{}".format("hparams-search", date_time),
        "image": "ls6-stud-registry.informatik.uni-wuerzburg.de/studerhard/ssdgm-pytorch:0.0.3",
        "cpus": 8,
        "memory": 4,
        "use_gpu": False,
        "script_path": "/workspace/ssdgm/train.py"
    }


    start_config(args, info_dict, yaml_dict, dry_run=True)