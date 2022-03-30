#!/usr/bin/env python3
import os
import subprocess
from argparse import ArgumentParser
from datetime import datetime
import yaml
import time

from jinja2 import Environment, FileSystemLoader

DATETIME_FORMAT = "%Y-%m-%d-%H-%M-%S"
TEMPLATE_FILE = "varying/labeled/kube_vary_labeled.jinja2"

EXPERIMENT_DIR = "varying/labeled/"

models = ["mlp", "autoencoderregressor",
          "m2vae", "deterministicpredictor", "probabilisticpredictor",
          "srgan", "semigan",
          "ssdkl"]


def start_config(run_info, model_info, yaml, dry_run=True):
    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    template = Environment(loader=FileSystemLoader(template_path)).get_template(TEMPLATE_FILE)



    job_name_base_string = yaml["job_name"]
    script_path_base_string = yaml["script_path"]
    for m in models:
        job_name = job_name_base_string + m
        yaml["job_name"] = job_name

        script_filename = "schedule_" + m + ".sh"
        yaml["script_path"] = script_path_base_string + script_filename

        output_text = template.render(
            **yaml
        )

        print(output_text)
        if not dry_run:
            try:
                command = "kubectl -n studerhard create -f -"
                p = subprocess.Popen(command.split(), stdin=subprocess.PIPE)
                p.communicate(output_text.encode())
                time.sleep(30)
            except:
                print(f'Could not start job for varying labeled data for model: {m}')

    
    

if __name__ == '__main__':


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
        "memory": 12,
        "use_gpu": False,
        "script_path": "/workspace/ssdgm/bash/varying/labeled/"
    }


    start_config(run_info, model_info, yaml_dict, dry_run=True)