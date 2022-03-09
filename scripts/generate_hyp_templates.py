from pipes import Template
import yaml
from pprint import pprint as p
from jinja2 import Environment, FileSystemLoader, Template
import os


def add_1(x):
    return str(int(x) + 1)

func_dict = {
    "add_1": add_1
}


template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yamls")

with open(os.path.join(yaml_path, "info.yaml")) as file:
    info_dict = yaml.safe_load(file)


env = Environment(loader=FileSystemLoader(template_path))
hparams_template = env.get_template("hparams_exp_template.jinja2")
hparams_template.globals.update(func_dict)

optuna_template = env.get_template("optuna_template.jinja2")

for model in info_dict["models"]:
    for datamodule in info_dict["datamodules"]:
        p(hparams_template.render(
            model_name=model["name"],
            datamodule_name=datamodule["name"],
            features=datamodule["features"],
            latent=datamodule["latent"]
            )
        )
        hparams_template_string = hparams_template.render(
                model_name=model["name"],
                datamodule_name=datamodule["name"],
                features=datamodule["features"],
                latent=datamodule["latent"]
        )
        with open("configs/experiment/hyperparameter_search/" + model["name"] + "_" + datamodule["name"] + ".yaml", "w") as file:
            file.write(hparams_template_string)

        optuna_template_string = optuna_template.render(
            model_name=model["name"]
        )

        with open("configs/hparams_search/optuna_" + model["name"] + ".yaml", "w") as file:
            file.write(optuna_template_string)