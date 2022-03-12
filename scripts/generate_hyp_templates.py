from argparse import ArgumentParser
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



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--experiment_name")
    args = parser.parse_args()

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yamls")

    with open(os.path.join(yaml_path, "exp_info.yaml")) as file:
        info_dict = yaml.safe_load(file)


    env = Environment(loader=FileSystemLoader(template_path))
    hparams_template = env.get_template("hparams/exp.jinja2")
    hparams_template.globals.update(func_dict)

    optuna_template = env.get_template("hparams/optuna.jinja2")

    for model in info_dict["models"]:
        for datamodule in info_dict["datamodules"]:
            p(hparams_template.render(
                model_name=model["name"],
                datamodule_name=datamodule["name"],
                features=datamodule["features"],
                latent=datamodule["latent"],
                experiment_name=args.experiment_name,
                )
            )
            hparams_template_string = hparams_template.render(
                    model_name=model["name"],
                    datamodule_name=datamodule["name"],
                    features=datamodule["features"],
                    latent=datamodule["latent"],
                    experiment_name=args.experiment_name,
            )
            with open("configs/experiment/hyper/" + args.experiment_name + "/" + model["name"] + "_" + datamodule["name"] + ".yaml", "w") as file:
                file.write(hparams_template_string)

            optuna_template_string = optuna_template.render(
                model_name=model["name"]
            )

            with open("configs/hparams_search/optuna_" + model["name"] + ".yaml", "w") as file:
                file.write(optuna_template_string)