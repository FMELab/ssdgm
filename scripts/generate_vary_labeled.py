from argparse import ArgumentParser
import yaml
from pprint import pprint as p
from jinja2 import Environment, FileSystemLoader
import os



def add_1(x):
    return str(int(x) + 1)

func_dict = {
    "add_1": add_1
}

EXP_CONF_DIR = "configs/experiment/varying/labeled"
BASH_DIR = "bash/varying/labeled"

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-e", "--experiment_name", help="name of the experiment")
    args = parser.parse_args()

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yamls")

    with open(os.path.join(yaml_path, "exp_info.yaml")) as file:
        info_dict = yaml.safe_load(file)


    env = Environment(loader=FileSystemLoader(template_path))
    exp_conf_template = env.get_template("varying/labeled/exp_conf.jinja2")
    exp_conf_template.globals.update(func_dict)

    bash_template = env.get_template("varying/labeled/bash_schedule.jinja2")

    #optuna_template = env.get_template("hparams/optuna.jinja2")


    models = info_dict["models"]    
    datamodules = info_dict["datamodules"]
    n_labeled_examples = [100, 200, 300, 400, 500]

    # iterate over the models
    for model in info_dict["models"]:
        # iterate over the datamodules
        bash_template_string = bash_template.render(
            model_name = model["name"],
            datamodules = datamodules,
            n_labeled = n_labeled_examples,
        )

        bash_filename = "_".join(["schedule", model["name"]]) + ".sh"
        bash_filepath = os.path.join(BASH_DIR, bash_filename)
        with open(bash_filepath, "w") as file:
            file.write(bash_template_string)

        for _, dm_info in datamodules.items():
            # iterate over the number of labeled examples (100, 200, 300, 400, 500)
            for n in n_labeled_examples:
                p(exp_conf_template.render(
                    experiment_name=args.experiment_name,
                    model_name=model["name"],
                    datamodule_name=dm_info["name"],
                    features=dm_info["features"],
                    latent=dm_info["latent"],
                    )
                )
                exp_conf_template_string = exp_conf_template.render(
                    experiment_name = args.experiment_name,
                    model_name=model["name"],
                    datamodule_name=dm_info["name"],
                    features=dm_info["features"],
                    latent=dm_info["latent"],
                )

                filename = "_".join([model["name"], dm_info["name"]]) + ".yaml"
                filepath = os.path.join(EXP_CONF_DIR, filename)
                with open(filepath, "w") as file:
                    file.write(exp_conf_template_string)


            #optuna_template_string = optuna_template.render(
            #    model_name=model["name"]
            #)

            #with open("configs/hparams_search/optuna_" + model["name"] + ".yaml", "w") as file:
            #    file.write(optuna_template_string)