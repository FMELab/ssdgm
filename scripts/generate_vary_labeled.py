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

TO_DIR = "configs/experiment/varying/labeled"

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-e", "--experiment_name", help="name of the experiment")
    args = parser.parse_args()

    template_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yamls")

    with open(os.path.join(yaml_path, "exp_info.yaml")) as file:
        info_dict = yaml.safe_load(file)


    env = Environment(loader=FileSystemLoader(template_path))
    template = env.get_template("varying/labeled.jinja2")
    template.globals.update(func_dict)

    #optuna_template = env.get_template("hparams/optuna.jinja2")
    

    # iterate over the models
    for model in info_dict["models"]:
        # iterate over the datamodules
        for datamodule in info_dict["datamodules"]:
            # iterate over the number of labeled examples (100, 200, 300, 400, 500)
            for n_labeled_examples in range(100, 600, 100):
                p(template.render(
                    experiment_name=args.experiment_name,
                    model_name=model["name"],
                    datamodule_name=datamodule["name"],
                    labeled_examples=n_labeled_examples,
                    features=datamodule["features"],
                    latent=datamodule["latent"],
                    )
                )
                template_string = template.render(
                    experiment_name = args.experiment_name,
                    model_name=model["name"],
                    datamodule_name=datamodule["name"],
                    labeled_examples=n_labeled_examples,
                    features=datamodule["features"],
                    latent=datamodule["latent"],
                )

                filename = "_".join([model["name"], datamodule["name"], str(n_labeled_examples)]) + ".yaml"
                filepath = os.path.join(TO_DIR, filename)
                with open(filepath, "w") as file:
                    file.write(template_string)


            #optuna_template_string = optuna_template.render(
            #    model_name=model["name"]
            #)

            #with open("configs/hparams_search/optuna_" + model["name"] + ".yaml", "w") as file:
            #    file.write(optuna_template_string)