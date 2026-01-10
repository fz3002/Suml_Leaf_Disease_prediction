from src.data_preparation.preprocessing.data_util import load_config_file
from src.process.train import *
from src.process.inference import *


def train(config_content: dict) -> None:
    project_path = config_content["path"]["project_path"]
    model_weights = config_content["path"]["model_weights"]
    run_name = config_content["train"]["run_name"]
    output_dir = os.path.join(project_path, model_weights, run_name)
    fit(model_section=config_content["model"],
        train_section=config_content["train"],
        output_dir=str(output_dir))


def validate(config_content: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config_content["model"]["num_classes"]
    model = SqueezeNet(num_classes=num_classes).to(device)
    project_path = config_content["path"]["project_path"]
    model_weights = config_content["path"]["model_weights"]
    run_name = config_content["train"]["run_name"]
    state_path: Path = Path(project_path) / model_weights / run_name / "weights" /"best.pt"
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)
    criterion = nn.CrossEntropyLoss()
    handler = DataHandler()
    val_loader = handler.get_dataloader('val')
    val_metrics = validate_one_epoch(model=model,
                                     loader=val_loader,
                                     criterion=criterion,
                                     device=device,
                                     num_classes=num_classes)
    print("VAL metrics:", val_metrics)


def interfere(config_content: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config_content["model"]["num_classes"]
    model = SqueezeNet(num_classes=num_classes).to(device)
    project_path = config_content["path"]["project_path"]
    model_weights = config_content["path"]["model_weights"]["root"]
    run_name = config_content["train"]["run_name"]
    state_path: Path = Path(project_path) / model_weights / run_name / "weights" /"best.pt"
    state_dict = torch.load(state_path, map_location=device)
    model.load_state_dict(state_dict)


if __name__ == '__main__':
    content = load_config_file()
    train(content)
