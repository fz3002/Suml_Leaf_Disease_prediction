from server.create_server import start_mlflow_server
from utils.load_config import load_config_file


def main():
    config_content = load_config_file()
    start_mlflow_server(config_content.mlflow)


if __name__ == "__main__":
    main()