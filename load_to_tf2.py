import os
import platform
import shutil
from pathlib import Path
import json


default_config = {
    "maps": "TF2 PATH/tf/maps",
    "scripts": "TF2 PATH/tf/scripts/vscripts",
    "cfg": "TF2 PATH/tf/cfg",
}


def load_config(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def create_config(path: Path):
    with path.open("w") as file:
        json.dump(default_config, file, indent=4)
    print("Config file created")



def copy_folder_contents(source_folder, destination_folder):
    # Copy all files and subdirectories from source to destination
    try:
        for item in source_folder.iterdir():
            destination_item = destination_folder / item.name
            if item.is_dir():
                # If it's a directory, copy its contents recursively
                shutil.copytree(item, destination_item, dirs_exist_ok=True)
            else:
                # If it's a file, copy it
                shutil.copy2(item, destination_item)
        print(f"Copied contents from {source_folder} to \n{destination_folder}")
    except Exception as e:
        print(f"Error copying contents from {source_folder} to \n {destination_folder}: {e}")

def create_and_copy_folders():
    current_directory = Path.cwd()  # Get the current working directory

    config_path = current_directory / "config.json"

    if not config_path.exists():
        create_config(config_path)

    folder_path = load_config(config_path)

    for key in folder_path.keys():
        if len(folder_path[key]) == 0:
            raise RuntimeError("Config file is empty")

    if len(folder_path.keys()) < len(default_config.keys()):
        create_config(config_path)
        raise RuntimeError("Config has been updated")


    for folder_name, destination in folder_path.items():

        source_folder = current_directory / folder_name  # Using relative path
        target_dir = current_directory / destination

        if not source_folder.exists():
            print(f"Source folder {source_folder} does not exist. Skipping.")
            continue

        # Check if the target directory exists before copying
        if not target_dir.exists():
            print(f"Target folder {target_dir} does not exist. Skipping copy.")
            continue

        try:
            # Copy the contents of the folder
            copy_folder_contents(source_folder, target_dir)

        except Exception as e:
            print(f"Failed to process {folder_name}: {e}")

if __name__ == "__main__":
    create_and_copy_folders()
