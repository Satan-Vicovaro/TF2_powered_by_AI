import os
import platform
import shutil
from pathlib import Path

def get_os():
    return platform.system()

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
    current_os = get_os()
    current_directory = Path(os.getcwd())  # Get the current working directory
    
    
    # Define the folders to copy and their relative destination paths
    folder_path_linux = {
        "maps": "../source-sdk-2013/game/mod_tf/maps",   
        "scripts": "../source-sdk-2013/game/mod_tf/scripts/vscripts", 
    }
    folder_path_windows = { #fill that in if you are windows user 🤮 
        "maps": "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Team Fortress 2\\tf\\maps",   
        "scripts": "C:\\Program Files (x86)\\Steam\\steamapps\\common\\Team Fortress 2\\tf\\scripts\\vscripts",   
    }

    folder_path = {}

    if current_os == "Windows":
        folder_path = folder_path_windows
    elif current_os == "Linux":
        folder_path = folder_path_linux
    else:
        print(f"Unsupported OS: {current_os}")
        return


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
