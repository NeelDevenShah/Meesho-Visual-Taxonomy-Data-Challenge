import torch
import zipfile
import os
import subprocess
from dataprep import start_process

# TODO
zip_file_path = "/content/drive/MyDrive/Meesho Hackathon/visual-taxonomy.zip"
output_dir = "/content/Meesho-Data-Challenge/data/"

def checkgpu():
    try:
        assert torch.cuda.is_available() is True
    except AssertionError:
        print("Please set up a GPU before using LLaMA Factory")

    print("Unzipping completed!")

def unzip_data():
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

# TODO: code is configured for the colab, change the settings as per kaggle and from amazon team
def get_main_code():

    # Change directory to /content/
    # os.chdir('/content/')

    # # Remove the existing LLaMA-Factory directory if it exists
    # if os.path.exists('LLaMA-Factory'):
    #     os.system('rm -rf LLaMA-Factory')

    # # Clone the LLaMA-Factory repository
    # subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/hiyouga/LLaMA-Factory.git'])

    # # Change directory to LLaMA-Factory
    # os.chdir('LLaMA-Factory')

    # # List the contents of the directory
    # print(os.listdir('.'))

    # Change path code here or at the start of process

    # Install the specified versions of torch, torchvision, and torchaudio
    subprocess.run(['pip', 'install', 'torch==2.3.1', 'torchvision==0.18.1', 'torchaudio==2.3.1'])

    # Uninstall jax if it exists
    subprocess.run(['pip', 'uninstall', '-y', 'jax'])

    # Install LLaMA-Factory with the specified optional dependencies
    subprocess.run(['pip', 'install', '-e', '.[torch,bitsandbytes,liger-kernel]'])

    print("Setup completed successfully!")

if __name__ = "__main__":
    checkgpu()
    unzip_data()
    get_main_code()
    start_process()