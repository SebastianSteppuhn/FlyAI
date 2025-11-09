# create the env from YAML
conda env create -f environment.yml

# if you also saved requirements.txt, reinstall exact pip deps:
conda activate cpacs-cfd
pip install -r requirements.txt