# because of the way the creators saved the model, the model has to be downloaded (and dependency modules placed) in the same location as the build_gum.py file
BEST_MODEL_PATH = 'best_parser.pt'

# Maybe save in a georgetown file server?
BEST_MODEL_REMOTE_PATH = 'https://drive.google.com/u/0/uc?id=1LC5iVcvgksQhNVJ-CbMigqXnPAaquiA2&export=download'

CONFIG_CALC_HEAD_CONTRIBUTIONS = 0 # 0 is faster, 1 is slower. Also see the original repo for more info: https://github.com/KhalilMrini/LAL-Parser#Inference
