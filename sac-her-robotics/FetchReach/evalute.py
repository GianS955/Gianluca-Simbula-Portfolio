from train import train
import os
from datetime import datetime

formatted_daytime = datetime.now().strftime("%d%m%y_%H%M")
result_folder = f'./results/{formatted_daytime}'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

train(result_folder)