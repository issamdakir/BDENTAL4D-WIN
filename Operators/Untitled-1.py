from os.path import expanduser, exists, join
import shutil
import os

BDENTAL_4D_Modules_DIR = join(expanduser("~/BDENTAL_4D_Modules"))

if exists(BDENTAL_4D_Modules_DIR) :
    shutil.rmtree(BDENTAL_4D_Modules_DIR)

os.mkdir(BDENTAL_4D_Modules_DIR)