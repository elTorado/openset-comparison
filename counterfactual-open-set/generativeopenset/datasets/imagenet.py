import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image
from scipy import io as sio
import gzip
import struct
import csv


##### CONVERT IMAGENT PROTOCOL CSV FILES INTO .DATASET FILES FOR GAN TRAINING 
##### EACH ROW IN PROTOCOL IS TRANSFORMED INTO DICT WITH FILENAME / FOLD / LABEL
##### TRAN AND VAL DATA IS COMBINED INTO ONE SINGLE FILE 


DATA_DIR = '/home/deanheizmann/masterthesis/openset-imagenet/protocols'
_DATA_DIR = '/home/user/heizmann//openset-imagenet/protocols/'