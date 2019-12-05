# ---------------------------------------------------------------------------------------
#   MS lesion segmentation pipeline
# ---------------------------------
#   - incorporates:
#         - MRI identification
#         - registration
#         - skull stripping
#         - MS lesion segmentation using the CNN Valverde et al (NI2017)
#
#  Sergi Valverde 2017
#  svalverde@eia.udg.edu
# ---------------------------------------------------------------------------------------

import os
import argparse
import sys
import platform
from timeit import time
import configparser
from  utils.load_options import load_options, print_options
from  utils.preprocess import preprocess_scan
from  utils.postprocess import invert_registration
from  shutil import copyfile
import click
import shutil
import numpy as np

os.system('cls' if platform.system() == 'Windows' else 'clear')
print("##################################################")
print("# MS WM lesion segmentation (Modified version)   #")
print("#                                                #")
print("# -------------------------------                #")
print("# (c) Sergi Valverde 2017                        #")
print("# Neuroimage Computing Group                     #")
print("# -------------------------------                #")
print("##################################################\n")

# link related libraries
CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(CURRENT_PATH, 'libs'))

# load options from input
parser = argparse.ArgumentParser()
parser.add_argument('--docker',
                    dest='docker',
                    action='store_true')
parser.set_defaults(docker=False)
args = parser.parse_args()
container = args.docker

# --------------------------------------------------
# load default options and update them with user information
# from utils.load_options import *
# --------------------------------------------------
default_config = configparser.SafeConfigParser()
default_config.read(os.path.join(CURRENT_PATH, 'config', 'default.cfg'))
user_config = configparser.RawConfigParser()
user_config.read(os.path.join(CURRENT_PATH, 'config', 'configuration.cfg'))

# read user's configuration file
options = load_options(default_config, user_config)

if options['debug']:
    print_options(options)

# tensorflow backend
device = str(options['gpu_number'])
print("DEBUG: ", device)
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ["CUDA_VISIBLE_DEVICES"] = device

# set paths taking into account the host OS
host_os = platform.system()
if host_os == 'Linux' or 'Darwin':
    options['niftyreg_path'] = CURRENT_PATH + '/libs/linux/niftyreg'
    options['robex_path'] = CURRENT_PATH + '/libs/linux/ROBEX/runROBEX.sh'
    options['test_slices'] = 256
elif host_os == 'Windows':
    options['niftyreg_path'] = os.path.normpath(
        os.path.join(CURRENT_PATH, 'libs', 'win', 'niftyreg'))
    options['robex_path'] = os.path.normpath(
        os.path.join(CURRENT_PATH, 'libs', 'win', 'ROBEX', 'runROBEX.bat'))
    options['test_slices'] = 256
else:
    "> ERROR: The OS system", host_os, "is not currently supported"


def check_oututs(current_folder, options, choice='testing'):
    """
    checking input errors, fixing  and writing it into the Input Issue Report File


    """
    erf =os.path.join(CURRENT_PATH, 'OutputIssueReportfile.txt')
    f = open(erf, "a")

    if os.path.isdir(os.path.join(options['test_folder'], current_folder)):
        if len(os.listdir(os.path.join(options['test_folder'], current_folder))) == 0:
           print(('Directory:', current_folder, 'is empty'))
           print('Warning: if the  directory is not going to be removed, the Testing could be later stopped!')
           if click.confirm('The empty directory will be removed. Do you want to continue?', default=True):
             f.write("The empty directory: %s has been removed from Testing set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(options['test_folder'], current_folder), ignore_errors=True)
             return
           return
    else:
        pass

    if choice == 'training':
        modalities = options['modalities'][:] + ['lesion']
        image_tags = options['image_tags'][:] + options['roi_tags'][:]
    else:
        modalities = options['modalities'][:]
        image_tags = options['image_tags'][:]

    if options['debug']:
        print("> DEBUG:", "number of input sequences to find:", len(modalities))


    print("> PRE:", current_folder, "identifying input modalities")

    found_modalities = 0
    if os.path.isdir(os.path.join(options['test_folder'], current_folder)):
        masks = [m for m in os.listdir(os.path.join(options['test_folder'], current_folder)) if m.find('.nii') > 0]
        pass  # do your stuff here for directory
    else:
        # shutil.rmtree(os.path.join(options['train_folder'], current_folder), ignore_errors=True)
        print(('The file:', current_folder, 'is not part of testing'))
        print('Warning: if the  file is not going to be removed, the Testing could be later stopped!')
        if click.confirm('The file will be removed. Do you want to continue?', default=True):
          f.write("The file: %s has been removed from Testing set!" % current_folder + os.linesep)
          f.close()
          os.remove(os.path.join(options['test_folder'], current_folder))
          return
        return




    for t, m in zip(image_tags, modalities):

        # check first the input modalities
        # find tag

        found_mod = [mask.find(t) if mask.find(t) >= 0
                     else np.Inf for mask in masks]

        if found_mod[np.argmin(found_mod)] is not np.Inf:
            found_modalities += 1


    # check that the minimum number of modalities are used
    if found_modalities < len(modalities):
           print("> ERROR:", current_folder, \
            "does not contain all valid input modalities")
           print('Warning: if the  folder is  not going to be removed, the Testing could be later stopped!')
           if click.confirm('The folder will be removed. Do you want to continue?', default=True):
             f.write("The folder: %s has been removed from Testing set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(options['test_folder'], current_folder), ignore_errors=True)







from CNN.base import test_cascaded_model
from CNN.build_model import cascade_model

# --------------------------------------------------
# net configuration
# take into account if the pretrained models have to be used
# all images share the same network model
# --------------------------------------------------
options['full_train'] = True
options['load_weights'] = True
options['weight_paths'] = os.path.join(CURRENT_PATH, 'nets')
options['net_verbose'] = 0

model = cascade_model(options)

# --------------------------------------------------
# process each of the scans
# - image identification
# - image registration
# - skull-stripping
# - WM segmentation
# --------------------------------------------------

if container:
    options['test_folder'] = os.path.normpath('/data' + options['test_folder'])
else:
    options['test_folder'] = os.path.normpath(options['test_folder'])

# set task to train
options['task'] = 'inference'

scan_list = os.listdir(options['test_folder'])
scan_list.sort()
# check and remove the folder which dose not contain the necessary modalities before prepossessing step
for check in scan_list:
    check_oututs(check, options)




# list scans
scan_list = os.listdir(options['test_folder'])
scan_list.sort()

for scan in scan_list:

    total_time = time.time()
    options['tmp_scan'] = scan
    # --------------------------------------------------
    # move things to a tmp folder before starting
    # --------------------------------------------------

    current_folder = os.path.join(options['test_folder'], scan)
    options['tmp_folder'] = os.path.normpath(
        os.path.join(current_folder,  'tmp'))

    # --------------------------------------------------
    # preprocess scans
    # --------------------------------------------------
    preprocess_scan(current_folder, options)

    # --------------------------------------------------
    # WM MS lesion inference
    # --------------------------------------------------
    seg_time = time.time()

    "> CNN:", scan, "running WM lesion segmentation"
    sys.stdout.flush()
    options['test_scan'] = scan

    test_x_data = {scan: {m: os.path.join(options['tmp_folder'], n)
                          for m, n in zip(options['modalities'],
                                          options['x_names'])}}

    out_seg = test_cascaded_model(model, test_x_data, options)

    print("> INFO:", scan, "CNN Segmentation time: ",\
        round(time.time() - seg_time), "sec")

    # If input images have been registered before segmentation -> T1w space,
    # then resample the segmentation  back to the original space
    if options['register_modalities']:
        print("> INFO:", scan, "Inverting lesion segmentation masks")
        invert_registration(current_folder, options)

    print("> INFO:", scan, "total pipeline time: ",\
        round(time.time() - total_time), "sec")

    # remove tmps if not set
    if options['save_tmp'] is False:
        try:
            copyfile(os.path.join(current_folder,
                                  options['experiment'],
                                  options['experiment'] +
                                  '_out_CNN.nii.gz'),
                     os.path.join(current_folder,
                                  'out_seg_' +
                                  options['experiment'] +
                                  '.nii.gz'))
            os.rmdir(options['tmp_folder'])
            os.rmdir(os.path.join(options['current_folder'],
                                  options['experiment']))
        except:
            pass

print("> INFO: All processes have been finished. Have a good day!")
