# ---------------------------------------------------------------------------------------------
#   MS lesion segmentation pipeline
# ---------------------------------
#   - incorporates:
#         - MRI identification
#         - registration
#         - skull stripping
#         - MS lesion segmentation training and testing using the CNN aproach
#           of Valverde et al (NI2017)
#
#  Sergi Valverde 2017
#  svalverde@eia.udg.edu
# ---------------------------------------------------------------------------------------------

import os
import sys
import platform
from timeit import time
import configparser
import numpy as np
import argparse
from utils.preprocess import preprocess_scan
from utils.load_options import load_options, print_options
import click
import shutil

os.system('cls' if platform.system() == 'Windows' else 'clear')

print("##################################################")
print("# ------------                                   #")
print("# nicMSlesions                                   #")
print("# ------------                                   #")
print("# MS WM lesion segmentation (Modified version)   #")
print("#                                                #")
print("# -------------------------------                #")
print("# (c) Sergi Valverde 2017                        #")
print("# Neuroimage Computing Group                     #")
print("# -------------------------------                #")
print("##################################################\n")


# load options from input
parser = argparse.ArgumentParser()
parser.add_argument('--docker',
                    dest='docker',
                    action='store_true')
parser.set_defaults(docker=False)
args = parser.parse_args()
container = args.docker

# link related libraries
CURRENT_PATH = CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
# sys.path.append(os.path.dirname(os.path.realpath(__file__)))




def check_inputs(current_folder, options, choice):
    """
    checking input errors, fixing  and writing it into the Input Issue Report File


    """
    erf =os.path.join(CURRENT_PATH, 'InputIssueReportfile.txt')
    f = open(erf, "a")

    if os.path.isdir(os.path.join(options['train_folder'], current_folder)):
        if len(os.listdir(os.path.join(options['train_folder'], current_folder))) == 0:
           print(('Directory:', current_folder, 'is empty'))
           print('Warning: if the  directory is not going to be removed, the Training could be later stopped!')
           if click.confirm('The empty directory will be removed. Do you want to continue?', default=True):
             f.write("The empty directory: %s has been removed from Training set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(options['train_folder'], current_folder), ignore_errors=True)
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
    if os.path.isdir(os.path.join(options['train_folder'], current_folder)):
        masks = [m for m in os.listdir(os.path.join(options['train_folder'], current_folder)) if m.find('.nii') > 0]
        pass  # do your stuff here for directory
    else:
        # shutil.rmtree(os.path.join(options['train_folder'], current_folder), ignore_errors=True)
        print(('The file:', current_folder, 'is not part of training'))
        print('Warning: if the  file is not going to be removed, the Training could be later stopped!')
        if click.confirm('The file will be removed. Do you want to continue?', default=True):
          f.write("The file: %s has been removed from Training set!" % current_folder + os.linesep)
          f.close()
          os.remove(os.path.join(options['train_folder'], current_folder))
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
           print('Warning: if the  folder is  not going to be removed, the Training could be later stopped!')
           if click.confirm('The folder will be removed. Do you want to continue?', default=True):
             f.write("The folder: %s has been removed from Training set!" % current_folder + os.linesep)
             f.close()
             shutil.rmtree(os.path.join(options['train_folder'], current_folder), ignore_errors=True)


        #return True







sys.path.append(os.path.join(CURRENT_PATH, 'libs'))

# load default options and update them with user information
default_config = configparser.SafeConfigParser()
default_config.read(os.path.join(CURRENT_PATH, 'config', 'default.cfg'))
user_config = configparser.RawConfigParser()
user_config.read(os.path.join(CURRENT_PATH, 'config', 'configuration.cfg'))

# read user's configuration file
options = load_options(default_config, user_config)
options['tmp_folder'] = CURRENT_PATH + '/tmp'
options['standard_lib'] = CURRENT_PATH + '/libs/standard'
if options['debug']:
    print_options(options)

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
    print("The OS system", host_os, "is not currently supported.")
    exit()


# set GPU mode from the configuration file. Trying to update
# the backend automatically from here in order to use either theano
# or tensorflow backends

# tensorflow backend
device = str(options['gpu_number'])
print("DEBUG: ", device)
os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ["CUDA_VISIBLE_DEVICES"] = device

from CNN.base import train_cascaded_model
from CNN.build_model import cascade_model

if container:
    options['train_folder'] = os.path.normpath(
        '/data' + options['train_folder'])
else:
    options['train_folder'] = os.path.normpath(options['train_folder'])

# set task to train
options['task'] = 'training'

scan_list = os.listdir(options['train_folder'])
scan_list.sort()
# check and remove the folder which dose not contain the necessary modalities before prepossessing step
for check in scan_list:
    check_inputs(check, options, 'training')


# list scan
scan_list = os.listdir(options['train_folder'])
scan_list.sort()

preprocess_time = time.time()

total_time = time.time()
if options['pre_processing'] is False:
    for scan in scan_list:



        # --------------------------------------------------
        # move things to a tmp folder before starting
        # --------------------------------------------------

        options['tmp_scan'] = scan
        current_folder = os.path.join(options['train_folder'], scan)
        options['tmp_folder'] = os.path.normpath(os.path.join(current_folder,
                                                              'tmp'))


        preprocess_scan(current_folder, options)



    # --------------------------------------------------
    # WM MS lesion training
    # - configure net and train
    # --------------------------------------------------
if options['pre_processing'] is False:
       default_config = configparser.ConfigParser()
       default_config.read(os.path.join(CURRENT_PATH , 'config', 'default.cfg'))
       default_config.set('completed', 'pre_processing', str(True))
       with open(os.path.join(CURRENT_PATH,
                           'config',
                           'default.cfg'), 'w') as configfile:
         default_config.write(configfile)

seg_time = time.time()
print("> CNN: Starting training session")
# select training scans

train_x_data = {f: {m: os.path.join(options['train_folder'], f, 'tmp', n)
                    for m, n in zip(options['modalities'], options['x_names'])}
                for f in scan_list}
train_y_data = {f: os.path.join(options['train_folder'], f, 'tmp',
                                'lesion.nii.gz')
                for f in scan_list}


options['weight_paths'] = os.path.join(CURRENT_PATH, 'nets')
options['load_weights'] = False

# train the model for the current scan

print("> CNN: training net with %d subjects" % (len(list(train_x_data.keys()))))

# --------------------------------------------------
# initialize the CNN and train the classifier
# --------------------------------------------------
model = cascade_model(options)
model = train_cascaded_model(model, train_x_data, train_y_data,  options, CURRENT_PATH)

print("> INFO: training time:", round(time.time() - seg_time), "sec")
print("> INFO: total pipeline time: ", round(time.time() - total_time), "sec")
print("> INFO: All processes have been finished. Have a good day!")
