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

import argparse
import configparser
import os
import platform
import shutil
import sys
import tempfile
from timeit import time
from utils.load_options import load_options, print_options
from utils.preprocess import preprocess_scan
from utils.postprocess import invert_registration
from shutil import copyfile
import click
import numpy as np


def print_credits():
    print("##################################################")
    print("# MS WM lesion segmentation                      #")
    print("#                                                #")
    print("# -------------------------------                #")
    print("# Based on nicmslesions by Sergi Valverde        #")
    print("# Updates by Kevin Bronik                        #")
    print("# -------------------------------                #")
    print("##################################################\n")


def parse_arguments(app_path):
    # load options from input
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--docker',
        dest='docker',
        action='store_true',
        default=False
    )
    parser.add_argument(
        '--configuration',
        default=os.path.join(app_path, 'config', 'configuration.cfg'),
        dest='configuration_path'
    )
    parser.add_argument(
        '--weights',
        default=os.path.join(app_path, 'nets'),
        dest='weights_path'
    )
    return parser.parse_args()


def read_options(app_path, user_config_path):
    # --------------------------------------------------
    # load default options and update them with user information
    # from utils.load_options import *
    # --------------------------------------------------
    default_config = configparser.ConfigParser()
    default_config.read(os.path.join(app_path, 'config', 'default.cfg'))
    user_config = configparser.ConfigParser()
    user_config.read(user_config_path)

    # read user's configuration file
    options = load_options(default_config, user_config)

    if options['debug']:
        print_options(options)
    return options


def set_utility_paths(options, app_path):
    # set paths taking into account the host OS
    host_os = platform.system()
    if host_os == 'Linux' or 'Darwin':
        options['niftyreg_path'] = app_path + '/libs/linux/niftyreg'
        options['robex_path'] = app_path + '/libs/linux/ROBEX/runROBEX.sh'
        options['test_slices'] = 256
    elif host_os == 'Windows':
        options['niftyreg_path'] = os.path.normpath(
            os.path.join(app_path, 'libs', 'win', 'niftyreg'))
        options['robex_path'] = os.path.normpath(
            os.path.join(app_path, 'libs', 'win', 'ROBEX', 'runROBEX.bat'))
        options['test_slices'] = 256
    else:
        "> ERROR: The OS system", host_os, "is not currently supported"
    return options


def main():
    print_credits()

    # link related libraries
    CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
    sys.path.append(os.path.join(CURRENT_PATH, 'libs'))

    args = parse_arguments(CURRENT_PATH)

    container = args.docker

    options = read_options(CURRENT_PATH, args.configuration_path)

    # tensorflow backend
    device = str(options['gpu_number'])
    print("DEBUG: ", device)
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    # os.environ["CUDA_VISIBLE_DEVICES"] = device

    options = set_utility_paths(options, CURRENT_PATH)

    # --------------------------------------------------
    # net configuration
    # take into account if the pretrained models have to be used
    # all images share the same network model
    # --------------------------------------------------
    options['full_train'] = True
    options['load_weights'] = True
    options['weight_paths'] = os.path.join(CURRENT_PATH, 'nets')
    options['net_verbose'] = 0
    options['use_pretrained_model'] = False

    from CNN.base import test_cascaded_model
    from CNN.build_model import cascade_model

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

    for scan in scan_list:
        total_time = time.time()
        options['tmp_scan'] = scan
        # --------------------------------------------------
        # move things to a tmp folder before starting
        # --------------------------------------------------

        current_folder = os.path.join(options['test_folder'], scan)
        options['tmp_folder'] = tempfile.mkdtemp()

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

        test_cascaded_model(model, test_x_data, options)

        # If input images have been registered before segmentation -> T1w space,
        # then resample the segmentation  back to the original space
        if options['register_modalities']:
            print("> INFO:", scan, "Inverting lesion segmentation masks")
            invert_registration(current_folder, options)

        print("> INFO:", scan, "CNN Segmentation time: ",
              round(time.time() - seg_time), "sec")
        print("> INFO:", scan, "total pipeline time: ",
              round(time.time() - total_time), "sec")

        # remove tmps if not set
        if options['save_tmp'] is False:
            try:
                os.rmdir(options['tmp_folder'])
                os.rmdir(os.path.join(options['current_folder'],
                                      options['experiment']))
            except:
                pass

    print("> INFO: All processes have been finished. Have a good day!")


if __name__ == "__main__":
    main()
