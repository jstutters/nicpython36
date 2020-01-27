
# ------------------------------------------------------------------------------------------------------------
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
# ------------------------------------------------------------------------------------------------------------
import click
import shutil
import argparse
import os
import sys
import platform
from timeit import time
import configparser
import numpy as np
from utils.preprocess import preprocess_scan
from  utils.postprocess import invert_registration
from utils.load_options import load_options, print_options
CURRENT_PATH = CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(CURRENT_PATH, 'libs'))
# load options from input
parser = argparse.ArgumentParser()
parser.add_argument('--docker',
                    dest='docker',
                    action='store_true')
parser.set_defaults(docker=False)
args = parser.parse_args()
container = args.docker

# check and remove the folder which dose not contain the necessary modalities before prepossessing step
CEND      = '\33[0m'
CBOLD     = '\33[1m'
CITALIC   = '\33[3m'
CURL      = '\33[4m'
CBLINK    = '\33[5m'
CBLINK2   = '\33[6m'
CSELECTED = '\33[7m'

CBLACK  = '\33[30m'
CRED    = '\33[31m'
CGREEN  = '\33[32m'
CYELLOW = '\33[33m'
CBLUE   = '\33[34m'
CVIOLET = '\33[35m'
CBEIGE  = '\33[36m'
CWHITE  = '\33[37m'

CBLACKBG  = '\33[40m'
CREDBG    = '\33[41m'
CGREENBG  = '\33[42m'
CYELLOWBG = '\33[43m'
CBLUEBG   = '\33[44m'
CVIOLETBG = '\33[45m'
CBEIGEBG  = '\33[46m'
CWHITEBG  = '\33[47m'

CGREY    = '\33[90m'
CRED2    = '\33[91m'
CGREEN2  = '\33[92m'
CYELLOW2 = '\33[93m'
CBLUE2   = '\33[94m'
CVIOLET2 = '\33[95m'
CBEIGE2  = '\33[96m'
CWHITE2  = '\33[97m'

CGREYBG    = '\33[100m'
CREDBG2    = '\33[101m'
CGREENBG2  = '\33[102m'
CYELLOWBG2 = '\33[103m'
CBLUEBG2   = '\33[104m'
CVIOLETBG2 = '\33[105m'
CBEIGEBG2  = '\33[106m'
CWHITEBG2  = '\33[107m'

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


def get_config():
    """
    Get the CNN configuration from file
    """
    default_config = configparser.SafeConfigParser()
    default_config.read(os.path.join(CURRENT_PATH, 'config', 'default.cfg'))
    user_config = configparser.RawConfigParser()
    user_config.read(os.path.join(CURRENT_PATH, 'config', 'configuration.cfg'))

    # read user's configuration file
    options = load_options(default_config, user_config)
    options['tmp_folder'] = CURRENT_PATH + '/tmp'
    options['standard_lib'] = CURRENT_PATH + '/libs/standard'
    # set paths taking into account the host OS
    host_os = platform.system()
    if host_os == 'Linux' or 'Darwin':
        options['niftyreg_path'] = CURRENT_PATH + '/libs/linux/niftyreg'
        options['robex_path'] = CURRENT_PATH + '/libs/linux/ROBEX/runROBEX.sh'
        # options['tensorboard_path'] = CURRENT_PATH + '/libs/bin/tensorboard'
        options['test_slices'] = 256
    elif host_os == 'Windows':
        options['niftyreg_path'] = os.path.normpath(
            os.path.join(CURRENT_PATH,
                         'libs',
                         'win',
                         'niftyreg'))

        options['robex_path'] = os.path.normpath(
            os.path.join(CURRENT_PATH,
                         'libs',
                         'win',
                         'ROBEX',
                         'runROBEX.bat'))
        options['test_slices'] = 256
    else:
        print("The OS system also here ...", host_os, "is not currently supported.")
        exit()

    # print options when debugging
    if options['debug']:
        print_options(options)

    return options

def define_backend(options):
    """
    Define the library backend and write options
    """
    #
    #    if options['backend'] == 'theano':
    #        device = 'cuda' + str(options['gpu_number']) if options['gpu_mode'] else 'cpu'
    #        os.environ['KERAS_BACKEND'] = options['backend']
    #        os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=' + device + ',floatX=float32,optimizer=fast_compile'
    #    else:
    #        device = str(options['gpu_number']) if options['gpu_mode'] is not None else " "
    #        print "DEBUG: ", device
    #        os.environ['KERAS_BACKEND'] = 'tensorflow'
    #        os.environ["CUDA_VISIBLE_DEVICES"] = device


    # forcing tensorflow
    device = str(options['gpu_number'])
    print("DEBUG: ", device)
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ["CUDA_VISIBLE_DEVICES"] = device




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





def pre_segmentation(options):
    """
    Infer segmentation given the input options passed as parameters
    """

    # define the training backend
    define_backend(options)

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

    # scan_list = os.listdir(options['test_folder'])
    # scan_list.sort()
    # # check and remove the folder which dose not contain the necessary modalities before prepossessing step
    # for check in scan_list:
    #    check_oututs(check, options)

    # update scan list after removing  the unnecessary folders before prepossessing step


    options['task'] = 'testing'
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


if __name__ == '__main__':

# 

   try:
       print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
       print('\x1b[6;30;42m' + 'preprocessing of testing data started.......................' + '\x1b[0m')
       print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
       options = get_config()
       pre_segmentation(options)
       print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
       print('\x1b[6;30;42m' + 'preprocessing of testing data completed.....................' + '\x1b[0m')
       print('\x1b[6;30;42m' + '.............................................................' + '\x1b[0m')
   except KeyboardInterrupt:
       print("KeyboardInterrupt has been caught.")
       time.sleep(1)
       os.kill(os.getpid(), signal.SIGTERM)    
    
   
    
    
    
    
