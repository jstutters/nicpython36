# MS_CNN

<br>
 <img height="300" src="CNN.jpeg" />
 </br>
 
# This is a modified version of nicMSlesions (https://github.com/NIC-VICOROB/nicMSlesions)

# This  version support additionally the following functionalities: 
<dl>
  <dt>(1) Runnable on a Mac system/computer</dt>
  <dt>(2) Cold start and warm start support:</dt>
  <dd>- Allowing to re-create the architecture of the model</dd>
  <dd>- Allowing to use the saved weights of the model</dd>
  <dd>- Allowing to use  the training configuration and avoiding to run preprocessing again</dd>
  <dd>- Allowing to resume training exactly where it left off(interrupting the training is     
    allowed throughout the training process)</dd>
  <dd>- Allowing to use pretrained model</dd>
  <dt>(3) Supporting Python 3</dt>
  <dt>(4) Integrated Tensorborad [to provide the measurements and visualisations of TensorFlow execution (to understand, debug, and optimisation of  the TensorFlow programs)]</dt>
  <dt>(5) Checking whether a file or directory is relevant for Training and Testing</dt> 
  <dt>(6) Easy HPC( High Performance Computing) support</dt> 
  <dt>(7) Bias correction of masks using FSL</dt>
  <dt>(8) Registration, moving all images to the Flair, T1 or Standard space</dt>
</dl>

<br>
 <img height="500" src="BR.jpg" />
 </br>


<br>
 <img height="100" src="note.jpg" />
 </br>
# Running the Program!

This modified version can be run with or without a GUI (similar to original version)

After lunching the graphical user interface, user will need to provide necessary information to start training/testing as follows:  

<br>
 <img height="500" src="GUI_NM.jpg" />
 </br>
 
 # 
# Running the Program on the HPC cluster (without any additional library/dependency installation):
First user needs to be sure that "singularity" 
https://singularity.lbl.gov/
is available on local or remote machine.

