# ICRA BARN Challenge 2025: LiCS-KI submission

ROS package containing LiCS-KI submission to ICRA BARN Challenge 2025.

## Installation

Follow the instruction below to run simulations in Singularity containers.

1. Follow this instruction to install Singularity: https://sylabs.io/guides/3.0/user-guide/installation.html. Singularity version >= 3.6.3 and <= 4.02 is required to successfully build the image!

2. Clone this repo

```bash
git clone https://github.com/LiCS-KI-BARN-2025/barn-challenge-2025.git
cd barn-challenge-2025
```

3. Build Singularity image (sudo access required)
```bash
sudo singularity build --notest nav_competition_image.sif Singularityfile.def
```

## Run Simulations

Navigate to the folder of this repo. Below is the example to run LiCS-KI algorithm.
```bash
./singularity_run.sh nav_competition_image.sif python3 the-barn-challenge/run.py --world_idx 0
```

## FAQ

-  **Q :** I got this error

    ```
    /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.34' not found (required by /.singularity.d/libs/libGLdispatch.so.0)
    /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found (required by /.singularity.d/libs/libGLX.so.0)
    ```
    **A :** Open singularity config and comment out `libGLdispatch.so` and `libGLX.so`
    ```bash
    sudo nano /etc/singularity/nvliblist.conf
    ---------------------------
    libcuda.so
    libEGL_installertest.so
    libEGL_nvidia.so
    libEGL.so
    #libGLdispatch.so       <-- Comment out this line
    libGLESv1_CM_nvidia.so
    libGLESv1_CM.so
    libGLESv2_nvidia.so
    libGLESv2.so
    libGL.so
    libGLX_installertest.so
    libGLX_nvidia.so
    libglx.so
    #libGLX.so              <-- And this line
    libnvcuvid.so
    libnvidia-cbl.so
    ```

## Authors

**LiCS-KI**\
Laboratory for information and Control Systems - KI branch [[Homepage](https://lics.kaist.ac.kr)]
Korea Advanced Institute of Science and Technology (KAIST)
- Joshua Julian Damanik
- Chala Adane Deresa
- Wajih Imliki
- Sujeong Park
