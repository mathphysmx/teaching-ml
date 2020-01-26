#!/bin/bash

ARCH=86_64
OS=amd64 # i386
UBUNTUV=bionic
OSt-get -y update 
sudo apt-get upgrade

# To be able to use 'add-apt-repository'
sudo apt -y install software-properties-common
# to download files
sudo apt -y install curl
sudo apt-get install python-software-properties

sudo apt -y install cmake libboost-all-dev libjpeg-dev libpq-dev libsdl2-dev sudo swig unzip xorg-dev xvfb zip zlib1g-dev

# C++ dependencies=amd64

sudo ap
sudo apt install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
sudo apt-get install -y libcgal-dev libglu1-mesa-dev libglu1-mesa-dev # for rgl package
sudo apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev # for rgl package

# VERSION CONTROL
# https://stackoverflow.com/questions/19109542/installing-latest-version-of-git-in-ubuntu
sudo add-apt-repository ppa:git-core/ppa
sudo apt -y update
sudo apt -y install git
# git config --global user.name "name lastName"
# git config --global user.email "mymail@gmail.com"
# git config --global --list # Check the last two steps were right

# Conda Python package manager. INSTALL WITHOUT ROOT PRIVILEGES!!!
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x$ARCH.sh
# bash Miniconda3-latest-Linux-x$ARCH.sh
# source ~/.bashrc
# printf 'export PATH=~/miniconda3/bin:$PATH' >> ~/.bashrc

## cmake. build, test and package software. https://cmake.org/download/
## dependencies
sudo apt install -y openssl libssl-dev
sudo apt update
sudo apt install snapd
# sudo snap install cmake

## DATA SCIENCE TOOLS
## https://computingforgeeks.com/how-to-install-apache-spark-on-ubuntu-debian/
## JAVA
sudo apt install default-jdk
sudo apt update
sudo add-apt-repository ppa:webupd8team/java
sudo apt update
sudo apt install oracle-java8-installer oracle-java8-set-default

## doxygen. http://www.stack.nl/~dimitri/doxygen/download.html
## dependencies
# cmake, flex, bison, graphviz, latex,
sudo apt install libc6 libc6-dev # for libiconv library
sudo apt install -y bison
sudo apt install -y flex # requires bison
# dependency graphs, the graphical inheritance graphs, and the collaboration graphs
sudo apt install -y graphviz  
# source ../bash/cmake.bash

# Node package manager (npm). https://github.com/creationix/nvm
sudo apt-get update
sudo apt-get install build-essential libssl-dev
sudo apt -y install nodejs
sudo apt -y install npm
# source ../bash/nodePackageManagerNPM.bash

## Parallel/distributed computing
## NVIDIA drivers
# conda install cudatoolkit
##Intel Processors drivers

## C++ Cpp development
sudo apt-get install -y python-pybind11
##  install MPI library. http://www.mpich.org/
sudo apt -y install mpich
## Check MPI version. https://stackoverflow.com/questions/17356765/how-do-i-check-the-version-of-mpich
sudo apt -y install libboost-all-dev

sudo apt install -y libx11-dev mesa-common-dev libglu1-mesa-dev

sudo apt -y install ca-certificates

# install package manager
sudo apt-get -y install gdebi-core
sudo apt update

sudo apt -y install ffmpegthumbnailer
sudo apt -y install atool

## Install Neo-Vim
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:neovim-ppa/unstable
sudo apt-get update
sudo apt-get -y install neovim
## Prerequisites for the Python modules:
sudo apt-get install -y python-dev python-pip python3-dev python3-pip

## VSCode IDE
## https://linuxize.com/post/how-to-install-visual-studio-code-on-ubuntu-18-04/
sudo apt update
sudo apt install software-properties-common apt-transport-https wget
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
sudo apt update
sudo apt install code

# sql database browser
sudo apt-get install -y sqlite3
## http://sqlitebrowser.org/
# sudo add-apt-repository -y ppa:linuxgndu/sqlitebrowser
# sudo apt update
sudo apt install -y sqlitebrowser

sudo apt-get install -y libncurses5-dev libncursesw5-dev

# install lates latexmk version
#https://www.devmanuals.net/install/ubuntu/ubuntu-16-04-LTS-Xenial-Xerus/how-to-install-latexmk.html
# Difficult install
# https://staging.wikimatze.de/vimtex-the-perfect-tool-for-working-with-tex-and-vim/
sudo apt -y update
sudo apt-get -y upgrade
sudo apt-get -y remove latexmk
sudo apt-get -y install latexmk

# Offline software documentation browser
# sudo add-apt-repository ppa:zeal-developers/ppa
# sudo apt update
sudo apt -y install zeal

sudo apt install -y pandoc

sudo apt-get update
## OCR (optical character recognition) for .pdf scanned files
## https://installion.co.uk/ubuntu/xenial/universe/t/tesseract-ocr/install/index.html
## https://www.linux.com/blog/using-tesseract-ubuntu
echo "deb http://us.archive.ubuntu.com/ubuntu bionic main universe" >> /etc/apt/sources.list
sudo apt update
sudo apt install -y tesseract-ocr
## http://askubuntu.com/questions/793634/how-do-i-install-a-new-language-pack-for-tesseract-on-16-04
sudo apt-get -y install tesseract-ocr-[all]

## prerequisites.  For High Performance Computing (HPC), graphics, xml,
sudo apt -y install libatlas3-base
sudo apt -y install libopenblas-base
sudo apt-get -y install libcurl4-openssl-dev
sudo apt-get -y install libcairo2-dev
sudo apt-get -y install libxt-dev
sudo apt-get -y install r-cran-xml
## INTEL PROCESSORS
## http://dirk.eddelbuettel.com/blog/2018/04/15/
# cd /tmp
# wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
# ## all products:
# #sudo wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
# ## just MKL
# sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
# ## other (TBB, DAAL, MPI, ...) listed on page
# apt-get update
# apt-get install intel-mkl-64bit-2018.2-046   ## wants 500+mb :-/  installs to 1.8 gb :-/
# ## update alternatives
# update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so     libblas.so-x86_64-linux-gnu      /opt/intel/mkl/lib/intel64/libmkl_rt.so 50
# update-alternatives --install /usr/lib/x86_64-linux-gnu/libblas.so.3   libblas.so.3-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 50
# update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so   liblapack.so-x86_64-linux-gnu    /opt/intel/mkl/lib/intel64/libmkl_rt.so 50
# update-alternatives --install /usr/lib/x86_64-linux-gnu/liblapack.so.3 liblapack.so.3-x86_64-linux-gnu  /opt/intel/mkl/lib/intel64/libmkl_rt.so 50
# echo "/opt/intel/lib/intel64"     >>  /etc/ld.so.conf.d/mkl.conf
# echo "/opt/intel/mkl/lib/intel64" >> /etc/ld.so.conf.d/mkl.conf
# ldconfig
# echo "MKL_THREADING_LAYER=GNU" >> /etc/environment

## video (.mp4, .flv,...)/audio (.mp3,..)  converter / conversion
## http://tipsonubuntu.com/2016/11/02/install-ffmpeg-3-2-via-ppa-ubuntu-16-04/
sudo add-apt-repository -y ppa:jonathonf/ffmpeg-4
sudo apt install -y ffmpeg libavcodec-extra
## remove:
#sudo apt install ppa-purge && sudo ppa-purge ppa:jonathonf/ffmpeg-3 && sudo apt autoremove

# LaTeX, latex
## Texlive , https://wiki.debian.org/Latex
sudo apt install -y texlive-full
sudo apt install -y texlive-latex-extra
sudo apt install -y texlive-fonts-extra
#apt-get install -y texlive-full --install-suggests

# http://tex.stackexchange.com/questions/150114/how-to-organize-a-bib-file-edited-by-hand
sudo apt -y install bibtool

# File comparison. winmerge, kdiff3
sudo apt-get -y install meld

## To run .Rnw (Literate programming, R noweb) files
sudo apt-get -y install wmctrl

# .RAR files decompression
sudo apt-get -y install unrar
sudo apt-get -y install   rar

# Vtk 3D visualization format
sudo apt-get install -y libvtk7-dev
sudo apt-get install python-vtk

## APACHE ARROW
## https://arrow.apache.org/install/
sudo apt update
sudo apt install -y -V apt-transport-https gnupg lsb-release wget
sudo wget -O /usr/share/keyrings/apache-arrow-keyring.gpg https://dl.bintray.com/apache/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/apache-arrow-keyring.gpg
sudo tee /etc/apt/sources.list.d/apache-arrow.list <<APT_LINE
deb [arch=amd64 signed-by=/usr/share/keyrings/apache-arrow-keyring.gpg] https://dl.bintray.com/apache/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/ $(lsb_release --codename --short) main
deb-src [signed-by=/usr/share/keyrings/apache-arrow-keyring.gpg] https://dl.bintray.com/apache/arrow/$(lsb_release --id --short | tr 'A-Z' 'a-z')/ $(lsb_release --codename --short) main
APT_LINE
sudo apt update
sudo apt install -y -V libarrow-dev # For C++
sudo apt install -y -V libarrow-glib-dev # For GLib (C)
sudo apt install -y -V libarrow-flight-dev # For Flight C++
sudo apt install -y -V libplasma-dev # For Plasma C++
sudo apt install -y -V libplasma-glib-dev # For Plasma GLib (C)
sudo apt install -y -V libgandiva-dev # For Gandiva C++
sudo apt install -y -V libgandiva-glib-dev # For Gandiva GLib (C)
sudo apt install -y -V libparquet-dev # For Apache Parquet C++
sudo apt install -y -V libparquet-glib-dev # For Apache Parquet GLib (C)

sudo apt-get update
sudo apt -y autoremove

## https://yadm.io/  dotfiles (configuration files) management 
sudo apt -y update

reboot

