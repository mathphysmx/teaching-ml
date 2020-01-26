## Homework assigment 0

1. Sofware for this course [installation estimated time: 3 hrs]
   1. Get the UNAM internet connection.
   2. Create a [GitHub account](https://github.com/join).
   3. Install
      1. Linux
         1. chmod u+x install_ubuntu_1804.sh  && bash install_ubuntu_1804.sh
      2. Windows
         1. Install [Git](https://git-scm.com/downloads).
         2. [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145).
         3. [Visual C++ Redistributable Packages for Visual Studio 2013](https://www.microsoft.com/en-us/download/details.aspx?id=40784).
         4. [Visual Studio](https://visualstudio.microsoft.com/downloads/?utm_medium=post-banner&utm_source=microsoft.com&utm_campaign=channel+banner&utm_content=launch+vs2019)
         5. **Intel** and/or **Nvidia** drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx?lang=en-us). Refer to TensorFlow's [GPU installation instructions](https://tensorflow.org/install/gpu) for more details on GPU speed up.
   4. Install [Anaconda Python](https://www.anaconda.com/distribution/).
   5. Install [VSCode](https://code.visualstudio.com/download).
   6. Install python packages inside an Anaconda virtual environment (*install_Python_ml_class_Geo.yml* see animated .gif)
      1. Test jupyterLab (Open) `$ jupyter lab`.
      2. test the (TensorFlow) installation `>_ python -c 'import tensorflow as tf;print(tf.__version__)'` (See animated .gif)

2. For $f(x) = x^2 - 6x + 5$, do the following:
   i. Compute the derivative (minimum maximum), intersection with the axis.
   ii. Plot $f(x)$

3. Find the gradient ($\nabla$) of $f(x, y) = x^2 + 9y^2$. Sketch the contour lines of $f$.

4. Do the following matrix operations
   1. $A^T$
   2. $Ax$
   3. $\theta^T x $
   4. $x^Tx$

5. Plot the vector (4,3) and find a unit vector orthogonal to it.

6. Find the orthogonal projection of $(4, 1)$ over $(3,4)$, and draw the vectors.
