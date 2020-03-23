# Homework assigment 0-A

1. Get the UNAM internet connection (RIU).
2. Install
   * Install [Visual Studio Code](https://code.visualstudio.com/download).

   * Install [Zeal](https://zealdocs.org/) ([Dash](https://kapeli.com/dash) for Mac) offline documentation.

   * Install [Anaconda Python](https://www.anaconda.com/distribution/) 3.x.

# Homework assigment 0-B

1. Sofware for this course [installation estimated time: 1-3 hrs]
   2. Create a [GitHub account](https://github.com/join).
   3. Install
      1. **Linux**
         1. `>_ chmod u+x install_ubuntu_1804.sh  && bash install_ubuntu_1804.sh`
      2. **Windows**
         1. Install [Git](https://git-scm.com/downloads).
         2. [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145).
         3. [Visual C++ Redistributable Packages for Visual Studio 2013](https://www.microsoft.com/en-us/download/details.aspx?id=40784).
         4. [Visual Studio](https://visualstudio.microsoft.com/downloads/?utm_medium=post-banner&utm_source=microsoft.com&utm_campaign=channel+banner&utm_content=launch+vs2019)
         5. **Intel** and/or **Nvidia** drivers from [nvidia.com](https://www.nvidia.com/Download/index.aspx?lang=en-us). Refer to TensorFlow's [GPU installation instructions](https://tensorflow.org/install/gpu) for more details on GPU speed up.
   5. Create an Anaconda virtual environment with the required packages ([install_python_ml_class_geo.yml](https://github.com/mathphysmx/teaching-ml/blob/master/install_python_ml_class_geo.yml) see animated .gif)
      1. Test jupyterLab (Open) `>_ jupyter lab`.
      2. Test TensorFlow installation `>_ python -c 'import tensorflow as tf;print(tf.__version__)'` (See animated .gif)

# Homework assigment 0-C

2. For $f(x) = x^2 - 6x + 5$, do the following:
   i. Compute the derivative, find the minimum and/or maximum, and the intersection with the $x$ and $y$ axis.
   ii. Plot $f(x)$

3. Find the gradient ($\nabla$) of $f(x, y) = x^2 + 9y^2$. Sketch the contour lines of $f$.

4. If 

   $x = \begin{pmatrix}
   1 \\
   2 \\
   \vdots \\
   6 \\
   \end{pmatrix}$, $\theta = \begin{pmatrix}
   \theta_1 \\
   \vdots \\
   \theta_n \\
   \end{pmatrix}$, $A = \begin{pmatrix}
   1 \times 1, 1 \times 2, \ldots, 1 \times 6 \\
   \vdots \\
   5 \times 1, 5 \times 2, \ldots, 5 \times 6 \\
   \end{pmatrix}$

   Do the following matrix operations

   i. $A^T$
   ii. $Ax$
   iii. $\theta^T x $. $\theta = (\theta_1, \ldots, \theta_n)^T$, $x = (x_1, \ldots, x_n)^T$
   iv. $x^Tx$. $x = (x_1, \ldots, x_n)^T$

5. Write in matrix form, the following system of equations.
   $3x_1 + 6x_2 = 56$
   $-5x_1 + 5.7x_2 = 20$ 

6. Plot the vector $(4,3)$ and find a unit vector orthogonal to it.

7. Find the orthogonal projection of $(4, 1)$ over $(3,4)$, and draw the vectors.
