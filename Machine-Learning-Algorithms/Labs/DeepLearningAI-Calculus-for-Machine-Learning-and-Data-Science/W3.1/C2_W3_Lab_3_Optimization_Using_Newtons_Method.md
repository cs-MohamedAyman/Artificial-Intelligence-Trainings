# Optimization Using Newton's Method

In this lab you will implement Newton's method optimizing some functions in one and two variables. You will also compare it with the gradient descent, experiencing advantages and disadvantages of each of the methods.

# Table of Contents

- [ 1 - Function in One Variable](#1)
- [ 2 - Function in Two Variables](#2)

## Packages

Run the following cell to load the packages you'll need.


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
```

<a name='1'></a>
## 1 - Function in One Variable

You will use Newton's method to optimize a function $f\left(x\right)$. Aiming to find a point, where the derivative equals to zero, you need to start from some initial point $x_0$, calculate first and second derivatives ($f'(x_0)$ and $f''(x_0)$) and step to the next point using the expression:

$$x_1 = x_0 - \frac{f'(x_0)}{f''(x_0)},\tag{1}$$

Repeat the process iteratively. Number of iterations $n$ is usually also a parameter.

Let's optimize function $f\left(x\right)=e^x - \log(x)$ (defined for $x>0$) using Newton's method. To implement it in the code, define function $f\left(x\right)=e^x - \log(x)$, its first and second derivatives $f'(x)=e^x - \frac{1}{x}$, $f''(x)=e^x + \frac{1}{x^2}$:


```python
def f_example_1(x):
    return np.exp(x) - np.log(x)

def dfdx_example_1(x):
    return np.exp(x) - 1/x

def d2fdx2_example_1(x):
    return np.exp(x) + 1/(x**2)

x_0 = 1.6
print(f"f({x_0}) = {f_example_1(x_0)}")
print(f"f'({x_0}) = {dfdx_example_1(x_0)}")
print(f"f''({x_0}) = {d2fdx2_example_1(x_0)}")
```

    f(1.6) = 4.483028795149379
    f'(1.6) = 4.328032424395115
    f''(1.6) = 5.343657424395115


Plot the function to visualize the global minimum:


```python
def plot_f(x_range, y_range, f, ox_position):
    x = np.linspace(*x_range, 100)
    fig, ax = plt.subplots(1,1,figsize=(8,4))

    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    ax.set_ylim(*y_range)
    ax.set_xlim(*x_range)
    ax.set_ylabel('$f\,(x)$')
    ax.set_xlabel('$x$')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position(('data', ox_position))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.autoscale(enable=False)

    pf = ax.plot(x, f(x), 'k')

    return fig, ax

plot_f([0.001, 2.5], [-0.3, 13], f_example_1, 0.0)
```




    (<Figure size 576x288 with 1 Axes>,
     <AxesSubplot: xlabel='$x$', ylabel='$f\\,(x)$'>)





![png](output_9_1.png)



Implement Newton's method described above:


```python
def newtons_method(dfdx, d2fdx2, x, num_iterations=100):
    for iteration in range(num_iterations):
        x = x - dfdx(x) / d2fdx2(x)
        print(x)
    return x
```

In addition to the first and second derivatives, there are two other parameters in this implementation: number of iterations `num_iterations`, initial point `x`. To optimize the function, set up the parameters and call the defined function gradient_descent:


```python
num_iterations_example_1 = 25; x_initial = 1.6
newtons_example_1 = newtons_method(dfdx_example_1, d2fdx2_example_1, x_initial, num_iterations_example_1)
print("Newton's method result: x_min =", newtons_example_1)
```

    0.7900617721793732
    0.5436324685389214
    0.5665913613835818
    0.567143002403454
    0.5671432904097056
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    0.5671432904097838
    Newton's method result: x_min = 0.5671432904097838


You can see that starting from the initial point $x_0 = 1.6$ Newton's method converges after $6$ iterations. You could actually exit the loop when there is no significant change of $x$ each step (or when first derivative is close to zero).

What if gradient descent was used starting from the same intial point?


```python
def gradient_descent(dfdx, x, learning_rate=0.1, num_iterations=100):
    for iteration in range(num_iterations):
        x = x - learning_rate * dfdx(x)
        print(x)
    return x

num_iterations = 25; learning_rate = 0.1; x_initial = 1.6
# num_iterations = 25; learning_rate = 0.2; x_initial = 1.6
gd_example_1 = gradient_descent(dfdx_example_1, x_initial, learning_rate, num_iterations)
print("Gradient descent result: x_min =", gd_example_1)
```

    1.1671967575604887
    0.9315747895403638
    0.7850695373565493
    0.693190848956033
    0.6374425307430822
    0.6051557294974615
    0.5872487093998153
    0.5776311173426577
    0.5725707323584608
    0.5699397792550739
    0.5685808560397663
    0.5678813962508925
    0.5675220281938029
    0.567337566350933
    0.5672429290172856
    0.567194387884144
    0.5671694934881042
    0.5671567271988156
    0.5671501806396171
    0.5671468236191124
    0.5671451021825211
    0.5671442194561769
    0.5671437668086716
    0.5671435346987772
    0.5671434156768685
    Gradient descent result: x_min = 0.5671434156768685


Gradient descent method has an extra parameter `learning_rate`. If you take it equal to `0.1` in this example, the method will not converge even with $25$ iterations. If you increase it to $0.2$, gradient descent will converge within about $12$ iterations, which is still slower than Newton's method.

So, those are disadvantages of gradient descent method in comparison with Newton's method: there is an extra parameter to control and it converges slower. However it has an advantage - in each step you do not need to calculate second derivative, which in more complicated cases is quite computationally expensive to find. So, one step of gradient descent method is easier to make than one step of Newton's method.

This is the reality of numerical optimization - convergency and actual result depend on the initial parameters. Also, there is no "perfect" algorithm - every method will have advantages and disadvantages.

<a name='2'></a>
## 2 - Function in Two Variables

In case of a function in two variables, Newton's method will require even more computations. Starting from the intial point $(x_0, y_0)$, the step to the next point shoud be done using the expression:

$$\begin{bmatrix}x_1 \\ y_1\end{bmatrix} = \begin{bmatrix}x_0 \\ y_0\end{bmatrix} -
H^{-1}\left(x_0, y_0\right)\nabla f\left(x_0, y_0\right),\tag{2}$$

where $H^{-1}\left(x_0, y_0\right)$ is an inverse of a Hessian matrix at point $(x_0, y_0)$ and $\nabla f\left(x_0, y_0\right)$ is the gradient at that point.

Let's implement that in the code. Define the function $f(x, y)$ like in the videos, its gradient and Hessian:

\begin{align}
f\left(x, y\right) &= x^4 + 0.8 y^4 + 4x^2 + 2y^2 - xy - 0.2x^2y,\\
\nabla f\left(x, y\right) &= \begin{bmatrix}4x^3 + 8x - y - 0.4xy \\ 3.2y^3 + 4y - x - 0.2x^2\end{bmatrix}, \\
H\left(x, y\right) &= \begin{bmatrix}12x^2 + 8 - 0.4y && -1 - 0.4x \\ -1 - 0.4x && 9.6y^2 + 4\end{bmatrix}.
\end{align}


```python
def f_example_2(x, y):
    return x**4 + 0.8*y**4 + 4*x**2 + 2*y**2 - x*y -0.2*x**2*y

def grad_f_example_2(x, y):
    return np.array([[4*x**3 + 8*x - y - 0.4*x*y],
                     [3.2*y**3 +4*y - x - 0.2*x**2]])

def hessian_f_example_2(x, y):
    hessian_f = np.array([[12*x**2 + 8 - 0.4*y, -1 - 0.4*x],
                         [-1 - 0.4*x, 9.6*y**2 + 4]])
    return hessian_f

x_0, y_0 = 4, 4
print(f"f{x_0, y_0} = {f_example_2(x_0, y_0)}")
print(f"grad f{x_0, y_0} = \n{grad_f_example_2(x_0, y_0)}")
print(f"H{x_0, y_0} = \n{hessian_f_example_2(x_0, y_0)}")
```

    f(4, 4) = 528.0
    grad f(4, 4) =
    [[277.6]
     [213.6]]
    H(4, 4) =
    [[198.4  -2.6]
     [ -2.6 157.6]]


Run the following cell to plot the function:


```python
def plot_f_cont_and_surf(f):

    fig = plt.figure( figsize=(10,5))
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False
    fig.set_facecolor('#ffffff')
    gs = GridSpec(1, 2, figure=fig)
    axc = fig.add_subplot(gs[0, 0])
    axs = fig.add_subplot(gs[0, 1],  projection='3d')

    x_range = [-4, 5]
    y_range = [-4, 5]
    z_range = [0, 1200]
    x = np.linspace(*x_range, 100)
    y = np.linspace(*y_range, 100)
    X,Y = np.meshgrid(x,y)

    cont = axc.contour(X, Y, f(X, Y), cmap='terrain', levels=18, linewidths=2, alpha=0.7)
    axc.set_xlabel('$x$')
    axc.set_ylabel('$y$')
    axc.set_xlim(*x_range)
    axc.set_ylim(*y_range)
    axc.set_aspect("equal")
    axc.autoscale(enable=False)

    surf = axs.plot_surface(X,Y, f(X,Y), cmap='terrain',
                    antialiased=True,cstride=1,rstride=1, alpha=0.69)
    axs.set_xlabel('$x$')
    axs.set_ylabel('$y$')
    axs.set_zlabel('$f$')
    axs.set_xlim(*x_range)
    axs.set_ylim(*y_range)
    axs.set_zlim(*z_range)
    axs.view_init(elev=20, azim=-100)
    axs.autoscale(enable=False)

    return fig, axc, axs

plot_f_cont_and_surf(f_example_2)
```




    (<Figure size 720x360 with 2 Axes>,
     <AxesSubplot: xlabel='$x$', ylabel='$y$'>,
     <Axes3DSubplot: xlabel='$x$', ylabel='$y$', zlabel='$f$'>)





![png](output_21_1.png)



Newton's method $(2)$ is implemented in the following function:


```python
def newtons_method_2(f, grad_f, hessian_f, x_y, num_iterations=100):
    for iteration in range(num_iterations):
        x_y = x_y - np.matmul(np.linalg.inv(hessian_f(x_y[0,0], x_y[1,0])), grad_f(x_y[0,0], x_y[1,0]))
        print(x_y.T)
    return x_y
```

Now run the following code to find the minimum:


```python
num_iterations_example_2 = 25; x_y_initial = np.array([[4], [4]])
newtons_example_2 = newtons_method_2(f_example_2, grad_f_example_2, hessian_f_example_2,
                                     x_y_initial, num_iterations=num_iterations_example_2)
print("Newton's method result: x_min, y_min =", newtons_example_2.T)
```

    [[2.58273866 2.62128884]]
    [[1.59225691 1.67481611]]
    [[0.87058917 1.00182107]]
    [[0.33519431 0.49397623]]
    [[0.04123585 0.12545903]]
    [[0.00019466 0.00301029]]
    [[-2.48536390e-08  3.55365461e-08]]
    [[ 4.15999751e-17 -2.04850948e-17]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    [[0. 0.]]
    Newton's method result: x_min, y_min = [[0. 0.]]


In this example starting from the initial point $(4, 4)$ it will converge after about $9$ iterations. Try to compare it with the gradient descent now:


```python
def gradient_descent_2(grad_f, x_y, learning_rate=0.1, num_iterations=100):
    for iteration in range(num_iterations):
        x_y = x_y - learning_rate * grad_f(x_y[0,0], x_y[1,0])
        print(x_y.T)
    return x_y

num_iterations_2 = 300; learning_rate_2 = 0.02; x_y_initial = np.array([[4], [4]])
# num_iterations_2 = 300; learning_rate_2 = 0.03; x_y_initial = np.array([[4], [4]])
gd_example_2 = gradient_descent_2(grad_f_example_2, x_y_initial, learning_rate_2, num_iterations_2)
print("Gradient descent result: x_min, y_min =", gd_example_2)
```

    [[-1.552 -0.272]]
    [[-1.00667816 -0.27035727]]
    [[-0.76722601 -0.26354393]]
    [[-0.61199381 -0.2542789 ]]
    [[-0.49957833 -0.24362609]]
    [[-0.41356991 -0.23220381]]
    [[-0.34561558 -0.22041345]]
    [[-0.29081322 -0.20852957]]
    [[-0.24600097 -0.19674484]]
    [[-0.20899755 -0.1851958 ]]
    [[-0.17822189 -0.17397885]]
    [[-0.15248504 -0.1631609 ]]
    [[-0.13086798 -0.15278673]]
    [[-0.11264557 -0.14288438]]
    [[-0.09723686 -0.13346909]]
    [[-0.08417097 -0.12454632]]
    [[-0.07306297 -0.11611405]]
    [[-0.0635961  -0.10816464]]
    [[-0.05550841 -0.10068622]]
    [[-0.04858239 -0.09366384]]
    [[-0.04263691 -0.08708035]]
    [[-0.03752071 -0.08091713]]
    [[-0.03310722 -0.07515463]]
    [[-0.02929035 -0.06977285]]
    [[-0.02598099 -0.06475166]]
    [[-0.02310421 -0.06007107]]
    [[-0.02059686 -0.05571146]]
    [[-0.01840572 -0.05165372]]
    [[-0.01648577 -0.04787936]]
    [[-0.01479896 -0.04437062]]
    [[-0.01331303 -0.04111048]]
    [[-0.01200059 -0.03808275]]
    [[-0.01083835 -0.03527203]]
    [[-0.0098065  -0.03266375]]
    [[-0.00888809 -0.03024417]]
    [[-0.00806868 -0.02800031]]
    [[-0.00733584 -0.02591999]]
    [[-0.00667896 -0.02399178]]
    [[-0.00608885 -0.02220496]]
    [[-0.00555764 -0.02054949]]
    [[-0.00507848 -0.019016  ]]
    [[-0.00464546 -0.01759575]]
    [[-0.00425344 -0.01628056]]
    [[-0.00389794 -0.01506284]]
    [[-0.00357505 -0.01393549]]
    [[-0.00328135 -0.01289193]]
    [[-0.00301383 -0.01192602]]
    [[-0.00276985 -0.01103207]]
    [[-0.00254707 -0.01020478]]
    [[-0.00234342 -0.00943925]]
    [[-0.00215708 -0.0087309 ]]
    [[-0.00198642 -0.00807551]]
    [[-0.00182997 -0.00746915]]
    [[-0.00168645 -0.00690818]]
    [[-0.00155469 -0.00638922]]
    [[-0.00143364 -0.00590915]]
    [[-0.00132237 -0.00546507]]
    [[-0.00122004 -0.00505429]]
    [[-0.00112587 -0.00467434]]
    [[-0.00103917 -0.00432289]]
    [[-0.00095933 -0.00399784]]
    [[-0.00088576 -0.00369719]]
    [[-0.00081796 -0.00341912]]
    [[-0.00075544 -0.00316195]]
    [[-0.00069779 -0.0029241 ]]
    [[-0.00064461 -0.00270412]]
    [[-0.00059554 -0.00250068]]
    [[-0.00055026 -0.00231253]]
    [[-0.00050846 -0.00213853]]
    [[-0.00046987 -0.00197762]]
    [[-0.00043423 -0.00182881]]
    [[-0.00040132 -0.00169118]]
    [[-0.00037093 -0.00156392]]
    [[-0.00034286 -0.00144622]]
    [[-0.00031692 -0.00133738]]
    [[-0.00029296 -0.00123673]]
    [[-0.00027081 -0.00114365]]
    [[-0.00025035 -0.00105757]]
    [[-0.00023145 -0.00097797]]
    [[-0.00021397 -0.00090436]]
    [[-0.00019782 -0.00083629]]
    [[-0.0001829  -0.00077335]]
    [[-0.0001691  -0.00071514]]
    [[-0.00015634 -0.00066131]]
    [[-0.00014455 -0.00061153]]
    [[-0.00013366 -0.0005655 ]]
    [[-0.00012358 -0.00052293]]
    [[-0.00011427 -0.00048357]]
    [[-0.00010565 -0.00044717]]
    [[-9.76923243e-05 -4.13507533e-04]]
    [[-9.03313798e-05 -3.82380734e-04]]
    [[-8.35256974e-05 -3.53596867e-04]]
    [[-7.72332868e-05 -3.26979601e-04]]
    [[-7.14153509e-05 -3.02365872e-04]]
    [[-6.60360394e-05 -2.79604888e-04]]
    [[-6.10622231e-05 -2.58557198e-04]]
    [[-5.64632851e-05 -2.39093851e-04]]
    [[-5.22109285e-05 -2.21095595e-04]]
    [[-4.82789994e-05 -2.04452154e-04]]
    [[-4.46433236e-05 -1.89061552e-04]]
    [[-4.12815554e-05 -1.74829486e-04]]
    [[-3.81730385e-05 -1.61668751e-04]]
    [[-3.52986780e-05 -1.49498706e-04]]
    [[-3.26408214e-05 -1.38244778e-04]]
    [[-3.01831494e-05 -1.27838007e-04]]
    [[-2.79105748e-05 -1.18214626e-04]]
    [[-2.58091489e-05 -1.09315664e-04]]
    [[-2.38659758e-05 -1.01086591e-04]]
    [[-2.20691322e-05 -9.34769812e-05]]
    [[-2.04075942e-05 -8.64402033e-05]]
    [[-1.88711691e-05 -7.99331372e-05]]
    [[-1.74504327e-05 -7.39159082e-05]]
    [[-1.61366713e-05 -6.83516429e-05]]
    [[-1.49218279e-05 -6.32062439e-05]]
    [[-1.37984528e-05 -5.84481800e-05]]
    [[-1.27596575e-05 -5.40482939e-05]]
    [[-1.17990727e-05 -4.99796229e-05]]
    [[-1.09108088e-05 -4.62172339e-05]]
    [[-1.00894200e-05 -4.27380709e-05]]
    [[-9.32987078e-06 -3.95208132e-05]]
    [[-8.62750477e-06 -3.65457452e-05]]
    [[-7.97801639e-06 -3.37946354e-05]]
    [[-7.37742432e-06 -3.12506246e-05]]
    [[-6.82204707e-06 -2.88981229e-05]]
    [[-6.30848042e-06 -2.67227139e-05]]
    [[-5.83357648e-06 -2.47110662e-05]]
    [[-5.39442442e-06 -2.28508523e-05]]
    [[-4.98833257e-06 -2.11306725e-05]]
    [[-4.61281197e-06 -1.95399852e-05]]
    [[-4.26556103e-06 -1.80690426e-05]]
    [[-3.94445150e-06 -1.67088303e-05]]
    [[-3.64751534e-06 -1.54510129e-05]]
    [[-3.37293269e-06 -1.42878821e-05]]
    [[-3.11902072e-06 -1.32123101e-05]]
    [[-2.88422328e-06 -1.22177057e-05]]
    [[-2.66710138e-06 -1.12979737e-05]]
    [[-2.46632439e-06 -1.04474778e-05]]
    [[-2.28066184e-06 -9.66100601e-06]]
    [[-2.10897589e-06 -8.93373875e-06]]
    [[-1.95021437e-06 -8.26121915e-06]]
    [[-1.80340433e-06 -7.63932589e-06]]
    [[-1.66764604e-06 -7.06424789e-06]]
    [[-1.54210754e-06 -6.53246097e-06]]
    [[-1.42601947e-06 -6.04070623e-06]]
    [[-1.31867041e-06 -5.58597011e-06]]
    [[-1.21940249e-06 -5.16546591e-06]]
    [[-1.12760736e-06 -4.77661668e-06]]
    [[-1.04272247e-06 -4.41703949e-06]]
    [[-9.64227629e-07 -4.08453077e-06]]
    [[-8.91641792e-07 -3.77705286e-06]]
    [[-8.24520136e-07 -3.49272146e-06]]
    [[-7.62451320e-07 -3.22979415e-06]]
    [[-7.05054972e-07 -2.98665964e-06]]
    [[-6.51979353e-07 -2.76182796e-06]]
    [[-6.02899201e-07 -2.55392131e-06]]
    [[-5.57513743e-07 -2.36166559e-06]]
    [[-5.15544845e-07 -2.18388262e-06]]
    [[-4.76735313e-07 -2.01948290e-06]]
    [[-4.40847314e-07 -1.86745898e-06]]
    [[-4.07660916e-07 -1.72687920e-06]]
    [[-3.76972748e-07 -1.59688208e-06]]
    [[-3.48594745e-07 -1.47667097e-06]]
    [[-3.22353001e-07 -1.36550919e-06]]
    [[-2.98086701e-07 -1.26271551e-06]]
    [[-2.75647136e-07 -1.16766001e-06]]
    [[-2.54896792e-07 -1.07976015e-06]]
    [[-2.35708506e-07 -9.98477272e-07]]
    [[-2.17964689e-07 -9.23313260e-07]]
    [[-2.01556602e-07 -8.53807493e-07]]
    [[-1.86383694e-07 -7.89534025e-07]]
    [[-1.72352983e-07 -7.30098977e-07]]
    [[-1.59378484e-07 -6.75138118e-07]]
    [[-1.47380688e-07 -6.24314638e-07]]
    [[-1.36286070e-07 -5.77317081e-07]]
    [[-1.26026640e-07 -5.33857436e-07]]
    [[-1.16539526e-07 -4.93669374e-07]]
    [[-1.07766588e-07 -4.56506614e-07]]
    [[-9.96540662e-08 -4.22141417e-07]]
    [[-9.21522436e-08 -3.90363185e-07]]
    [[-8.52151480e-08 -3.60977175e-07]]
    [[-7.88002676e-08 -3.33803304e-07]]
    [[-7.28682907e-08 -3.08675045e-07]]
    [[-6.73828649e-08 -2.85438407e-07]]
    [[-6.23103745e-08 -2.63950992e-07]]
    [[-5.76197343e-08 -2.44081120e-07]]
    [[-5.32821991e-08 -2.25707025e-07]]
    [[-4.92711876e-08 -2.08716107e-07]]
    [[-4.55621197e-08 -1.93004242e-07]]
    [[-4.21322653e-08 -1.78475145e-07]]
    [[-3.89606057e-08 -1.65039779e-07]]
    [[-3.60277043e-08 -1.52615809e-07]]
    [[-3.33155877e-08 -1.41127098e-07]]
    [[-3.08076356e-08 -1.30503242e-07]]
    [[-2.84884787e-08 -1.20679135e-07]]
    [[-2.63439048e-08 -1.11594574e-07]]
    [[-2.43607715e-08 -1.03193886e-07]]
    [[-2.25269258e-08 -9.54255907e-08]]
    [[-2.08311294e-08 -8.82420820e-08]]
    [[-1.92629904e-08 -8.15993380e-08]]
    [[-1.78128986e-08 -7.54566508e-08]]
    [[-1.64719679e-08 -6.97763767e-08]]
    [[-1.52319805e-08 -6.45237059e-08]]
    [[-1.40853378e-08 -5.96664491e-08]]
    [[-1.30250127e-08 -5.51748399e-08]]
    [[-1.20445075e-08 -5.10213529e-08]]
    [[-1.11378133e-08 -4.71805349e-08]]
    [[-1.02993739e-08 -4.36288483e-08]]
    [[-9.52405102e-09 -4.03445279e-08]]
    [[-8.80709341e-09 -3.73074467e-08]]
    [[-8.14410740e-09 -3.44989929e-08]]
    [[-7.53103007e-09 -3.19019556e-08]]
    [[-6.96410437e-09 -2.95004197e-08]]
    [[-6.43985606e-09 -2.72796682e-08]]
    [[-5.95507246e-09 -2.52260919e-08]]
    [[-5.5067827e-09 -2.3327106e-08]]
    [[-5.09223959e-09 -2.15710732e-08]]
    [[-4.70890272e-09 -1.99472321e-08]]
    [[-4.35442292e-09 -1.84456316e-08]]
    [[-4.02662789e-09 -1.70570695e-08]]
    [[-3.72350881e-09 -1.57730365e-08]]
    [[-3.44320813e-09 -1.45856638e-08]]
    [[-3.18400811e-09 -1.34876748e-08]]
    [[-2.94432031e-09 -1.24723410e-08]]
    [[-2.72267588e-09 -1.15334401e-08]]
    [[-2.51771654e-09 -1.06652184e-08]]
    [[-2.32818626e-09 -9.86235530e-09]]
    [[-2.15292357e-09 -9.11993060e-09]]
    [[-1.99085441e-09 -8.43339462e-09]]
    [[-1.84098559e-09 -7.79854014e-09]]
    [[-1.70239870e-09 -7.21147664e-09]]
    [[-1.57424444e-09 -6.66860649e-09]]
    [[-1.45573746e-09 -6.16660286e-09]]
    [[-1.34615152e-09 -5.70238938e-09]]
    [[-1.24481507e-09 -5.27312126e-09]]
    [[-1.15110708e-09 -4.87616786e-09]]
    [[-1.06445331e-09 -4.50909657e-09]]
    [[-9.84322709e-10 -4.16965791e-09]]
    [[-9.10224233e-10 -3.85577173e-09]]
    [[-8.41703791e-10 -3.56551448e-09]]
    [[-7.78341474e-10 -3.29710740e-09]]
    [[-7.19748986e-10 -3.04890563e-09]]
    [[-6.65567261e-10 -2.81938816e-09]]
    [[-6.15464262e-10 -2.60714845e-09]]
    [[-5.69132949e-10 -2.41088586e-09]]
    [[-5.26289395e-10 -2.22939765e-09]]
    [[-4.86671045e-10 -2.06157163e-09]]
    [[-4.50035110e-10 -1.90637932e-09]]
    [[-4.16157079e-10 -1.76286968e-09]]
    [[-3.84829340e-10 -1.63016324e-09]]
    [[-3.55859910e-10 -1.50744677e-09]]
    [[-3.29071260e-10 -1.39396823e-09]]
    [[-3.04299223e-10 -1.28903219e-09]]
    [[-2.81391991e-10 -1.19199560e-09]]
    [[-2.60209185e-10 -1.10226380e-09]]
    [[-2.40620991e-10 -1.01928688e-09]]
    [[-2.22507370e-10 -9.42556345e-10]]
    [[-2.05757318e-10 -8.71601985e-10]]
    [[-1.90268187e-10 -8.05988972e-10]]
    [[-1.75945056e-10 -7.45315218e-10]]
    [[-1.62700152e-10 -6.89208902e-10]]
    [[-1.50452305e-10 -6.37326193e-10]]
    [[-1.39126460e-10 -5.89349144e-10]]
    [[-1.28653210e-10 -5.44983741e-10]]
    [[-1.18968371e-10 -5.03958106e-10]]
    [[-1.10012594e-10 -4.66020825e-10]]
    [[-1.01730995e-10 -4.30939411e-10]]
    [[-9.40728241e-11 -3.98498878e-10]]
    [[-8.69911498e-11 -3.68500424e-10]]
    [[-8.04425743e-11 -3.40760213e-10]]
    [[-7.43869667e-11 -3.15108248e-10]]
    [[-6.87872170e-11 -2.91387327e-10]]
    [[-6.36090088e-11 -2.69452085e-10]]
    [[-5.88206091e-11 -2.49168099e-10]]
    [[-5.43926736e-11 -2.30411063e-10]]
    [[-5.02980671e-11 -2.13066031e-10]]
    [[-4.6511697e-11 -1.9702671e-10]]
    [[-4.30103597e-11 -1.82194807e-10]]
    [[-3.97725983e-11 -1.68479430e-10]]
    [[-3.67785712e-11 -1.55796528e-10]]
    [[-3.40099303e-11 -1.44068377e-10]]
    [[-3.14497090e-11 -1.33223105e-10]]
    [[-2.90822177e-11 -1.23194251e-10]]
    [[-2.68929479e-11 -1.13920355e-10]]
    [[-2.48684833e-11 -1.05344586e-10]]
    [[-2.29964177e-11 -9.74143886e-11]]
    [[-2.12652786e-11 -9.00811659e-11]]
    [[-1.96644574e-11 -8.32999782e-11]]
    [[-1.81841438e-11 -7.70292691e-11]]
    [[-1.68152661e-11 -7.12306104e-11]]
    [[-1.55494358e-11 -6.58684669e-11]]
    [[-1.43788954e-11 -6.09099783e-11]]
    [[-1.32964717e-11 -5.63247579e-11]]
    [[-1.22955314e-11 -5.20847067e-11]]
    [[-1.13699405e-11 -4.81638408e-11]]
    [[-1.05140268e-11 -4.45381324e-11]]
    [[-9.72254518e-12 -4.11853623e-11]]
    [[-8.99064520e-12 -3.80849842e-11]]
    [[-8.31384165e-12 -3.52179984e-11]]
    [[-7.68798695e-12 -3.25668353e-11]]
    [[-7.10924575e-12 -3.01152483e-11]]
    Gradient descent result: x_min, y_min = [[-7.10924575e-12]
     [-3.01152483e-11]]


Obviously, gradient descent will converge much slower than Newton's method. And trying to increase learning rate, it might not even work at all. This illustrates again the disadvantages of gradient descent in comparison with Newton's method. However, note, that Newton's method required calculation of an inverted Hessian matrix, which is a very computationally expensive calculation to perform when you have, say, a thousand of parameters.

Well done on finishing this lab!


```python

```
