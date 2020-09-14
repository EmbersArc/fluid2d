# fluid2d
Fluid simulation with Numpy, Pytorch and Eigen

<p align="center">
  <img width="512" height="512" src="https://thumbs.gfycat.com/HeartfeltHopefulDiscus-small.gif">
</p>

[Higher Quality Video](https://gfycat.com/heartfelthopefuldiscus)

Based on [Real-Time Fluid Dynamics for Games](https://www.researchgate.net/publication/2560062_Real-Time_Fluid_Dynamics_for_Games) by Jos Stam


## Performance Comparison
FPS values for different grid sizes.
| N | Numpy | Pytorch (CUDA) | Eigen |
|---|-------|---------|-------|
|128|160|60|**320**|
|256|40|75|**85**|
|512|8|**65**|15|
|1024|1.5|**30**|2.5|
