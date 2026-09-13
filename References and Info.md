# REFERENCES AND INFO 

The video links and information given below are the references and info that might help you to understand the project, its functionaning and its physics more efficiently. I suggest going through this to understand the mechanism of how the things are working before you open the code to learn.

## An introduction to rejection sampling - Ben Lambert
https://youtu.be/kYWHfgkRc9s?si=ooHQOY9czt_JPjm3

This project relied on Monte Carlo rejection sampling as discussed in the video to create the electron cloud corresponding to the appropriate orbital. The Schrödinger equation involves dealing with a probability density $\vert{}\psi\vert{}^2$ function rather than with straightforward $(x, y, z)$ coordinates, which is why I had to sample several points uniformly inside a bounding sphere at random as well as test them with random test values. I compute the actual value of the wavefunction's density at the sampled point; in case I obtain $u \le \vert{}\psi(r, \theta, \phi)\vert{}^2$, I keep the point. Otherwise, it is discarded. By repeating this operation many times and, with the acceptance procedure, we get a clear three-dimensional representation of the electron cloud through the clustering of the accepted points.

## Quantum Wavefunction | Quantum physics | Physics | Khan Academy - Khan Academy
https://youtu.be/OFwskHrtYQ4?si=fNzYtAlTQibB1iBc

In this project, I applied the Born Rule and the concept of the Schrödinger wavefunction discussed in the video in order to estimate the probabilities of finding the electron at certain locations. In classical physics, the behavior of an electron is considered similar to that of a planet, but quantum mechanics defines it as a probability wave. In the project, my physics engine determines the radial $R_{n,l}(r)$ and angular components $Y_{l}^m(\theta, \phi)$ of an electron in an ion $\text{Li}^{2+}$. By multiplying the components together, I receive the complete wavefunction $\vert{}\psi\vert{}^2$ and then apply the findings to estimate the physical probabilities of the situation. This information is utilized by the simulator to determine the areas of more and less dense electron clouds.

## Spherical Harmonics and Atomic Orbitals - MrJackpots
https://youtu.be/9nj68x24Wug?si=7eqxCEl6yoaWxUAC

Spherical Harmonics theory has been used by me in this project where I controlled 3D shapes of orbitals as shown in the video. The Schrödinger equation allows splitting electron’s wavefunction into space part $R(r)$ and direction part $Y(\theta, \phi)$. Spherical harmonics techniques  $Y_l^m$, based on quantum numbers $l$ and $m$, were used by me to calculate directional shape. For instance, $l=0$, gives a spherically symmetric orbital (no orientation), whereas $l=2$ creates clover-shaped orbital. Combining the developed angular shapes with radial decay allowed me to plot electron probability density and where the exact nodal planes appear.





