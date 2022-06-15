# sample_codes

A sample of coding work done during my PhD, which was on statistical fitting of stellar rotation evolution models. Physical models that optimally fit rotation data allow for the more robust use of the gyrochronology technique (whereby a star's age can be inferred from its mass and rotation rate), and also give us a greater insight into the magnetised stellar wind physics (which govern the rotational evolution) of these stars. 


Codes:

calculateSolidBodyEvolution.py:

Given an initial age, mass and angular velocity, the code can evolve a single star's rotation rate forwards in time. Figure rotation_convergence.png shows the rotational evolution of five stars, all solar mass but initialised at different rotation rates. The fastest and slowest stars are depicted with dashed lines.  The rotation rates converge with time. Figure mass_convergence.png similarly shows how the convergence of rotation rates depends on mass. It can also evolve entire clusters of stars (consisting of a range of initial rotation rates and stellar masses) forwards in time, to some desired final age. The resulting evolved cluster can be plotted on a rotation-period vs mass diagram (PMD) to show a distribution of rotation rates for a group of coeval stars. This distribution can then be expressed as a probability density on the PMD using either brute force (see Breimann et al 2021) or deterministic methods (see Breimann et al 2022 in prep). Such a distribution is shown in Figure PMD.png, where a uniform distribution of model stars has been evolved to the age of the Prasepe cluster, shown as white points. 

