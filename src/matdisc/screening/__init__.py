"""Machine-learning-potential relaxation, convex-hull and phonon screening.

Three steps, each in its own module and usable on its own:

* :mod:`matdisc.screening.relax` -- relax candidates with an ASE calculator and collect
  their energies.
* :mod:`matdisc.screening.hull` -- place those energies on the convex hull of the competing
  phases and report an energy above the hull per candidate.
* :mod:`matdisc.screening.phonons` -- check the survivors for imaginary phonon frequencies.

:func:`matdisc.screening.hull.toy_system` gives a small offline example that needs no
calculator, no credentials and no network.
"""
