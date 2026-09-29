# --- Goal
# Write a program to explore the properties of a few elementary Particles.
# The program must contain a Base class Particle and two Child classes, Proton and Alpha, that inherit from it.

# --- Specifications
# - instances of the class Particle must be initialized with their mass, charge, and name (all read-only)
# - the class constructor must also accept (optionally) and store one and only one of the following quantities: energy, momentum, beta or gamma
# - whatever the choice, the user should be able to read and set any of these quantities using just the '.' (dot) operator e.g.
#   print(my_particle.energy), my_particle.beta = 0.5
# - attempts to set non physical values should be rejected
# - the Particle class must have a method to print the Particle information in a formatted way
# - the child classes Alpha and Protons must use class attributes to store their mass, charge and name

import math

class Particle:
    def __init__ (self, mass, charge=0., name=None, momentum=0.):
        self._mass = mass
        self._charge = charge
        self._name = name
        self.momentum = momentum
    
    def energy(self):
        return math.sqrt(self._mass**2 + self.momentum**2)
    
    def info(self):
        return f"Particle: {self._name}\n" \
               f"Mass: {self._mass} MeV/c^2\n" \
               f"Charge: {self._charge} e\n" \
               f"Momentum: {self.momentum} MeV/c\n" \
               f"Energy: {self.energy():.3f} MeV\n"

class Electron(Particle):
    def __init__(self, momentum=0.):
        Particle.__init__(self, mass=0.511, charge=-1., name="electron", momentum=momentum)

class Proton(Particle):
    def __init__(self, momentum=0.):
        Particle.__init__(self, mass=938.272, charge=1., name="proton", momentum=momentum)

if __name__ == "__main__":
    e = Electron(momentum=1.)
    print(e.info())
    print(f"Energy of {e._name} with momentum {e.momentum} MeV is {e.energy():.3f} MeV")