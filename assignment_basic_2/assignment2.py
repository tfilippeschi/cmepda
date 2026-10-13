# 2nd assignment (lecture version)

import math

class Particle:
    def __init__(self, mass, charge, name, momentum=0.):
        self._mass = mass
        self._charge = charge
        self._name = name
        if momentum < 0.:
            print("ERROR: particle momentum must be non negative")
            self._momentum = 0.

        else:
            self._momentum = momentum
    
    @property
    def name(self):
        return self._name
    
    @property
    def mass(self):
        return self._mass
    
    @property
    def charge(self):
        return self._charge
    
    @property
    def energy(self):
        return math.sqrt(self.mass ** 2. + self.momentum ** 2.)
    
    @property
    def beta(self):
        return self.momentum / self.energy

    @property
    def momentum(self):
        return self._momentum

    @momentum.setter
    def momentum(self, new_momentum):
        if new_momentum < 0.:
            print("ERROR: particle momentum must be non negative")
            self._momentum = 0.
        else:
            self._momentum = new_momentum
    
    @energy.setter
    def energy(self, new_energy):
        if new_energy < self.mass:
            print("ERROR: energy cannot be smaller than the mass")
            return
        new_momentum = math.sqrt(new_energy ** 2. - self._mass ** 2.)
        self.momentum = new_momentum

    @beta.setter
    def beta(self, new_beta):
        if new_beta < 0. or beta > 1.:
            print("ERROR: beta must be non negative")
        else:
            self.momentum = new_beta * self.mass / (1. - new_beta ** 2.)
    
    def __str__(self):
        return f"Particle {self.name} of mass {self.mass} and charge {self.charge} and momentum {self.momentum}"

class Proton(Particle):
    NAME = "Proton"
    MASS = 938.5
    CHARGE = +1

    def __init__(self, momentum=0.):
        super().__init__(mass=Proton.mass,
                        name=Proton.name,
                        charge=Proton.charge,
                        momentum=momentum)

if __name__ == '__main__':
    muon = Particle(mass=105.6, charge=-1, name='Muon', momentum=45.)
    print(muon._name)
    print(f"Muon energy = {muon.energy}")
    print(f"Muon beta = {muon.beta}")
    muon.energy = 50.
    print(f"Muon energy {muon.energy}")
    muon.energy = 200.
    print(f"Muon energy {muon.energy}")
    muon.momentum = -200.
    new_beta = 2.
    proton = Proton(momentum=200.)
    print(proton)