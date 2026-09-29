import math

class Particle:
    def __init__ (self, mass, charge=0., name=None, momentum=0.):
        self.name = name
        self.mass = mass
        self.charge = charge
        self.momentum = momentum
    
    def energy(self):
        return math.sqrt(self.mass**2 + self.momentum**2)

class Electron(Particle):
    def __init__(self, momentum=0.):
        Particle.__init__(self, mass=0.511, charge=-1., name="electron", momentum=momentum)

class Proton(Particle):
    def __init__(self, momentum=0.):
        Particle.__init__(self, mass=938.272, charge=1., name="proton", momentum=momentum)

if __name__ == "__main__":
    e = Electron(momentum=1.)
    print(f"Energy of {e.name} with momentum {e.momentum} MeV is {e.energy():.3f} MeV")