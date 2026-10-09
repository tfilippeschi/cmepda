import math
from array import array

class Vector:
    """Class representing a multidimensional vector."""
    TYPE_CODE = 'd' # Class attribute, shared by all instances of the class

    def __init__(self, components):
        self._components = array(self.TYPE_CODE, components)
    
    def __repr__(self):
        """Calling str() of an array produces a string like
        array ('d', [1.0, 2.0, 3.0]). We remove everything outside the
        square parenthesis and add our class name at the beginning."""
        components = str(self._components)
        components = components[components.find('['): -1]
        return f"{self.__class__.__name__} ({components})"
    
    def __str__(self):
        return str(tuple(self._components))
    
    def __getitem__(self, index):
        return self._components[index]
    
    def __setitem__(self, index, new_value):
        self._components[index] = new_value
    
    def __len__(self):
        return len(self._components)
    
    def __iter__(self):
        return iter(self._components)
    