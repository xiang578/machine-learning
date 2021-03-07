# %% 19.3.1 特性会覆盖实例属性

class Class:
    data = 'the class data attr'
    @property 
    def prop(self):
        return 'the prop value'

obj = Class()
print(Class.prop)
print(obj.prop)

Class.prop = 'test'
print(Class.prop)
print(obj.prop)
print(Class.__slots__)
# %%

# %% 11.10 goose-typing

class Struggle: 
    def __len__(self):
        return 23 
from collections import abc 
print(isinstance(Struggle(), abc.Sized))
print(issubclass(Struggle, abc.Sized))
# %%
