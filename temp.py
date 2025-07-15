# %%
import numpy as np

# %%
a = np.random.poisson(1, (2, 4))
print(a)
print(np.array([[2], [2]]) * a)

# %%