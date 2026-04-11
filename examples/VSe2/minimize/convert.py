from ase.io import read, write
import numpy as np
import matplotlib.pyplot as plt

a = read("trajectory.xyz", ":")
hehe = []
for i in range(len(a)):
    a[i].arrays["force"] = a[i].info["spin"]
    hehe.append(np.linalg.norm(a[i].info["spin"].sum(0))/len(a[i])*3)

write("spin_traj.xyz", a)
plt.plot(hehe)
plt.savefig("1.png")
