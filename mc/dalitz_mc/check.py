import numpy as np

data = np.load("eventi_d0_2.npz")
s12 = data["s12"]
s23 = data["s23"]
p1 = data["p1"]
p2 = data["p2"]
p3 = data["p3"]
M, m1, m2, m3 = data["metadata"]

# Ricalcoliamo gli invarianti direttamente dai quadri-vettori salvati
p1_dot_p2 = p1[:, 0]*p2[:, 0] - np.sum(p1[:, 1:] * p2[:, 1:], axis=1)
s12_from_p = m1**2 + m2**2 + 2 * p1_dot_p2

p2_dot_p3 = p2[:, 0]*p3[:, 0] - np.sum(p2[:, 1:] * p3[:, 1:], axis=1)
s23_from_p = m2**2 + m3**2 + 2 * p2_dot_p3

print(f"File {data}")
print(f"s12 salvato vs ricalcolato (max diff): {np.max(np.abs(s12 - s12_from_p)):.2e}")
print(f"s23 salvato vs ricalcolato (max diff): {np.max(np.abs(s23 - s23_from_p)):.2e}")
print(f"s12 min: {np.min(s12):.4f}, s12 max: {np.max(s12):.4f}")
print(f"s23 min: {np.min(s23):.4f}, s23 max: {np.max(s23):.4f}")