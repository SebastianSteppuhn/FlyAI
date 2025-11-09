import meshio
m = meshio.read("mesh.cgns")
print(sorted(m.cell_sets_dict.keys()))