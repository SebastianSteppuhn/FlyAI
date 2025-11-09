import meshio
m = meshio.read("../mesh/mesh.cgns")
print(sorted(m.cell_sets_dict.keys()))