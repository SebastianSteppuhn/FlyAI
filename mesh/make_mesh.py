import gmsh
gmsh.initialize()
gmsh.model.add("from_step")
gmsh.model.occ.importShapes("../geo/test.step")
gmsh.model.occ.synchronize()

# ... your boolean(s) to make the fluid volume ...
# ... create Physical groups, e.g.:
# gmsh.model.addPhysicalGroup(3, [fluid_vol_tag], name="fluid")
# gmsh.model.addPhysicalGroup(2, inlet_surf_tags,  name="inlet")
# gmsh.model.addPhysicalGroup(2, outlet_surf_tags, name="outlet")
# gmsh.model.addPhysicalGroup(2, wall_surf_tags,   name="walls")

gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.01)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.05)
gmsh.model.mesh.generate(3)
gmsh.option.setNumber("Mesh.SaveAll", 0)      # only physicals
gmsh.write("../mesh/mesh.su2")                # SU2-ASCII
gmsh.finalize()
