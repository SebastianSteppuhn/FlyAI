import aspose.cad as cad
from aspose.cad import Image, Color
from aspose.cad.imageoptions import CadRasterizationOptions, PngOptions, RenderMode3D

img = Image.load("plane2.stp")

r = CadRasterizationOptions()
r.page_width = 800.0
r.page_height = 800.0
r.zoom = 1.5
r.background_color = Color.green      # or Color.white
r.embed_background = True             # ensure the color appears in raster output
r.render_mode_3d = RenderMode3D.SOLID # or RenderMode3D.SOLID_WITH_EDGES

# Note: `layers` is for DWG/DXF; it’s not needed for STEP
# r.layers = ["Layer"]

opts = PngOptions()
opts.vector_rasterization_options = r

img.save("result.png", opts)
