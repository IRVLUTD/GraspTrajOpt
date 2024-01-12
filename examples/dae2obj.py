import os
import sys
import glob
import trimesh

src_dir = sys.argv[1]

for infile in glob.glob(os.path.join(src_dir, '*.dae')):
    print(infile)
    mesh = trimesh.load(infile)

    mtl_name = infile.replace('dae', 'mtl')
    obj, mtl = trimesh.exchange.obj.export_obj(mesh, include_color=True, include_texture=True, return_texture=True, mtl_name=os.path.basename(mtl_name))

    filename = infile.replace('dae', 'obj')
    with open(filename, "w") as file:
        file.write(obj)

    print(mtl)
    with open(mtl_name, 'wb') as f:  
        for key, value in mtl.items():  
            f.write(value)