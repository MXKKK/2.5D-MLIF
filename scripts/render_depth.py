import sys

sys.path.append('.')

import lib.renderer.opengl_util as opengl_util
from lib.renderer.mesh import load_fit_body, load_scan, compute_tangent, load_ori_fit_body
import lib.renderer.prt_util as prt_util
from lib.renderer.gl.init_gl import initialize_GL_context
from lib.renderer.gl.prt_render import PRTRender
from lib.renderer.gl.color_render import ColorRender
from lib.renderer.camera import Camera
import argparse
import os
import glob
import cv2
import numpy as np
import random
import math
import time
import trimesh
from matplotlib import cm

t0 = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--subject', type=str, help='subject name')
parser.add_argument('-o', '--out_dir', type=str, help='output dir')
parser.add_argument('-r', '--rotation', type=str, help='rotation num')
parser.add_argument('-w', '--size', type=str, help='render size')
args = parser.parse_args()

subject = args.subject
save_folder = args.out_dir
rotation = int(args.rotation)
size = int(args.size)

# headless
egl = False

# render
initialize_GL_context(width=size, height=size, egl=egl)

dataset = save_folder.split("/")[-1].split("_")[0]
subject = subject.split("/")[-1]

format = 'obj'
scale = 100.0
up_axis = 1
pcd = False
smpl_type = "smplx"
with_light = True
depth = True
normal = True

mesh_file = os.path.join(f'./data/{dataset}/scans',
                         f'{subject}.{format}')
smplx_file = f'./data/{dataset}/smpl/{subject}.obj'
fit_file = f'./data/{dataset}/fits/{subject}.npz'

print(mesh_file)
# mesh
mesh = trimesh.load(mesh_file,
                    skip_materials=True,
                    process=False,
                    maintain_order=True,
                    force='mesh')

if not pcd:
    vertices, faces, normals, faces_normals, textures, face_textures = load_scan(
        mesh_file, with_normal=True, with_texture=True)
else:
    # remove floating outliers of scans
    mesh_lst = mesh.split(only_watertight=False)
    comp_num = [mesh.vertices.shape[0] for mesh in mesh_lst]
    mesh = mesh_lst[comp_num.index(max(comp_num))]

    vertices = mesh.vertices
    faces = mesh.faces
    normals = mesh.vertex_normals

# center

scan_scale = 1.8 / (vertices.max(0)[up_axis] - vertices.min(0)[up_axis])
rescale_fitted_body, joints = load_fit_body(fit_file,
                                            scale,
                                            smpl_type='smplx',
                                            smpl_gender='male')

os.makedirs(os.path.dirname(smplx_file), exist_ok=True)
ori_smplx = load_ori_fit_body(fit_file, smpl_type='smplx', smpl_gender='male')
ori_smplx.export(smplx_file)

vertices *= scale
vmin = vertices.min(0)
vmax = vertices.max(0)
vmed = joints[0]
vmed[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])

rndr_depth = ColorRender(width=size, height=size, egl=egl)
rndr_depth.set_mesh(rescale_fitted_body.vertices, rescale_fitted_body.faces,
                    rescale_fitted_body.vertices,
                    rescale_fitted_body.vertex_normals)
rndr_depth.set_norm_mat(scan_scale, vmed)

# camera
cam = Camera(width=size, height=size)
cam.ortho_ratio = 0.4 * (512 / size)


for y in range(0, 360, 360 // rotation):

    cam.near = -100
    cam.far = 100
    cam.sanity_check()

    R = opengl_util.make_rotate(0, math.radians(y), 0)
    R_B = opengl_util.make_rotate(0, math.radians((y + 180) % 360), 0)

    if up_axis == 2:
        R = np.matmul(R, opengl_util.make_rotate(math.radians(90), 0, 0))


    if smpl_type != "none":
        rndr_depth.rot_matrix = R
        rndr_depth.set_camera(cam)

    dic = {
        'ortho_ratio': cam.ortho_ratio,
        'scale': scan_scale,
        'center': vmed,
        'R': R
    }



    # ==================================================================

    # front render
    if smpl_type != "none":
        rndr_depth.display()
        opengl_util.render_result(
            rndr_depth, 2,
            os.path.join(save_folder, subject, 'T_depth_F',
                        f'{y:03d}.png'))

    # ==================================================================
    # back render
    cam.near = 100
    cam.far = -100
    cam.sanity_check()


    if smpl_type != "none":
        rndr_depth.set_camera(cam)
        rndr_depth.display()
        opengl_util.render_result(
            rndr_depth, 2,
            os.path.join(save_folder, subject, 'T_depth_B',
                            f'{y:03d}.png'))

done_jobs = len(glob.glob(f"{save_folder}/*/render"))
all_jobs = len(os.listdir(f"./data/{dataset}/scans"))
print(
    f"Finish rendering {subject}| {done_jobs}/{all_jobs} | Time: {(time.time()-t0):.0f} secs"
)
