from pathlib import Path

from matplotlib import animation

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

from common.lattice_tools.common import change_basis
from md.configuration import MDConfig

anim_duration = 20
fps = 15
working_directory = Path("/Users/jeremywilkinson/research_data/md_data/test")
config = MDConfig.load(working_directory)
position_deltas = np.load(working_directory / 'substrate_last_deltas.npy')
absorbate_positions = np.load(working_directory / 'absorbate_positions.npy')[:, -position_deltas.shape[1]:]

pnts = config.equilibrium_lattice_coordinates + position_deltas[:, 0]


num_frames = anim_duration * fps
bounding_parallelpiped = change_basis(config.in_plane_basis, config.lattice_shape + [1, 1, 1])
bounding_parallelpiped_offset = change_basis(config.in_plane_basis, np.asarray([0.5, 0.5, 0.5]))


fig = plt.figure()
ax = p3.Axes3D(fig)

substrate = ax.scatter3D(*pnts, s=10, alpha=0.5)
absorbate = ax.scatter3D(*absorbate_positions[:, 0], s=50)


def update(idx):
    pnts = config.equilibrium_lattice_coordinates + position_deltas[:, idx]
    substrate._offsets3d = tuple(pnts.reshape((3, -1)))

    absorbate_lattice_coords = change_basis(np.linalg.inv(config.in_plane_basis), absorbate_positions[:, idx])
    first_cell_absorbate_lattice_coords = absorbate_lattice_coords + [0.5, 0.5, 0.0]
    first_cell_absorbate_lattice_coords[:2] %= config.lattice_shape[:2]
    first_cell_absorbate_lattice_coords -= [0.5, 0.5, 0.0]
    first_cell_absorbate_position = change_basis(config.in_plane_basis, first_cell_absorbate_lattice_coords)
    absorbate._offsets3d = first_cell_absorbate_position.reshape((3, -1))

    print(f"{idx} / {position_deltas.shape[1]}")
    return substrate, absorbate

# Set limits so that plot aspect ratios are equal

X, Y, Z = pnts

max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

mid_x = (X.max()+X.min()) * 0.5
mid_y = (Y.max()+Y.min()) * 0.5
mid_z = (Z.max()+Z.min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)


anim = animation.FuncAnimation(fig, update, np.linspace(0, position_deltas.shape[1] - 1, num_frames, dtype=int), blit=False)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800, extra_args=['-vcodec', 'libx264'])

ax.view_init(elev=100, azim=0)
anim.save(working_directory / 'crystal_top.mp4', writer=writer)
ax.view_init(elev=0, azim=0)
anim.save(working_directory / 'crystal_side.mp4', writer=writer)

plt.close()
