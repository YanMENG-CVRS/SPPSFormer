import open3d as o3d
import numpy as np
import open3d.core as o3c
def boundary_detection(path,r=0.2,max_nn=30, show=True):
    pcd = o3d.t.geometry.PointCloud()
    data = np.loadtxt(path)
    points = data[:, :3]
    labels = data[:, 3:]
    pcd = o3d.t.geometry.PointCloud(o3c.Tensor(points, o3c.float32))
    if show:
        o3d.visualization.draw_geometries([pcd.to_legacy()])
        pcd.estimate_normals(max_nn=30, radius=0.1)
        o3d.visualization.draw_geometries([pcd.to_legacy()], point_show_normal=True)
        boundarys, mask = pcd.compute_boundary_points(0.2, 30)
        # TODO: not good to get size of points.
        print(f"Detect {boundarys.point.positions.shape[0]} bnoundary points from {pcd.point.positions.shape[0]} points.")
        boundary_array = np.zeros(len(points), dtype='float32')
        boundary_array[mask.cpu().numpy()] = 1
        boundarys = boundarys.paint_uniform_color([1.0, 0.0, 0.0])
        pcd = pcd.paint_uniform_color([0.6, 0.6, 0.6])
        o3d.visualization.draw_geometries([pcd.to_legacy(), boundarys.to_legacy()],
                                      )
    else:
        pcd.estimate_normals(max_nn=30, radius=0.3)
        boundarys, mask = pcd.compute_boundary_points(0.2, 30)
        boundary_array = np.zeros(len(points), dtype='float32')
        boundary_array[mask.cpu().numpy()] = 1
        return boundary_array


def boundary_detection_xyz(xyz, r=0.2, max_nn=30, show=True):
    points = xyz
    pcd = o3d.t.geometry.PointCloud(o3c.Tensor(points, o3c.float32))
    if show:
        o3d.visualization.draw_geometries([pcd.to_legacy()])
        pcd.estimate_normals(max_nn, radius=r)
        o3d.visualization.draw_geometries([pcd.to_legacy()], point_show_normal=True)
        boundarys, mask = pcd.compute_boundary_points(r, max_nn)
        # TODO: not good to get size of points.
        print(
            f"Detect {boundarys.point.positions.shape[0]} bnoundary points from {pcd.point.positions.shape[0]} points.")
        boundary_array = np.zeros(len(points), dtype='float32')
        boundary_array[mask.cpu().numpy()] = 1
        boundarys = boundarys.paint_uniform_color([1.0, 0.0, 0.0])
        pcd = pcd.paint_uniform_color([0.6, 0.6, 0.6])
        o3d.visualization.draw_geometries([pcd.to_legacy(), boundarys.to_legacy()],
                                          )
    else:
        pcd.estimate_normals(max_nn, radius=r)
        boundarys, mask = pcd.compute_boundary_points(r, max_nn)
        boundary_array = np.zeros(len(points), dtype='float32')
        boundary_array[mask.cpu().numpy()] = 1
        return boundary_array
