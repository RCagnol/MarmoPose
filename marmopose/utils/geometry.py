import numpy as np
import open3d as o3d

def unit(v):
    # Return unit vector of vector v.
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def orthonormal_basis(u):
    # Build an orthonormal basis (u, v, w) where u is normalized input axis.
    u = unit(u)
    if abs(u[2]) < 0.99:
        v = np.array([-u[1], u[0], 0.0])
    else:
        v = np.array([0.0, 1.0, 0.0])
    v = unit(v)
    w = np.cross(u, v)
    return u, v, w

def orthogonalize_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u - project_vector(v, u)


def project_vector(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return u * np.dot(v, u) / np.dot(u, u)


def rotation_matrix_rodrigues(or_dir, new_dir):
    """
    Compute rotation matrix using Rodrigues' formula.
    
    Rotates from or_dir to new_dir.
    
    Parameters:
    -----------
    or_dir : array-like, shape (3,)
        Initial direction vector (will be normalized)
    new_dir : array-like, shape (3,)
        Target direction vector (will be normalized)
    
    Returns:
    --------
    R : ndarray, shape (3, 3)
        Rotation matrix
    """
    # Normalize input vectors
    
    norm_or_dir = unit(np.array(or_dir, dtype=float))
    norm_new_dir = unit(np.array(new_dir, dtype=float))
    
    # Compute rotation axis using cross product
    v = np.cross(norm_or_dir, norm_new_dir)
    
    # Compute sine and cosine of rotation angle
    s = np.linalg.norm(v)  # sin(theta)
    c = np.dot(norm_or_dir, norm_new_dir)        # cos(theta)
    
    # Handle edge cases (parallel vectors)
    if s < 1e-10:  # Vectors are parallel or nearly parallel
        if c > 0:  # Same direction
            return np.eye(3)
        else:  # Opposite directions
            # Find arbitrary perpendicular axis
            axis = np.array([1.0, 0.0, 0.0]) if abs(norm_or_dir[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            v = np.cross(norm_or_dir, axis)
            v = v / np.linalg.norm(v)
            return 2 * np.outer(v, v) - np.eye(3)
    
    # Build skew-symmetric matrix from rotation axis
    v_x = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    # Rodrigues' formula: R = I + v_x + v_x^2 * (1 - cos(theta)) / sin^2(theta)
    R = np.eye(3) + v_x + np.dot(v_x, v_x) * (1 - c) / (s ** 2)
    
    return R

def points_in_cone(points, apex, axis, angle, height, eps=1e-10):
    ap = points - apex
    # Projection of points - apex vectors on the axis of the cone
    dot = np.einsum('...i, ...i ->  ...', ap, axis)
    # Distance between points and apex
    norm_ap = np.linalg.norm(ap, axis = -1)
    # Here we test whether the points are actually so close to the apex that we can consider they overlap
    is_apex = norm_ap < eps
    # Larger angle means larger hypothenuse for a given projection length on axis. So smaller cosine.
    # Thus here the cosyne of the angle formed by points - apex vectors and their projection needs to be larger than the cone angle 
    valid_angle = dot/norm_ap >= np.cos(angle)
    valid_height = np.logical_and(dot > eps, dot < height)
    return np.logical_or(is_apex, np.logical_and(valid_height,valid_angle))

def cones_intersect(apexes1, axes1, heights1, angles1,
                              apexes2, axes2, heights2, angles2,
                              num_t=3, num_phi=4, eps=1e-10):
    """
    Vectorized intersection check for N cone pairs.
    Inputs: (N,3) for apexes/axes, (N,) for heights/angles.
    Returns: (N,) boolean array.
    """
    N = apexes1.shape[0]

    # Quick rejection: distance between apexes > sum of heights
    d_norm = np.linalg.norm(apexes2 - apexes1, axis=1)
    valid_dist = d_norm <= (heights1 + heights2)

    # Quick check: apexes inside other cone
    in1 = points_in_cone(apexes2, apexes1, axes1, heights1, angles1, eps)
    in2 = points_in_cone(apexes1, apexes2, axes2, heights2, angles2, eps)
    valid_apex = in1 | in2

    # Quick valid pairs (no need for surface sampling)
    quick_valid = valid_dist & valid_apex

    # For remaining pairs, generate surface sample points on cone1
    # Get perpendicular basis vectors for each cone1
    use_x = np.abs(axes1[:, 0]) < 0.5
    initial = np.zeros((N, 3))
    initial[use_x] = [1, 0, 0]
    initial[~use_x] = [0, 1, 0]

    a1 = initial - np.einsum('ij,ij->i', initial, axes1)[:, None] * axes1
    a1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    b1 = np.cross(axes1, a1)

    tan1 = np.tan(angles1)

    # Generate sample points on all cone1 surfaces
    t_vals = np.linspace(0, 1, num_t)  # (num_t,)
    phi_vals = np.linspace(0, 2*np.pi, num_phi, endpoint=False)  # (num_phi,)

    # Expand for broadcasting: (N, 1, 1, 3) and (1, num_t, 1, 1) etc.
    apex1_exp = apexes1[:, None, None, :]
    axes1_exp = axes1[:, None, None, :]
    a1_exp = a1[:, None, None, :]
    b1_exp = b1[:, None, None, :]
    h1_exp = heights1 if isinstance(heights1, float) or isinstance(heights1, int) else heights1[:, None, None, None]
    tan1_exp = tan1 if isinstance(tan1, float) else tan1[:, None, None, None]

    t_exp = t_vals[None, :, None, None]
    cos_phi = np.cos(phi_vals)[None, None, :, None]
    sin_phi = np.sin(phi_vals)[None, None, :, None]

    # P = apex + t*h*v + t*h*tanθ*(cosφ*a + sinφ*b)
    points = (apex1_exp +
              t_exp * h1_exp * axes1_exp +
              t_exp * h1_exp * tan1_exp * (cos_phi * a1_exp + sin_phi * b1_exp))

    # Reshape to (N * num_t * num_phi, 3)
    points = points.reshape(-1, 3)

    # Repeat cone2 parameters for each sample point
    apex2_rep = np.repeat(apexes2, num_t * num_phi, axis=0)
    axes1_rep = np.repeat(axes2, num_t * num_phi, axis=0)
    h2_rep = heights2 if isinstance(heights2, float) or isinstance(heights2, int) else np.repeat(heights2, num_t * num_phi)
    a2_rep = angles2 if isinstance(angles2, float) else np.repeat(angles2, num_t * num_phi)

    # Check if any sample point is in its corresponding cone2
    in_cone2 = points_in_cone(points, apex2_rep, axes1_rep, h2_rep, a2_rep, eps)
    any_in_cone2 = np.any(in_cone2.reshape(N, -1), axis=1)

    return quick_valid | any_in_cone2


# Cone class: stores apex, axis (unit), height and degree
class Cone:
    def __init__(self, apex, axis, height, degree, with_mesh = False, resolution = 16, color = (0.8,0.5,0.)):
        self.apex = np.asarray(apex, dtype=float)
        self.u = unit(axis)
        self.h = float(height)
        self.d = float(degree)
        self.with_mesh = with_mesh
        self.hidden = False
        if self.with_mesh:
            self.init_mesh(resolution, color)
            self.color = color

    def init_mesh(self, resolution, color):
        self.mesh = o3d.geometry.TriangleMesh.create_cone(radius=np.tan(np.pi * self.d / 360) * self.h, height=self.h, resolution=resolution, split=1)
        self.mesh.compute_vertex_normals()
        self.mesh.paint_uniform_color(color)
        self.mesh.translate(([0, 0, - self.h]))
        self.mesh.rotate(rotation_matrix_rodrigues([0,0,-1], self.u), center=(0, 0, 0))
        self.mesh.translate(self.apex)


    def update(self, apex, axis):
        """
        Moves the cone to a new apex/axis. Returns False and leaves the cone
        untouched if apex/axis is invalid (e.g. NaN because the underlying
        keypoints are missing) - translate/rotate are relative to the mesh's
        current vertices, so applying a NaN delta would corrupt them
        permanently, even once valid data returns. Callers should treat a
        False return as "hide this cone".
        """
        apex = np.asarray(apex, dtype=float)
        norm_axis = unit(axis)
        if not (np.all(np.isfinite(apex)) and np.all(np.isfinite(norm_axis))):
            return False

        if self.with_mesh:
            translation = apex - self.apex
            rotation = rotation_matrix_rodrigues(self.u, norm_axis)
        self.apex = apex
        self.u = norm_axis
        if self.with_mesh:
            self.update_mesh(translation, rotation)
        return True

    def update_color(self, color = (0.8,0.5,0.)):
        if color != self.color:
            # print('UPDATE', color, self.color)
            self.mesh.paint_uniform_color(color)
            self.color = color
    
    def update_mesh(self, translation, rotation):
        self.mesh.translate(translation)
        self.mesh.rotate(rotation, center = self.apex)
    
    def get_mesh(self):
        return None if not self.with_mesh else self.mesh


