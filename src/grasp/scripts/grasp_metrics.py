# !/usr/bin/env python -W ignore::DeprecationWarning
"""
Grasp Metrics for C106B Grasp Planning Lab
Author: Chris Correa
"""
import numpy as np
from utils import vec, adj, look_at_general, find_grasp_vertices, normal_at_point, hat, normalize
from casadi import Opti, sin, cos, tan, vertcat, mtimes, sumsqr, sum1
from scipy.spatial.transform import Rotation

# Can edit to make grasp point selection more/less restrictive
MAX_GRIPPER_DIST = .070
MIN_GRIPPER_DIST = .03

def get_friction_cone(mu, num_facets, normal):
        alpha = np.arctan(mu)
        facet_rotation = 2 * np.pi/num_facets
        friction_cone_edges = np.zeros((num_facets, 3))

        Rx = lambda theta: np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        Rz = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), -np.cos(theta), 0], [0, 0, 1]])

        starting_edge = normalize(np.matmul(Rx(alpha), normal))
        for i in range(num_facets):
            #Construct matrix for rotation around normal direction
            normal_rotation = Rz(i * facet_rotation)

            #Compute rotated edge and add it to list of friction cone edges
            edge = np.matmul(normal_rotation, starting_edge)
            friction_cone_edges[i] = normalize(edge)
        return friction_cone_edges

def compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Compute the force closure of some object at contacts, with normal vectors 
    stored in normals. You can use the line method described in the project document.
    If you do, you will not need num_facets. This is the most basic (and probably least useful)
    grasp metric.
    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object
    Returns
    -------
    float : 1 or 0 if the grasp is/isn't force closure for the object
    """
    normal0 = -1.0 * normals[0] / (1.0 * np.linalg.norm(normals[0]))
    normal1 = -1.0 * normals[1] / (1.0 * np.linalg.norm(normals[1]))

    alpha = np.arctan(mu)
    line = vertices[0] - vertices[1]
    line = line / (1.0 * np.linalg.norm(line))
    angle1 = np.arccos(normal1.dot(line))

    line = -1 * line
    angle2 = np.arccos(normal0.dot(line))

    if angle1 > alpha or angle2 > alpha:
        return 0
    if gamma == 0:
        return 0
    return 1

def get_grasp_map(vertices, normals, num_facets, mu, gamma):
    """ 
    Defined in the book on page 219. Compute the grasp map given the contact
    points and their surface normals
    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    Returns
    -------
    6x8 :obj:`numpy.ndarray` : grasp map
    """
    def adj_inv(transform):
        adj = np.zeros((6, 6))
        rotation = transform[0:3, 0:3]
        point = transform[0:3, 3]
        adj[0:3, 0:3] = rotation
        adj[3:6, 3:6] = rotation
        p_hat = hat(point)
        adj[3:6, 0:3] = np.matmul(p_hat, rotation)
        return adj

    grasp_map = np.zeros((6, 8))
    B = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]])
    transform_1 = look_at_general(vertices[0], -normals[0])
    transform_2 = look_at_general(vertices[1], -normals[1])
    adj_1 = adj_inv(transform_1)
    adj_2 = adj_inv(transform_2)
    grasp_map[:, :4] = np.matmul(adj_1, B)
    grasp_map[:, 4:] = np.matmul(adj_2, B)
    return grasp_map

def find_contact_forces(vertices, normals, num_facets, mu, gamma, desired_wrench):
    """
    Compute that contact forces needed to produce the desired wrench
    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    desired_wrench : 6x :obj:`numpy.ndarray` potential wrench to be produced
    Returns
    -------
    bool: whether contact forces can produce the desired_wrench on the object
    """
    raise NotImplementedError

def compute_gravity_resistance(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Gravity produces some wrench on your object. Computes how much normal force is required
    to resist the wrench produced by gravity.
    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object
    Returns
    -------
    float: quality of the grasp
    """

    z = np.array([0, 0, 1])
    print('HERE')
    friction_cone_1_edges = get_friction_cone(mu, num_facets, z)
    friction_cone_2_edges = get_friction_cone(mu, num_facets, z)
    try:
        opti = Opti()

        fa = opti.variable(4, 1)
        fb = opti.variable(4, 1)

        alpha1 = opti.variable(num_facets + 1, 1)
        alpha2 = opti.variable(num_facets + 1, 1)

        f_stacked = vertcat(fa, fb)
        f_magnitude = mtimes(f_stacked.T, f_stacked)
        opti.minimize(f_magnitude)

        constraints = []

        gravity_wrench = np.array([0, 0, -9.8 * object_mass, 0, 0, 0])
        
        alpha_constraints = [alpha1>=0, alpha2>=0]
        constraints.extend(alpha_constraints)
        
        gamma_constraints = [f_stacked[3] <= gamma * f_stacked[2], f_stacked[7] <= gamma * f_stacked[6]]
        constraints.extend(gamma_constraints)

        cone_constraint = [fa[:3] == alpha1[0] * z + sum([alpha1[i+1] * friction_cone_1_edges[i] for i in range(num_facets)]), fb[:3] == alpha2[0] * z + sum([alpha2[i+1] * friction_cone_2_edges[i] for i in range(num_facets)])]
        constraints.extend(cone_constraint)

        grasp_map = get_grasp_map(vertices, normals, num_facets, mu, gamma)
        wrench_constraint = [mtimes(grasp_map, f_stacked) == -gravity_wrench]
        constraints.extend(wrench_constraint)

        opti.subject_to(constraints)

        p_opts = {"expand": False, "print_time": 0}
        s_opts = {"max_iter": 1e4, "print_level": 0}

        opti.solver('ipopt', p_opts, s_opts)
        sol = opti.solve()

        f = sol.value(f_stacked)

        J = lambda f: f[2] + f[6]
        print("Feasible: ", f, J(f))
    except:
        print("Infeasible")
        f = None
        J = lambda f: 1e4
    return -J(f) 

def compute_robust_force_closure(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Should return a score for the grasp according to the robust force closure metric.
    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object
    Returns
    -------
    float: quality of the grasp
    """
    if np.all(np.abs(vertices[0] - vertices[1]) <= .001):
        print("vertices are almost the same: ", vertices)
        return 0
    NUM_PERTURBATIONS = 50
    standard_deviation = 0.0035
    count = 0
    print("computing")
    for _ in range(NUM_PERTURBATIONS):
        #Create random noise with mean and std and add it to first vertex
        random_vertex_noise_1 = np.random.normal(0.0, standard_deviation, (1, 3))
        new_vertex_1 = vertices[0] + random_vertex_noise_1

        #Do the same for second vertex
        random_vertex_noise_2 = np.random.normal(0.0, standard_deviation, (1, 3))
        new_vertex_2 = vertices[1] + random_vertex_noise_2

        if np.all(new_vertex_1 == new_vertex_2):
            continue

        midpoint = (new_vertex_1 + new_vertex_2) / 2.0 
        towards_v2 = new_vertex_2 - new_vertex_1
        towards_v2 = towards_v2 / np.linalg.norm(towards_v2)

        outer_1 = midpoint - towards_v2 * MAX_GRIPPER_DIST / 2.0
        outer_2 = midpoint + towards_v2 * MAX_GRIPPER_DIST / 2.0

        outer_1 = outer_1.reshape((3,))
        outer_2 = outer_2.reshape((3,))
        #Calculate new grasp points if gripper starts from these points
        # print(outer_1, outer_2)
        contact_points, _ = find_grasp_vertices(mesh, outer_1, outer_2)

        if contact_points.shape[0] != 2:
            continue
        if all(contact_points[0] == contact_points[1]):
            continue

        #Get new normal vectors 
        new_normals = np.array([normal_at_point(mesh, point) for point in contact_points])
        if all(new_normals[0] == np.zeros(new_normals[0].shape)) or all(new_normals[1] == np.zeros(new_normals[1].shape)):
            continue

        #Add 1 to count if this grasp is in force closure
        count += compute_force_closure(contact_points, new_normals, num_facets, mu, gamma, object_mass, mesh)
    return (1.0 * count)/NUM_PERTURBATIONS

def compute_ferrari_canny(vertices, normals, num_facets, mu, gamma, object_mass, mesh):
    """
    Should return a score for the grasp according to the Ferrari Canny metric.
    Use your favourite python convex optimization package. We suggest casadi.
    Parameters
    ----------
    vertices : 2x3 :obj:`numpy.ndarray`
        obj mesh vertices on which the fingers will be placed
    normals : 2x3 :obj:`numpy.ndarray`
        obj mesh normals at the contact points
    num_facets : int
        number of vectors to use to approximate the friction cone.  these vectors 
        will be along the friction cone boundary
    mu : float 
        coefficient of friction
    gamma : float
        torsional friction coefficient
    object_mass : float
        mass of the object
    mesh : :obj:`Trimesh`
        mesh object
    Returns
    -------
    float: quality of the grasp
    """

    if(compute_force_closure(vertices, normals, num_facets, mu, gamma, object_mass, mesh) == 0):
        print("not in force closure")
        return 0
    z = np.array([0, 0, 1])
    friction_cone_1_edges = get_friction_cone(mu, num_facets, z)
    friction_cone_2_edges = get_friction_cone(mu, num_facets, z)

    best_metric = None
    N = 200
    for i in range(N):
        w = normalize(np.random.sample((6, 1)))
        w[3:5] = 0
        w = normalize(w)
        feasible = True
        try:
            opti = Opti()

            fa = opti.variable(4, 1)
            fb = opti.variable(4, 1)

            alpha1 = opti.variable(num_facets + 1, 1)
            alpha2 = opti.variable(num_facets + 1, 1)

            f_stacked = vertcat(fa, fb)
            f_magnitude = mtimes(f_stacked.T, f_stacked)
            opti.minimize(f_magnitude)

            constraints = []
            
            alpha_constraints = [alpha1>=0, alpha2>=0]
            constraints.extend(alpha_constraints)
            
            gamma_constraints = [f_stacked[3] <= gamma * f_stacked[2], f_stacked[3] >= -gamma * f_stacked[2], f_stacked[7] <= gamma * f_stacked[6], f_stacked[7] >= -gamma * f_stacked[6]]
            constraints.extend(gamma_constraints)

            cone_constraint = [fa[:3] == alpha1[0] * z + sum([alpha1[i+1] * friction_cone_1_edges[i] for i in range(num_facets)]), fb[:3] == alpha2[0] * z + sum([alpha2[i+1] * friction_cone_2_edges[i] for i in range(num_facets)])]
            constraints.extend(cone_constraint)

            grasp_map = get_grasp_map(vertices, normals, num_facets, mu, gamma)
            wrench_constraint = [mtimes(grasp_map, f_stacked) == w]
            constraints.extend(wrench_constraint)

            opti.subject_to(constraints)

            p_opts = {"expand": False, "print_time": 0}
            s_opts = {"max_iter": 1e4, "print_level": 0}

            opti.solver('ipopt', p_opts, s_opts)
            sol = opti.solve()

            f_mag = sol.value(f_magnitude)
            metric = 1/np.sqrt(f_mag)
        except Exception as e:
            print("Infeasible", e)
            feasible = False
        if feasible and (best_metric == None or metric < best_metric):
            best_metric = metric
    return best_metric