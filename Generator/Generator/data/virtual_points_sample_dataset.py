from contextlib import nullcontext
from functools import lru_cache

import numpy as np
import torch
from unicore.data import Dictionary, data_utils
from scipy.spatial import distance_matrix
from unicore.data import BaseWrapperDataset, LRUCacheDataset
from queue import PriorityQueue 
import math
from itertools import combinations
from unicore import utils
import itertools
from scipy.spatial.transform import Rotation as R
from typing import List, Callable, Any, Dict
from scipy.linalg import svd

def random_rotation_z(coords):
    """
    Apply a random rotation around the z-axis to a set of coordinates.
    
    Parameters:
    - coords: A numpy array of shape (n_points, 3)
    
    Returns:
    - A numpy array of shape (n_points, 3) with the coordinates after rotation
    """
    theta = np.radians(np.random.uniform(0, 360))  # Random angle for rotation
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    
    return coords.dot(rotation_matrix)

# def rotation_matrix_from_vectors(vec1, vec2):
#     a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
#     v = np.cross(a, b)
#     c = np.dot(a, b)
#     s = np.linalg.norm(v)
#     kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
#     rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
#     return rotation_matrix

def calculate_centroid(coords):
    return np.mean(coords, axis=0)

def calculate_normal_vector(coords):
    centroid = calculate_centroid(coords)
    coords_centered = coords - centroid
    _, _, vh = svd(coords_centered, full_matrices=False)
    normal = vh[2, :]
    return normal

def rotate_to_xy_plane(coords):
    normal = calculate_normal_vector(coords)
    rotation_matrix = rotation_matrix_from_vectors(normal, np.array([0, 0, 1]))
    return coords.dot(rotation_matrix)

def rotate_coordinates(coords_np):
    if not isinstance(coords_np, np.ndarray):
        raise ValueError("The input must be a numpy array.")
    rotated_coords = rotate_to_xy_plane(coords_np)
    return rotated_coords

def rotation_matrix_from_vectors(vec1, vec2):
    """Create a rotation matrix that rotates vec1 to vec2."""
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def pca_based_rotation(coords):
    """Rotate the coordinates based on PCA to align with the principal axes."""
    # Calculate the centroid
    centroid = calculate_centroid(coords)
    # Center the coordinates
    coords_centered = coords - centroid
    # Perform SVD
    _, _, vh = svd(coords_centered, full_matrices=False)
    # Create the rotation matrix using the first two principal components
    rotation_matrix = rotation_matrix_from_vectors(vh[0], np.array([1, 0, 0]))
    rotation_matrix = rotation_matrix.dot(rotation_matrix_from_vectors(vh[1], np.array([0, 1, 0])))
    # Apply the rotation matrix to the centered coordinates
    rotated_coords = coords_centered.dot(rotation_matrix.T)
    # Translate the coordinates back to the original center
    rotated_coords += centroid
    return rotated_coords

def add_proportional_noise(coords, noise_factor=0.01):
    """
    Add noise to the coordinates based on the largest range of XYZ axes.
    
    Parameters:
    - coords: A numpy array of shape (n_points, 3)
    - noise_factor: The factor of noise to add based on the largest axis range
    
    Returns:
    - A numpy array of shape (n_points, 3) with the coordinates after adding noise
    """
    # Calculate the range of each axis
    ranges = np.ptp(coords, axis=0)
    # Determine the largest range
    max_range = np.max(ranges)
    # Calculate noise level based on the largest range
    noise_level = max_range * noise_factor
    # Add noise to each axis proportionally
    noise = np.random.uniform(-noise_level, noise_level, coords.shape)
    return coords + noise

def softmax(x):
    max_num = np.max(x) 
    e_x = np.exp(x - max_num) 
    sum_num = np.sum(e_x) 
    f_x = e_x / sum_num 
    return f_x


import os
from datetime import datetime

def save_to_xyz(coord_tensor, atom_symbols, base_file_path):
    """
    Save the coordinates and corresponding atom types to an XYZ file.
    If the file already exists, appends a unique index to the file name.
    
    Parameters:
    coord_tensor (torch.Tensor): A tensor of shape (N, 3) where N is the number of atoms.
    atom_symbols (list): A list of atomic symbols corresponding to each atom.
    base_file_path (str): The base file path to save the XYZ file.
    """
    # Ensure the tensor is on CPU and convert to numpy
    coord = coord_tensor.cpu().numpy()
    
    # Check that the number of symbols matches the number of coordinates
    assert len(atom_symbols) == len(coord), "The number of atomic symbols must match the number of coordinates"
    
    # Generate a unique file name if file already exists

    directory = os.path.dirname(base_file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = base_file_path
    file_index = 0
    while os.path.exists(file_path):
        file_index += 1
        file_path = f"{base_file_path.rstrip('.xyz')}_{file_index}.xyz"

    # Open the file at the specified path
    with open(file_path, 'w') as xyz_file:
        # Write the number of atoms on the first line
        xyz_file.write(f"{len(coord)}\n")
        # Write a comment line with a timestamp
        xyz_file.write(f"XYZ file generated by PyTorch at {datetime.now()}\n")
        # Write each atom's data
        for symbol, pos in zip(atom_symbols, coord):
            x, y, z = pos
            xyz_file.write(f"{symbol} {x:.6f} {y:.6f} {z:.6f}\n")
    
    return file_path  # Return the path of the file that was actually written

def generate_cube_edges(x_min, x_max, y_min, y_max, z_min, z_max, points_per_edge):
    x_points = np.linspace(x_min, x_max, num=points_per_edge)
    y_points = np.linspace(y_min, y_max, num=points_per_edge)
    z_points = np.linspace(z_min, z_max, num=points_per_edge)

    edge_points = []
    for x in x_points:
        edge_points.append([x, y_min, z_min])
        edge_points.append([x, y_max, z_min])
        edge_points.append([x, y_min, z_max])
        edge_points.append([x, y_max, z_max])
    for y in y_points:
        edge_points.append([x_min, y, z_min])
        edge_points.append([x_max, y, z_min])
        edge_points.append([x_min, y, z_max])
        edge_points.append([x_max, y, z_max])
    for z in z_points:
        edge_points.append([x_min, y_min, z])
        edge_points.append([x_max, y_min, z])
        edge_points.append([x_min, y_max, z])
        edge_points.append([x_max, y_max, z])

    return np.unique(edge_points, axis=0)


class VirtualPointsSampleDataset(BaseWrapperDataset):

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        coord_dataset: torch.utils.data.Dataset,
        frag_dataset: torch.utils.data.Dataset,
        # frag_id_dataset: torch.utils.data.Dataset,
        # frag_coord_dataset: torch.utils.data.Dataset,
        bond_type_dataset: torch.utils.data.Dataset,
        vocab: Dictionary,
        pad_idx: int,
        mask_idx: int,
        null_idx: int,
        noise_type: str,
        noise: float = 1.0,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        neg_num: int = 10,
        temperature: float = 100.0,
        args=None,
        frag_coord_dataset=None,
        dictionary=None
    ):
        assert 0.0 < mask_prob #< 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.coord_dataset = coord_dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx 
        self.null_idx = null_idx
        self.noise_type = noise_type
        self.noise = noise
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.neg_num = neg_num
        self.frag_dataset = frag_dataset
        self.frag_coord_dataset = frag_coord_dataset
        self.bond_dataset = bond_type_dataset
        self.temperature = temperature
        self.args = args
        self.dictionary = dictionary
        self.indices = {k:v for k,v in enumerate(self.dictionary.indices)}

        if random_token_prob > 0.0:
            weights = np.ones(len(self.vocab))
            weights[vocab.special_index()] = 0
            self.weights = weights / weights.sum()

        self.epoch = None # None
        # self.set_epoch(1)
        if self.noise_type == 'trunc_normal':
            self.noise_f = lambda num_mask,noise: np.clip(np.random.randn(num_mask, 3) * noise, a_min=-noise*2.0, a_max=noise*2.0)
        elif self.noise_type == 'normal':
            self.noise_f = lambda num_mask,noise: np.random.randn(num_mask, 3) * noise
        elif self.noise_type == 'uniform':
            self.noise_f = lambda num_mask,noise: np.random.uniform(low=-noise, high=noise, size=(num_mask, 3))
        else:
            self.noise_f = lambda num_mask,noise: 0.0
        

    def set_epoch(self, epoch, **unused):
        super().set_epoch(epoch)
        self.coord_dataset.set_epoch(epoch)
        self.epoch = epoch

    def __getitem__(self, index: int):
        return self.__getitem_cached__(self.epoch, index)

    @lru_cache(maxsize=16)
    def __getitem_cached__(self, epoch: int, index: int):
        ret = {}
        
        with data_utils.numpy_seed(self.seed, epoch, index):
            item = self.dataset[index]
            coord = self.coord_dataset[index]
            # ret['origin_coord'] = torch.from_numpy(coord).float()
            # atom_symbol = [self.indices[i.item()] for i in item]
            # ret['origin_atom'] = torch.from_numpy(np.array(atom_symbol))
            # if self.args.compress_xy > 0:
            #     coord = rotate_coordinates(coord)
            #     if self.args.z_rotate > 0 :
            #        coord = random_rotation_z(coord)

            if self.args.compress_xy > 0:
                if not np.all(coord[:, 2] == 0):
                    coord = rotate_coordinates(coord)
                    if self.args.z_rotate > 0:
                        coord = random_rotation_z(coord)

            if self.args.compress_optim > 0:
                coord = pca_based_rotation(coord)
                if hasattr(self.args, "compress_optim_noise"):
                    if self.args.compress_optim_noise > 0:
                        coord = add_proportional_noise(coord, noise_factor=0.7)




            sz = len(item)
            assert sz > 0
            if self.mask_prob == 1:
                frag_selected = [list(range(sz))]
            elif self.args.optim_frag is not None:
                frag_selected = [list(set(self.args.optim_frag))]

            elif self.args.cos_mask > 0:
                central = coord.mean(axis=0)
                direction =  np.random.randn(1, 3)
                direction = direction/np.linalg.norm(direction,axis=1)
                direction = direction.reshape(-1,1)
                project = coord.dot(direction)
                central_proj = central.reshape(1,-1).dot(direction)
                project = (project - central_proj).reshape(-1) 
                point_select = np.argmax(project)
                if self.args.cos_distance > 0:
                    frag_atoms = []
                    distance = distance_matrix(coord,coord)

                    if 1 < sz < 6:
                        select_num = np.random.choice(range(1, sz), 1, replace=False)[0]
                    elif sz >= 6:
                        select_num = np.random.choice(range(max(int(sz*self.args.low_bound),1), max(int(sz*self.args.high_bound),1)+1), 1, replace=False)[0]
                    else:
                        select_num = 0
                    assert select_num < sz, (select_num, sz)
                    selected_id = set()
                    selected_id.add(point_select)
                    frag_atoms.append(point_select)
                    i=0
                    distance_queue = PriorityQueue()
                    for j in range(sz):
                        distance_queue.put((distance[point_select][j],j))
                    while i <select_num:
                        _, point_select = distance_queue.get()
                        if point_select in selected_id:
                            continue
                        selected_id.add(point_select)
                        frag_atoms.append(point_select)
                        for j in range(sz):
                            distance_queue.put((distance[point_select][j],j))
                        i+=1
                    frag_selected = [list(set(frag_atoms))]
                    frag_coord = [coord[frag_selected[0]].mean(axis=0)]
                    del frag_atoms
            else:
                frag_atom = self.frag_dataset[index]
                frag_sz = len(frag_atom)
                assert frag_sz > 0
                num_mask_frag = 1
                frag_selected = 2
                frag_selected = [frag_atom[frag_selected]]
                frag_selected = [[25,40]]
                frag_coord = [coord[frag_selected[0]].mean(axis=0)]
            
            

            mask_idc = [] 
            virtual_atom_coord_flatten = []
            assert len(frag_selected)==1, "only support select one fragment"
            
            frag_index = 0
            frag_selected_atom = frag_selected[0]

            mask_idc += list(frag_selected_atom)
            mask_num = len(frag_selected_atom)

            if self.args.use_add_num > 0:
                add_num = int(np.random.uniform(-4, 6))
                virtual_neg_num = min(int(np.random.uniform(int(self.args.low_neg_bound * max(mask_num + add_num, 2)) , int(self.args.high_neg_bound * max(mask_num + add_num, 2)) )), self.args.max_atoms-(len(coord) - len(mask_idc)))
                ret["add_num"] =  max(mask_num + add_num, 2) - mask_num
            else:
                if self.args.tune_virtual > 0:
                    virtual_neg_num = int(np.random.uniform(int(self.args.low_neg_bound * mask_num) , int(self.args.high_neg_bound * mask_num) ))
                else:
                    virtual_neg_num = min(int(np.random.uniform(int(self.args.low_neg_bound * mask_num) , int(self.args.high_neg_bound * mask_num) )), self.args.max_atoms-(len(coord) - len(mask_idc)))
                
                ret["add_num"] = 0
            
            if self.args.rotation > 0:
                rotations = R.random(5).as_matrix()
                volume = (np.max(coord[:,0]) - np.min(coord[:,0])) * (np.max(coord[:,1]) - np.min(coord[:,1])) * (np.max(coord[:,2]) - np.min(coord[:,2]))
                best_matrix = None
                for rotation in rotations:
                    matrix = rotation
                    coord_r = coord.dot(matrix)
                    volume_r =  (np.max(coord_r[:,0]) - np.min(coord_r[:,0])) * (np.max(coord_r[:,1]) - np.min(coord_r[:,1])) * (np.max(coord_r[:,2]) - np.min(coord_r[:,2]))
                    if volume_r < volume:
                        best_matrix = matrix
                        volume = volume_r
                if best_matrix is not None:
                    coord = coord.dot(best_matrix)

            if self.args.full_gravity > 0:
                delta_coord = coord
            else:
                delta_coord = coord[frag_selected_atom]
            
            noise_scale_large = np.random.uniform(low=1, high=2, size=(1))

            grid_coords = []
            if self.args.cos_mask <= 0:
                if self.args.full_gravity > 0:
                    scale = 1.5
                else:
                    scale = 1.5
                delta_x = np.max(delta_coord[:,0]) + scale
                delta_y = np.max(delta_coord[:,1]) + scale
                delta_z = np.max(delta_coord[:,2]) + scale

                delta_x_min = np.min(delta_coord[:,0]) - scale
                delta_y_min = np.min(delta_coord[:,1]) - scale
                delta_z_min = np.min(delta_coord[:,2]) - scale

                n_x = np.random.uniform(low=delta_x_min , high=delta_x , size=(virtual_neg_num, 1))
                n_y = np.random.uniform(low=delta_y_min , high=delta_y , size=(virtual_neg_num, 1))
                n_z = np.random.uniform(low=delta_z_min , high=delta_z , size=(virtual_neg_num, 1))
                noise = np.concatenate((n_x, n_y, n_z), axis=1)    
                noise_atom_coord = noise                          
            else:
                
                if self.args.cubic > 0.0:
                    delta_x = np.max(delta_coord[:,0]) + 2
                    delta_y = np.max(delta_coord[:,1]) + 2
                    delta_z = np.max(delta_coord[:,2]) + 2

                    delta_x_min = np.min(delta_coord[:,0]) - 2
                    delta_y_min = np.min(delta_coord[:,1]) - 2
                    delta_z_min = np.min(delta_coord[:,2]) - 2

                    n_x = np.random.uniform(low=delta_x_min , high=delta_x , size=(virtual_neg_num, 1))
                    n_y = np.random.uniform(low=delta_y_min , high=delta_y , size=(virtual_neg_num, 1))
                    n_z = np.random.uniform(low=delta_z_min , high=delta_z , size=(virtual_neg_num, 1))
                    noise = np.concatenate((n_x, n_y, n_z), axis=1)    
                    noise_atom_coord = noise     


                    if self.args.grid_vis:
                        assert len(noise_atom_coord) == virtual_neg_num, f"Generated {len(noise_atom_coord)} points, expected {virtual_neg_num}."

                        noise_atom_coord = np.array(noise_atom_coord)                        
                        points_per_edge = 5  
                        big_cube_edges = generate_cube_edges(delta_x_min, delta_x, delta_y_min, delta_y, delta_z_min, delta_z, points_per_edge)
                        grid_coords = np.array(big_cube_edges)

                        ret['grid_coords'] = torch.from_numpy(grid_coords).float()  
                        ret['grid_noise_atom_coords'] = torch.from_numpy(noise_atom_coord).float()  
                                                           
                else:
                    grid_size = self.args.grid_size 
                    
                    noise_atom_coords = []
                    total_points = virtual_neg_num 
                    current_index = 0

                    base_points_per_atom = virtual_neg_num // len(delta_coord)
                    extra_points = virtual_neg_num % len(delta_coord)

                    atom_indices = list(range(len(delta_coord)))

                    extra_points_indices = np.random.choice(atom_indices, extra_points, replace=False)
                    grid_coords = []
                    for idx, atom_coord in enumerate(delta_coord):
                        grid_offset = np.random.uniform(-self.args.grid_offset_size, self.args.grid_offset_size, size=(3,))
                        x_min, y_min, z_min = atom_coord - grid_size / 2 + grid_offset
                        x_max, y_max, z_max = atom_coord + grid_size / 2 + grid_offset

                        actual_points_per_atom = base_points_per_atom
                        if idx in extra_points_indices:
                            actual_points_per_atom += 1

                        for _ in range(actual_points_per_atom):
                            point_offset = np.random.uniform(-0.5, 0.5, size=(3,))
                            n_x = np.random.uniform(low=x_min, high=x_max) + point_offset[0]
                            n_y = np.random.uniform(low=y_min, high=y_max) + point_offset[1]
                            n_z = np.random.uniform(low=z_min, high=z_max) + point_offset[2]
                            noise_atom_coords.append([n_x, n_y, n_z])

                            if self.args.grid_vis:
           
                                points_per_edge = 5  
                                cube_edges = generate_cube_edges(x_min, x_max, y_min, y_max, z_min, z_max, points_per_edge)
                                grid_coords.extend(cube_edges)




                    assert len(noise_atom_coords) == virtual_neg_num, f"Generated {len(noise_atom_coords)} points, expected {virtual_neg_num}."

                    noise_atom_coord = np.array(noise_atom_coords)

                    if self.args.grid_vis:
                        grid_coords = np.array(grid_coords)


                        ret['grid_coords'] = torch.from_numpy(grid_coords).float()
                        ret['grid_noise_atom_coords'] = torch.from_numpy(noise_atom_coord).float()


            if self.args.rotation > 0:
                if best_matrix is not None:
                    noise_atom_coord  = noise_atom_coord.dot(best_matrix.transpose(-1,-2))
                    coord = coord.dot(best_matrix.transpose(-1,-2))

            if self.mask_prob == 1:
                ret['centroid_dataset'] = torch.from_numpy(noise_atom_coord.mean(axis=0)).float() 
            else:
                frag_not_selected_atom = [_ for _ in range(len(item)) if _ not in frag_selected_atom]
                ret['centroid_dataset'] = torch.from_numpy((coord[frag_not_selected_atom].sum(axis=0) * (virtual_neg_num // mask_num) + noise_atom_coord.sum(axis=0))/ (virtual_neg_num + len(frag_not_selected_atom) * (virtual_neg_num // mask_num)) ).float()
            ret['centroid_label_dataset'] = torch.from_numpy(coord.mean(axis=0)).float() 


            assert len(noise_atom_coord) == virtual_neg_num
            virtual_atom_coord_flatten.extend(noise_atom_coord.tolist())

            virtual_atom_coord_flatten = np.array(virtual_atom_coord_flatten)
            no_mask = np.full(sz, True)
            no_mask[mask_idc] = False

            new_sz = len(item) + virtual_neg_num - len(mask_idc)
            random_idx = np.array(range(new_sz))
            mask_null_idx = random_idx[:virtual_neg_num]
            no_mask_idx = random_idx[virtual_neg_num:]

            new_atom = np.full(new_sz, self.mask_idx)
            new_tgt = np.full(new_sz, self.pad_idx)
            new_tgt_null = np.full(new_sz, self.pad_idx)
            inp_coord = np.zeros((new_sz, 3), dtype=np.float32)
            out_coord = np.zeros((new_sz, 3), dtype=np.float32)
                        
            new_atom[no_mask_idx] = np.copy(item[no_mask]) 
            inp_coord[no_mask_idx, :] = np.copy(coord[no_mask]) 
            inp_coord[mask_null_idx, :] = np.copy(virtual_atom_coord_flatten) 
            
            new_coord = np.zeros((sz, 3), dtype=np.float32)
            new_coord[:len(mask_idc)] = coord[mask_idc] 
            new_coord[len(mask_idc):] = coord[no_mask] 

            new_atom2 = np.full(sz, self.pad_idx)
            new_atom2[:len(mask_idc)] = item[mask_idc] 
            new_atom2[len(mask_idc):] = item[no_mask]

            ret['atoms'] = torch.from_numpy(new_atom).long() 
            ret['coordinates'] = torch.from_numpy(inp_coord).float() 
            ret['unmask_coord'] = torch.from_numpy(coord[mask_idc]).float() 
            ret['unmask_atom'] = item[mask_idc]
            ret['unmask_index'] = torch.ones(len(mask_idc)).long()

            
            ret['all_coord'] = torch.from_numpy(new_coord).float()
            ret['all_atom'] = torch.from_numpy(new_atom2).long()
            ret['all_index'] = torch.ones(sz).long() # 
            
            ret['virtual_index'] = torch.ones(virtual_neg_num).long() # 
            

            if self.args.pre_set_label > 0 :
                
                if self.args.full_gravity > 0:
                    no_mask_idx = set([_ for _ in range(len(item)) if _ not in mask_idc ])
                    mask_idc = np.arange(len(item))
                
                noise_t_add = np.random.randn(len(mask_null_idx), 3) * 0.05 
                distance_null_to_mask = distance_matrix(inp_coord[mask_null_idx]+noise_t_add,coord[mask_idc]) 

                if self.args.sample_greedy==1:
                    mask_null_idx_tgt_init = [0 for _ in range(len(distance_null_to_mask))]
                    select_num = np.random.choice(range(len(distance_null_to_mask) ), 1, replace=False )[0]
                    traverse_list = list(np.arange(select_num,len(distance_null_to_mask) ) )+ list(np.arange(select_num))
                    for i in traverse_list:
                        tgt_index = np.argmin(distance_null_to_mask[i])
                        assert distance_null_to_mask[i][tgt_index]!=100, distance_null_to_mask
                        distance_null_to_mask[:,tgt_index] = 100
                        mask_null_idx_tgt_init[i] = tgt_index
                    mask_null_idx_tgt_init = torch.tensor(np.array(mask_null_idx_tgt_init))
                    mask_null_idx_tgt = [mask_idc[i] for i in mask_null_idx_tgt_init.reshape(-1).numpy()]
                    new_tgt_null_sample = item[mask_null_idx_tgt]
                    mask_null_idx_tgt_prob = torch.ones_like(mask_null_idx_tgt_init).unsqueeze(1)
                else:
                    seed = int(hash((self.seed, epoch, index)) % 1e8)
                    torch.manual_seed(seed)
                    mask_null_idx_tgt_init = torch.argmin(torch.from_numpy(distance_null_to_mask), dim=-1).unsqueeze(-1) # 选定距离中离最近的mask点当做target
                    
                    mask_null_idx_tgt = [mask_idc[i] for i in mask_null_idx_tgt_init.reshape(-1).numpy()]
                    new_tgt_null_sample = item[mask_null_idx_tgt].reshape(-1)

                real_pos_hit = 0
                real_pos_label_hit = 0
                virtual_atom_hit_pre = 0
                if self.args.full_gravity > 0:
                    mask_null_idx_tgt_init_list = mask_null_idx_tgt_init.reshape(-1).numpy().tolist()
                    for item_i in range(len(new_tgt_null_sample)):
                        if new_tgt_null_sample[item_i]!=self.null_idx and mask_null_idx_tgt_init_list[item_i] not in no_mask_idx:
                            virtual_atom_hit_pre +=1
                    real_pos_label_hit = len(set([mask_null_idx_tgt_init_list[i] for i in range(len(mask_null_idx_tgt_init_list)) if mask_null_idx_tgt_init_list[i] not in no_mask_idx ]))
                    real_pos_hit = len(set([ mask_null_idx_tgt_init_list[i] for i in range(len(mask_null_idx_tgt_init_list)) if mask_null_idx_tgt_init_list[i] not in no_mask_idx]))
                else:
                    mask_null_idx_tgt_init_list = mask_null_idx_tgt_init.reshape(-1).numpy().tolist()
                    for item_i in range(len(new_tgt_null_sample)):
                        if new_tgt_null_sample[item_i]!=self.null_idx:
                            virtual_atom_hit_pre +=1
                    real_pos_label_hit = len(set([mask_null_idx_tgt_init_list[i] for i in range(len(mask_null_idx_tgt_init_list)) ]))
                    real_pos_hit = len(set([ mask_null_idx_tgt_init_list[i] for i in range(len(mask_null_idx_tgt_init_list))]))


                assert len(mask_null_idx_tgt) == virtual_neg_num, (len(mask_null_idx_tgt) , virtual_neg_num)

                ret['encoder_null_target'] = new_tgt_null_sample.long()
                ret['coord_null_target'] = torch.from_numpy(np.copy(coord[mask_null_idx_tgt])).float()
                ret['encoder_null_target_idx'] = mask_null_idx_tgt_init.reshape(-1).long() #这个用来check一下是不是对的上, 同时用来重新算label矩阵
                ret['coord_null_target_post'] = torch.cat((ret['coord_null_target'],torch.from_numpy(coord[no_mask]).float()),dim=0)

                merge_label_1 = mask_null_idx_tgt_init.reshape(-1)
                merge_label_3 = (merge_label_1.unsqueeze(0).unsqueeze(-1) - merge_label_1.unsqueeze(0).unsqueeze(1)).eq(0).long().squeeze(0)
                

                row_index = torch.arange(merge_label_1.size(0)) # torch.Size([242])
                index_size = torch.max(torch.sum(merge_label_3, dim=-1))
                if self.args.sample_greedy==1:
                    index = torch.zeros((ret['atoms'].size(0)+1, index_size),  dtype=torch.long) 
                else:
                    index = torch.zeros((ret['atoms'].size(0), index_size),  dtype=torch.long)
                index_weight = torch.zeros_like(index)

                coord_label = []
                atom_type_label = []

                src_tokens2 = []

                remain_ =ret['atoms'].ne(self.mask_idx)
                remain_num = torch.sum(remain_, dim=-1)
                init_row_index = torch.arange(ret['atoms'].size(0))
                pos_hit = 0
                count = 1
                count2 = 0
                new_index_size3 = 0
                mydict={}
                for row_i in range(len(merge_label_3)):
                    merge_id = merge_label_3[row_i].ne(0)
                    merge_id_num = torch.sum(merge_id)
                    if merge_id_num <=0:
                        continue
                    index[count][:merge_id_num] = row_index[merge_id] + 1
                    index_weight[count][:merge_id_num] = 1
                    if merge_id_num > new_index_size3:
                        new_index_size3 = merge_id_num                    

                    assert row_index[merge_id][0] == row_i
                    coord_label.append(coord[mask_null_idx_tgt][row_index[merge_id][0]].tolist())
                    atom_type_label.append(item[mask_null_idx_tgt][row_index[merge_id][0]].item())
                    src_tokens2.append(self.mask_idx)

                    assert item[mask_null_idx_tgt][row_index[merge_id][0]].item() == item[mask_idc][merge_label_1[row_index[merge_id][0]]]
                    assert merge_label_1[row_index[merge_id][0]].item() not in mydict
                    mydict[merge_label_1[row_index[merge_id][0]].item()]=1

                    merge_label_3[:,row_index[merge_id]] = 0
                    count2+=1
                    if self.args.full_gravity > 0:
                        if mask_null_idx_tgt[row_index[merge_id][0]].item() not in no_mask_idx:
                            pos_hit+=1
                    else:
                        pos_hit+=1
                    count+=1
                
                assert count2 == len(atom_type_label)
                assert len(item[no_mask]) == remain_num
                src_tokens2.extend(item[no_mask].numpy().tolist())
                
                assert len(init_row_index) == len(remain_), (init_row_index, remain_)
                index[count:count+remain_num,0] = init_row_index[remain_] + 1 
                index_weight[count:count+remain_num,0] = 1

                assert count+remain_num == len(src_tokens2) + 1
                
                index =  index[:len(src_tokens2) + 1,:new_index_size3]  
                index_weight = index_weight[:len(src_tokens2) + 1,:new_index_size3]

                ret['merge_idx'] = index
                ret['merge_weight'] = index_weight
                ret['encoder_null_target2'] = torch.from_numpy(np.array(atom_type_label)).long()
                ret['coord_null_target2'] = torch.from_numpy(np.array(coord_label)).float()
                ret['pos_hit'] = pos_hit

                ret['virtual_index2'] = torch.ones(len(ret['encoder_null_target2'])).long()
                ret['src_tokens2'] = torch.from_numpy(np.array(src_tokens2)).long()

                ret['real_pos_hit'] = real_pos_hit
                ret['real_pos_label_hit'] = real_pos_label_hit
                ret['virtual_atom_hit_pre'] = virtual_atom_hit_pre
            assert len(ret['atoms']) == len(ret['coordinates']), (len(ret['atoms']) ,len(ret['coordinates']))
            assert len(ret['coordinates']) ==len(ret['coord_null_target_post']), (len(ret['coordinates']), len(ret['coord_null_target_post']))
            assert len(ret['atoms'])!=0 , ('target',)
            assert len(ret['coordinates'])!=0, ('coordinates',)
            del index, index_weight, grid_coords

            return ret