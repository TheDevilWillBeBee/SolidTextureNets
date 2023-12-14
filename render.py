import numpy as np
import torch
import kaolin as kal


def get_camera_from_view2(elev, azim, r=3.0, look_at=None):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    if look_at is None:
        look_at = torch.tensor([0, 0, 0]).unsqueeze(0)

        # look_at = torch.tensor([0, -0.1, 0.15]).unsqueeze(0) # only for cube
    else:
        look_at = torch.tensor(look_at).unsqueeze(0)
        look_at = -pos - look_at

    
    
    direction = torch.tensor([0., 1., 0.]).unsqueeze(0)
    direction = direction / direction.norm()
    # direction = -pos/pos.norm()

    camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, direction)
    return camera_proj




class Renderer:
    def __init__(self, device, 
                 lights=torch.tensor([1.0, 1.0,1.0,1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                 camera=None,
                 dim=(224, 224)):

        self.device = device
        if camera is None:
            camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)
        # Third-order SH lighting. First parameter: ambient lighting. 2-4: diffuse lighting from x,y,z. last 5: specular
        self.lights = lights.unsqueeze(0).to(device)
        self.camera_projection = camera
        self.dim = dim
        
    def get_projected_coordinates(self, verts, faces, num_views=8, std=8, center_elev=0, center_azim=0, radius = 1.5, look_at=None):
        device = self.device
        # Front view with small perturbations in viewing angle
        # verts = mesh.vertices
        # faces = mesh.faces
        vertices = verts.float()
        faces = faces.long()
        n_faces = faces.shape[0]
        elev = torch.cat((torch.tensor([center_elev], device=device), torch.zeros(num_views - 1, device=device) + center_elev))
        azim = torch.cat((torch.tensor([center_azim], device=device), torch.linspace(1.57, 6.28, num_views, device=device) + center_azim))


        dilation = Dilation2d(1, 1, soft_max=False).to(self.device)
        erosion = Erosion2d(1, 1, soft_max=False).to(self.device)
        
        
        coordinates = []
        masks = []
        lightings = []
        vertices_features = verts
        verts_features = vertices_features.unsqueeze(2).expand(-1, -1, faces.shape[-1], -1)
        indices = faces[None, ..., None].expand(vertices_features.shape[0], -1, -1, vertices_features.shape[-1])
        face_attr = torch.gather(input=verts_features, index=indices, dim=1)
        
        face_attr = [
                face_attr,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=radius, look_at=look_at).to(device)
            
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                vertices.to(device), faces.to(device), self.camera_projection,
                camera_transform=camera_transform)
            coords, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1], 
                face_vertices_image, face_attr, face_normals[:, :, -1], )

            coords, mask = coords
            coordinates.append(coords)
            mask = erosion(dilation(mask.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
            
            masks.append(mask)
            
            image_normals = face_normals[:, face_idx].squeeze(0)
            image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
            lightings.append(image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1))
            
        return torch.cat(coordinates, dim=0), torch.cat(masks, dim=0), torch.cat(lightings, dim=0)


    
    def render_fixed_views(self, verts, faces, vertices_features, num_views=8, std=8, center_elev=0, center_azim=0, lighting=True,
                       background=None, radius = 1.5, look_at=None):
        device = self.device
        # Front view with small perturbations in viewing angle
        # verts = mesh.vertices
        # faces = mesh.faces
        vertices = verts.float()
        faces = faces.long()
        n_faces = faces.shape[0]
        elev = torch.cat((torch.tensor([center_elev], device=device), torch.zeros(num_views - 1, device=device) + center_elev))
        azim = torch.cat((torch.tensor([center_azim], device=device), torch.linspace(1.57, 6.28, num_views, device=device) + center_azim))

        dilation = Dilation2d(1, 1, soft_max=False).to(self.device)
        erosion = Erosion2d(1, 1, soft_max=False).to(self.device)
        
        images = []
        masks = []
        
        verts_features = vertices_features.unsqueeze(2).expand(-1, -1, faces.shape[-1], -1)
        indices = faces[None, ..., None].expand(vertices_features.shape[0], -1, -1, vertices_features.shape[-1])
        face_attr = torch.gather(input=verts_features, index=indices, dim=1)

        if background is not None:
            face_attributes = [
                face_attr,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        
        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=radius, look_at=look_at).to(device)
            
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                vertices.to(device), faces.to(device), self.camera_projection,
                camera_transform=camera_transform)
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            if background is not None:
                image_features, mask = image_features
                mask = erosion(dilation(mask.permute(0, 3, 1, 2))).permute(0, 2, 3, 1)

            image = torch.clip(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clip(image, 0.0, 1.0)

            if background is not None:
                # background_mask = torch.zeros(image.shape).to(device)
                # mask = mask.squeeze(-1)
                # background_mask[torch.where(mask == 0.0)] = background
                
                image[mask == 0] = 1.0
                # image = torch.clip(image + background_mask, 0., 1.)
                
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        return images


import math
import pdb
import torch.nn as nn
import torch.nn.functional as F


class Morphology(nn.Module):
    '''
    https://github.com/lc82111/pytorch_morphological_dilation2d_erosion2d/tree/master
    Base class for morpholigical operators 
    For now, only supports stride=1, dilation=1, kernel_size H==W, and padding='same'.
    '''
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=15, type=None):
        '''
        in_channels: scalar
        out_channels: scalar, the number of the morphological neure. 
        kernel_size: scalar, the spatial size of the morphological neure.
        soft_max: bool, using the soft max rather the torch.max(), ref: Dense Morphological Networks: An Universal Function Approximator (Mondal et al. (2019)).
        beta: scalar, used by soft_max.
        type: str, dilation2d or erosion2d.
        '''
        super(Morphology, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.soft_max = soft_max
        self.beta = beta
        self.type = type

        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size), requires_grad=False)
        self.unfold = nn.Unfold(kernel_size, dilation=1, padding=0, stride=1)

    def forward(self, x):
        '''
        x: tensor of shape (B,C,H,W)
        '''
        # padding
        x = fixed_padding(x, self.kernel_size, dilation=1)
        
        # unfold
        x = self.unfold(x)  # (B, Cin*kH*kW, L), where L is the numbers of patches
        x = x.unsqueeze(1)  # (B, 1, Cin*kH*kW, L)
        L = x.size(-1)
        L_sqrt = int(math.sqrt(L))

        # erosion
        weight = self.weight.view(self.out_channels, -1) # (Cout, Cin*kH*kW)
        weight = weight.unsqueeze(0).unsqueeze(-1)  # (1, Cout, Cin*kH*kW, 1)

        if self.type == 'erosion2d':
            x = weight - x # (B, Cout, Cin*kH*kW, L)
        elif self.type == 'dilation2d':
            x = weight + x # (B, Cout, Cin*kH*kW, L)
        else:
            raise ValueError
        
        if not self.soft_max:
            x, _ = torch.max(x, dim=2, keepdim=False) # (B, Cout, L)
        else:
            x = torch.logsumexp(x*self.beta, dim=2, keepdim=False) / self.beta # (B, Cout, L)

        if self.type == 'erosion2d':
            x = -1 * x

        # instead of fold, we use view to avoid copy
        x = x.view(-1, self.out_channels, L_sqrt, L_sqrt)  # (B, Cout, L/2, L/2)

        return x 

class Dilation2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Dilation2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'dilation2d')

class Erosion2d(Morphology):
    def __init__(self, in_channels, out_channels, kernel_size=5, soft_max=True, beta=20):
        super(Erosion2d, self).__init__(in_channels, out_channels, kernel_size, soft_max, beta, 'erosion2d')



def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs