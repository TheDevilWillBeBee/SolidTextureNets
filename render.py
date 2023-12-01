import numpy as np
import torch
import kaolin as kal


def get_camera_from_view2(elev, azim, r=3.0):
    x = r * torch.cos(elev) * torch.cos(azim)
    y = r * torch.sin(elev)
    z = r * torch.cos(elev) * torch.sin(azim)
    # print(elev,azim,x,y,z)

    pos = torch.tensor([x, y, z]).unsqueeze(0)
    look_at = -pos
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
        
    def get_projected_coordinates(self, verts, faces, num_views=8, std=8, center_elev=0, center_azim=0, radius = 1.5):
        device = self.device
        # Front view with small perturbations in viewing angle
        # verts = mesh.vertices
        # faces = mesh.faces
        vertices = verts.float()
        faces = faces.long()
        n_faces = faces.shape[0]
        elev = torch.cat((torch.tensor([center_elev], device=device), torch.zeros(num_views - 1, device=device) + center_elev))
        azim = torch.cat((torch.tensor([center_azim], device=device), torch.linspace(1.57, 6.28, num_views, device=device) + center_azim))


        coordinates = []
        masks = []
        vertices_features = verts
        verts_features = vertices_features.unsqueeze(2).expand(-1, -1, faces.shape[-1], -1)
        indices = faces[None, ..., None].expand(vertices_features.shape[0], -1, -1, vertices_features.shape[-1])
        face_attr = torch.gather(input=verts_features, index=indices, dim=1)
        
        face_attr = [
                face_attr,
                torch.ones((1, n_faces, 3, 1), device=device)
            ]
        for i in range(num_views):
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=radius).to(device)
            
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                vertices.to(device), faces.to(device), self.camera_projection,
                camera_transform=camera_transform)
            coords, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attr, face_normals[:, :, -1])

            coords, mask = coords
            coordinates.append(coords)
            masks.append(mask)
            
        return torch.cat(coordinates, dim=0), torch.cat(masks, dim=0)


    
    def render_fixed_views(self, verts, faces, vertices_features, num_views=8, std=8, center_elev=0, center_azim=0, lighting=True,
                       background=None, radius = 1.5):
        device = self.device
        # Front view with small perturbations in viewing angle
        # verts = mesh.vertices
        # faces = mesh.faces
        vertices = verts.float()
        faces = faces.long()
        n_faces = faces.shape[0]
        elev = torch.cat((torch.tensor([center_elev], device=device), torch.zeros(num_views - 1, device=device) + center_elev))
        azim = torch.cat((torch.tensor([center_azim], device=device), torch.linspace(1.57, 6.28, num_views, device=device) + center_azim))


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
            camera_transform = get_camera_from_view2(elev[i], azim[i], r=radius).to(device)
            
            face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
                vertices.to(device), faces.to(device), self.camera_projection,
                camera_transform=camera_transform)
            image_features, soft_mask, face_idx = kal.render.mesh.dibr_rasterization(
                self.dim[1], self.dim[0], face_vertices_camera[:, :, :, -1],
                face_vertices_image, face_attributes, face_normals[:, :, -1])
            masks.append(soft_mask)

            if background is not None:
                image_features, mask = image_features

            image = torch.clip(image_features, 0.0, 1.0)

            if lighting:
                image_normals = face_normals[:, face_idx].squeeze(0)
                image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, self.lights).unsqueeze(0)
                image = image * image_lighting.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).to(device)
                image = torch.clip(image, 0.0, 1.0)

            if background is not None:
                background_mask = torch.zeros(image.shape).to(device)
                mask = mask.squeeze(-1)
                background_mask[torch.where(mask == 0.0)] = background
                
                
                image = torch.clip(image + background_mask, 0., 1.)
            images.append(image)

        images = torch.cat(images, dim=0).permute(0, 3, 1, 2)
        masks = torch.cat(masks, dim=0)

        return images

