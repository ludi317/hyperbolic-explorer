//! A custom material whose vertex shader replaces the usual Euclidean
//! view/projection with the hyperboloid-model projection. The only per-frame
//! input is the player's Lorentz view matrix.

use crate::world::ATTRIBUTE_COLOR;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::reflect::TypePath;
use bevy::render::mesh::MeshVertexBufferLayoutRef;
use bevy::render::render_resource::{
    AsBindGroup, RenderPipelineDescriptor, ShaderRef, SpecializedMeshPipelineError,
};

#[derive(Asset, TypePath, AsBindGroup, Clone)]
pub struct HyperMaterial {
    /// Lorentz transform mapping world hyperboloid points into the camera frame.
    #[uniform(0)]
    pub view: Mat4,
    /// `(f = 1/tan(fov_y/2), aspect, fog_density, unused)`.
    #[uniform(1)]
    pub params: Vec4,
    /// Background/fog color.
    #[uniform(2)]
    pub fog_color: Vec4,
}

impl Material for HyperMaterial {
    fn vertex_shader() -> ShaderRef {
        "shaders/hyper.wgsl".into()
    }

    fn fragment_shader() -> ShaderRef {
        "shaders/hyper.wgsl".into()
    }

    fn specialize(
        _pipeline: &MaterialPipeline<Self>,
        descriptor: &mut RenderPipelineDescriptor,
        layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        let vertex_layout = layout.0.get_layout(&[
            Mesh::ATTRIBUTE_POSITION.at_shader_location(0),
            ATTRIBUTE_COLOR.at_shader_location(1),
        ])?;
        descriptor.vertex.buffers = vec![vertex_layout];
        // Tiles and pillar faces should be visible from both sides.
        descriptor.primitive.cull_mode = None;
        Ok(())
    }
}
