// Copyright (c) 2016 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

extern crate cgmath;
extern crate time;
extern crate vulkano_win;
extern crate winit;

use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::attachment::AttachmentImage;
use vulkano::image::SwapchainImage;
use vulkano::instance;
use vulkano::instance::debug::{DebugCallback, MessageTypes};
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::vertex::{SingleBufferDefinition, TwoBuffersDefinition};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::GpuFuture;

use vulkano_win::VkSurfaceBuild;

use winit::Window;

use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};

use std::iter;
use std::sync::Arc;
use std::time::Instant;
use crate::state::*;
mod point_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450

        layout(location = 0) in vec3 position;
        // layout(location = 1) in vec3 normal;

        // layout(location = 0) out vec3 v_normal;

        layout(set = 0, binding = 0) uniform Data {
            mat4 world;
            mat4 view;
            mat4 proj;
        } uniforms;

        void main() {
            mat4 worldview = uniforms.view * uniforms.world;
            // v_normal = transpose(inverse(mat3(worldview))) * normal;
            gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
            gl_PointSize = 1.0/(1.0e-3 + abs(gl_Position.z)) * 100.0;
        }
        "
    }
}
mod edge_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450

        layout(location = 0) in vec3 position;

        layout(set = 0, binding = 0) uniform Data {
            mat4 world;
            mat4 view;
            mat4 proj;
        } uniforms;

        void main() {
            mat4 worldview = uniforms.view * uniforms.world;
            gl_Position = uniforms.proj * worldview * vec4(position, 1.0);
        }
        "
    }
}
mod white_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
        #version 450
        layout(location = 0) out vec4 f_color;

        void main() {
            f_color = vec4(1.0, 1.0, 1.0, 1.0);
        }
        "
    }
}

#[derive(Copy, Clone)]
pub struct Vertex {
    position: (f32, f32, f32),
}

impl_vertex!(Vertex, position);

#[derive(Copy, Clone)]
pub struct Normal {
    normal: (f32, f32, f32),
}

impl_vertex!(Normal, normal);

pub fn render_main(tick: Box<Fn()->Sim_State>) {
    let instance = {
        let mut extensions = vulkano_win::required_extensions();
        // extensions.ext_debug_report = true;

        println!("List of Vulkan debugging layers available to use:");
        let mut layers = instance::layers_list().unwrap();
        while let Some(l) = layers.next() {
            println!("\t{}", l.name());
        }
        let layers = vec![
            // "VK_LAYER_LUNARG_standard_validation",
            // "VK_LAYER_LUNARG_parameter_validation",
        ];
        Instance::new(None, &extensions, layers).expect("failed to create Vulkan instance")
    };
    // let all = MessageTypes {
    //     error: true,
    //     warning: true,
    //     performance_warning: true,
    //     information: true,
    //     debug: true,
    // };

    // let _debug_callback = DebugCallback::new(&instance, all, |msg| {
    //     let ty = if msg.ty.error {
    //         "error"
    //     } else if msg.ty.warning {
    //         "warning"
    //     } else if msg.ty.performance_warning {
    //         "performance_warning"
    //     } else if msg.ty.information {
    //         "information"
    //     } else if msg.ty.debug {
    //         "debug"
    //     } else {
    //         panic!("no-impl");
    //     };
    //     println!("{} {}: {}", msg.layer_prefix, ty, msg.description);
    // })
    // .ok();

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!(
        "Using device: {} (type: {:?})",
        physical.name(),
        physical.ty()
    );

    let mut events_loop = winit::EventsLoop::new();
    let surface = winit::WindowBuilder::new()
        .build_vk_surface(&events_loop, instance.clone())
        .unwrap();
    let window = surface.window();

    // unlike the triangle example we need to keep track of the width and height so we can calculate
    // render the teapot with the correct aspect ratio.
    let mut dimensions = if let Some(dimensions) = window.get_inner_size() {
        let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
        [dimensions.0, dimensions.1]
    } else {
        return;
    };

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let format = caps.supported_formats[0].0;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            caps.min_image_count,
            format,
            dimensions,
            1,
            usage,
            &queue,
            SurfaceTransform::Identity,
            alpha,
            PresentMode::Fifo,
            true,
            None,
        )
        .unwrap()
    };

    let uniform_buffer =
        CpuBufferPool::<point_vs::ty::Data>::new(device.clone(), BufferUsage::all());

    let point_vs = point_vs::Shader::load(device.clone()).unwrap();
    let white_fs = white_fs::Shader::load(device.clone()).unwrap();

    let render_pass = Arc::new(
        single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        )
        .unwrap(),
    );

    let mut point_pipeline: Option<Arc<GraphicsPipelineAbstract + Send + Sync>> = None;
    let mut line_pipeline: Option<Arc<GraphicsPipelineAbstract + Send + Sync>> = None;
    let mut framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>> = Vec::new();

    let mut recreate_swapchain = false;

    let mut previous_frame = Box::new(sync::now(device.clone())) as Box<GpuFuture>;
    let rotation_start = Instant::now();
    let mut phi = 0.0;
    let mut theta = std::f32::consts::FRAC_PI_2;
    let (mut old_mx, mut old_my): (f64, f64) = (0.0, 0.0);
    loop {
        previous_frame.cleanup_finished();

        if recreate_swapchain || point_pipeline.is_none() || line_pipeline.is_none() {
            dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) =
                    dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                std::panic!();
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err),
            };
            swapchain = new_swapchain;

            let depth_buffer =
                AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();

            framebuffers = new_images
                .iter()
                .map(|image| {
                    Arc::new(
                        Framebuffer::start(render_pass.clone())
                            .add(image.clone())
                            .unwrap()
                            .add(depth_buffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    ) as Arc<FramebufferAbstract + Send + Sync>
                })
                .collect::<Vec<_>>();

            point_pipeline = Some(Arc::new(
                GraphicsPipeline::start()
                    .vertex_input(SingleBufferDefinition::<Vertex>::new())
                    .vertex_shader(point_vs.main_entry_point(), ())
                    .point_list()
                    .viewports_dynamic_scissors_irrelevant(1)
                    .viewports(iter::once(Viewport {
                        origin: [0.0, 0.0],
                        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                        depth_range: 0.0..1.0,
                    }))
                    .fragment_shader(white_fs.main_entry_point(), ())
                    .depth_stencil_simple_depth()
                    .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                    .build(device.clone())
                    .unwrap(),
            ));

            line_pipeline = Some(Arc::new(
                GraphicsPipeline::start()
                    .vertex_input(SingleBufferDefinition::<Vertex>::new())
                    .vertex_shader(point_vs.main_entry_point(), ())
                    .line_list()
                    .viewports_dynamic_scissors_irrelevant(1)
                    .viewports(iter::once(Viewport {
                        origin: [0.0, 0.0],
                        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                        depth_range: 0.0..1.0,
                    }))
                    .fragment_shader(white_fs.main_entry_point(), ())
                    .depth_stencil_simple_depth()
                    .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                    .build(device.clone())
                    .unwrap(),
            ));

            recreate_swapchain = false;
        }

        let uniform_buffer_subbuffer = {
            let elapsed = rotation_start.elapsed();
            // let rotation =
            //     elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
            let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
            let proj =
                cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
            let view = Matrix4::look_at(
                Point3::new(
                    f32::sin(theta) * f32::cos(phi),
                    f32::sin(theta) * f32::sin(phi),
                    f32::cos(theta),
                ) * 20.0,
                Point3::new(0.0, 0.0, 0.0),
                Vector3::new(0.0, 0.0, -1.0),
            );
            let scale: Matrix4<f32> = Matrix4::from_scale(1.0)
            //* Matrix4::from_angle_x(Rad(std::f64::consts::PI as f32 / 2.0))
            //* Matrix4::from_angle_z(Rad(std::f64::consts::PI as f32))
            ;

            let uniform_data = point_vs::ty::Data {
                world: scale.into(),
                view: view.into(),
                proj: proj.into(),
            };

            uniform_buffer.next(uniform_data).unwrap()
        };

        let (image_num, acquire_future) =
            match swapchain::acquire_next_image(swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => panic!("{:?}", err),
            };

        let sim_state = tick();
        let vertices = sim_state.pos.iter().cloned();
        let vertex_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), vertices).unwrap();
        let mut edges: Vec<vec3> = Vec::new();
        for edge in sim_state.links {
            edges.push(sim_state.pos[edge.0 as usize]);
            edges.push(sim_state.pos[edge.1 as usize]);
        }
        let edges = edges.iter().cloned();
        let edges_buffer =
            CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), edges).unwrap();

        let command_buffer =
            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap()
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    false,
                    vec![[0.0, 0.0, 1.0, 1.0].into(), 1f32.into()],
                )
                .unwrap()
                .draw(
                    point_pipeline.clone().unwrap(),
                    &DynamicState::none(),
                    vec![vertex_buffer.clone()],
                    Arc::new(
                        PersistentDescriptorSet::start(line_pipeline.clone().unwrap(), 0)
                            .add_buffer(uniform_buffer_subbuffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    ),
                    (),
                )
                .unwrap()
                .draw(
                    line_pipeline.clone().unwrap(),
                    &DynamicState::none(),
                    vec![edges_buffer.clone()],
                    Arc::new(
                        PersistentDescriptorSet::start(line_pipeline.clone().unwrap(), 0)
                            .add_buffer(uniform_buffer_subbuffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    ),
                    (),
                )
                .unwrap()
                .end_render_pass()
                .unwrap()
                .build()
                .unwrap();

        let future = previous_frame
            .join(acquire_future)
            .then_execute(queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                previous_frame = Box::new(future) as Box<_>;
            }
            Err(sync::FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false;
        events_loop.poll_events(|ev| match ev {
            winit::Event::WindowEvent {
                event: winit::WindowEvent::CloseRequested,
                ..
            } => done = true,
            winit::Event::WindowEvent {
                event:
                    winit::WindowEvent::MouseInput {
                        device_id,
                        state,
                        button,
                        modifiers,
                    },
                ..
            } => done = true,
            winit::Event::WindowEvent {
                event:
                    winit::WindowEvent::CursorMoved {
                        device_id,
                        position,
                        modifiers,
                    },
                ..
            } => {
                if (old_mx, old_my) == (0.0, 0.0) {
                } else {
                    let dx = position.x - old_mx;
                    let dy = position.y - old_my;
                    phi -= (dx as f32) * 0.01;
                    theta += (dy as f32) * 0.01;
                    let eps = 1.0e-4;
                    phi = if phi > std::f32::consts::PI * 2.0 {
                        phi - std::f32::consts::PI * 2.0
                    } else if phi < 0.0 {
                        phi + std::f32::consts::PI * 2.0
                    } else {
                        phi
                    };
                    theta = if theta > std::f32::consts::PI - eps {
                        std::f32::consts::PI - eps
                    } else if theta < eps {
                        eps
                    } else {
                        theta
                    };
                }
                old_mx = position.x;
                old_my = position.y;
            }
            winit::Event::WindowEvent {
                event: winit::WindowEvent::Resized(_),
                ..
            } => recreate_swapchain = true,
            _ => (),
        });
        if done {
            return;
        }
    }
}
