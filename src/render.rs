extern crate cgmath;
extern crate time;
extern crate vulkano_shaders;
extern crate vulkano_win;
extern crate winit;

use vulkano::buffer::cpu_pool::CpuBufferPool;
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::sampler::{Filter, MipmapMode, Sampler, SamplerAddressMode};

use vulkano::image::attachment::AttachmentImage;
use vulkano::image::*;
use vulkano::image::{Dimensions, StorageImage, SwapchainImage};

use std::ffi::CStr;
use std::fs::File;
use std::io::Read;
use vulkano::instance;
use vulkano::instance::debug::{DebugCallback, MessageTypes};
use vulkano::instance::Instance;
use vulkano::instance::InstanceExtensions;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::shader::{
    EntryPointAbstract, GraphicsShaderType, ShaderInterfaceDef, ShaderInterfaceDefEntry,
    ShaderModule,
};
use vulkano::pipeline::vertex::{SingleBufferDefinition, TwoBuffersDefinition};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{ComputePipeline, GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain;
use vulkano::swapchain::{
    AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError,
};
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano_win::VkSurfaceBuild;
use winit::Window;

use cgmath::{Matrix, Matrix3, Matrix4, Point3, Rad, Vector3};

use crate::state::*;
use std::iter;
use std::sync::Arc;
use std::time::Instant;
mod point_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450

        layout(location = 0) in vec3 position;
        // layout(location = 1) in vec3 normal;

        layout(location = 0) out float v_depth;

        layout(set = 0, binding = 0) uniform Data {
            mat4 world;
            mat4 view;
            mat4 proj;
        } uniforms;

        void main() {
            mat4 worldview = uniforms.view * uniforms.world;
            vec4 wpos = worldview * vec4(position, 1.0);
            v_depth = wpos.z;
            // v_normal = transpose(inverse(mat3(worldview))) * normal;
            gl_Position = uniforms.proj * wpos;
            gl_PointSize = 1.0/(1.0e-3 + abs(gl_Position.z))
            * 40.0;
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
        layout(location = 0) out float v_depth;
        layout(set = 0, binding = 0) uniform Data {
            mat4 world;
            mat4 view;
            mat4 proj;
        } uniforms;

        void main() {
             mat4 worldview = uniforms.view * uniforms.world;
            vec4 wpos = worldview * vec4(position, 1.0);
            v_depth = wpos.z;
            gl_Position = uniforms.proj * wpos;
            gl_PointSize = 1.0/(1.0e-3 + abs(gl_Position.z))
            * 60.0;
        }
        "
    }
}
mod fs_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
        #version 450
        layout(location = 0) in vec3 position;
        layout(location = 0) out vec2 tex_coords;
        void main() {
            float x = -1.0 + float((gl_VertexIndex & 1) << 2);
            float y = -1.0 + float((gl_VertexIndex & 2) << 1);
            tex_coords = vec2(x * 0.5 + 0.5, y * 0.5 + 0.5);
            gl_Position = vec4(x, y, 0, 1);
        }
        "
    }
}
mod fs_ps {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
        #version 450
        layout(set = 0, binding = 0) uniform sampler2D tex;
        layout(location = 0) in vec2 tex_coords;
        layout(location = 0) out vec4 f_color;
        void main() {
            f_color = texture(tex, tex_coords);
        }
        "
    }
}
mod fs_cs {
    vulkano_shaders::shader! {
        ty: "compute",
        src: "
        #version 450
        layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
        layout (set = 0, binding = 0, rgba8) uniform writeonly image2D resultImage;
        #extension GL_KHR_shader_subgroup_vote: enable
        #extension GL_KHR_shader_subgroup_shuffle: enable

        layout (set = 0, binding = 1, std140) uniform UBO 
        {
            vec3 camera_pos;
            vec3 camera_look;
            vec3 camera_up;
            vec3 camera_right;
            float camera_fov;
            float ug_size;
            uint ug_bins_count;
        } g_ubo;
        layout(set = 0, binding = 0) buffer Bins {
            uint data[];
        } g_bins;
        layout(set = 0, binding = 0) buffer Particles {
            vec3 data[];
        } g_particles;

        void main() {
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
        layout(location = 0) in float v_depth;
        void main() {
            float r = clamp(-v_depth/5.0, 0.0, 1.0);
            float g = clamp(-v_depth/10.0, 0.0, 1.0);
            float b = clamp(-v_depth/20.0, 0.0, 1.0);
            r = 1.1 - pow(r, 2.0);
            g = 1.1 - pow(g, 2.0);
            b = 1.1 - pow(b, 2.0);
            f_color = vec4(r, g, b, 1.0);
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

pub fn render_main(state: &mut Sim_State, tick: Box<Fn(&mut Sim_State)>) {
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

    let cs_uniform_buffer =
        CpuBufferPool::<fs_cs::ty::UBO>::new(device.clone(), BufferUsage::all());

    let sampler = Sampler::new(
        device.clone(),
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    )
    .unwrap();

    let point_vs = point_vs::Shader::load(device.clone()).unwrap();
    let white_fs = white_fs::Shader::load(device.clone()).unwrap();
    let fs_vs = fs_vs::Shader::load(device.clone()).unwrap();
    let fs_ps = fs_ps::Shader::load(device.clone()).unwrap();
    let fs_cs = fs_cs::Shader::load(device.clone()).unwrap();
    let fs_vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        [
            Vertex {
                position: (-1.0, -1.0, 0.0),
            },
            Vertex {
                position: (2.0, -1.0, 0.0),
            },
            Vertex {
                position: (-1.0, -2.0, 0.0),
            },
        ]
        .iter()
        .cloned(),
    )
    .unwrap();
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
    let mut fs_pipeline: Option<Arc<GraphicsPipelineAbstract + Send + Sync>> = None;
    // glslangValidator --target-env vulkan1.1 -o compute.comp.spv compute.comp.glsl
    let fs_cs_proxy = {
        let mut f =
            File::open("src/shaders/compute.comp.spv").expect("src/shaders/compute.comp.spv");
        let mut v = vec![];
        f.read_to_end(&mut v).unwrap();
        unsafe { ShaderModule::new(device.clone(), &v) }.unwrap()
    };
    let mut cs_pipeline = unsafe {
        Arc::new(
            ComputePipeline::new(
                device.clone(),
                &fs_cs_proxy.compute_entry_point(
                    CStr::from_bytes_with_nul_unchecked(b"main\0"),
                    fs_cs.main_entry_point().layout().clone(),
                ),
                &(),
            )
            .unwrap(),
        )
    };
    let mut framebuffers: Vec<Arc<FramebufferAbstract + Send + Sync>> = Vec::new();
    let mut diffuse_buffer = StorageImage::with_usage(
        device.clone(),
        Dimensions::Dim2d {
            width: dimensions[0],
            height: dimensions[1],
        },
        Format::A2B10G10R10UnormPack32,
        ImageUsage {
            storage: true,
            sampled: true,
            ..ImageUsage::none()
        },
        Some(queue.family()),
    )
    .unwrap();
    let mut recreate_swapchain = false;

    let mut previous_frame = Box::new(sync::now(device.clone())) as Box<GpuFuture>;
    let rotation_start = Instant::now();
    let mut phi = 0.0;
    let mut theta = std::f32::consts::FRAC_PI_2;
    let mut camera_zoom = 10.0;
    let mut camera_moved = false;
    let (mut old_mx, mut old_my): (f64, f64) = (0.0, 0.0);

    loop {
        previous_frame.cleanup_finished();

        if recreate_swapchain || point_pipeline.is_none() {
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

            fs_pipeline = Some(Arc::new(
                GraphicsPipeline::start()
                    .vertex_input(SingleBufferDefinition::<Vertex>::new())
                    .vertex_shader(fs_vs.main_entry_point(), ())
                    .triangle_list()
                    .viewports_dynamic_scissors_irrelevant(1)
                    .viewports(iter::once(Viewport {
                        origin: [0.0, 0.0],
                        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
                        depth_range: 0.0..1.0,
                    }))
                    .fragment_shader(fs_ps.main_entry_point(), ())
                    .depth_stencil_simple_depth()
                    .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                    .build(device.clone())
                    .unwrap(),
            ));
            diffuse_buffer = StorageImage::with_usage(
                device.clone(),
                Dimensions::Dim2d {
                    width: dimensions[0],
                    height: dimensions[1],
                },
                Format::R32G32B32A32Sfloat,
                ImageUsage {
                    storage: true,
                    sampled: true,
                    ..ImageUsage::none()
                },
                Some(queue.family()),
            )
            .unwrap();
            recreate_swapchain = false;
        }
        let ug_bins_count = 128;
        let ug_size = state.params.can_radius;

        let (uniform_buffer_subbuffer, cs_uniform_buffer_subbuffer) = {
            let elapsed = rotation_start.elapsed();
            // let rotation =
            //     elapsed.as_secs() as f64 + elapsed.subsec_nanos() as f64 / 1_000_000_000.0;
            let aspect_ratio = dimensions[0] as f32 / dimensions[1] as f32;
            let proj =
                cgmath::perspective(Rad(std::f32::consts::FRAC_PI_2), aspect_ratio, 0.01, 100.0);
            let camera_pos = Point3::new(
                f32::sin(theta) * f32::cos(phi),
                f32::sin(theta) * f32::sin(phi),
                f32::cos(theta),
            ) * camera_zoom;
            let view = Matrix4::look_at(
                camera_pos,
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
            (
                
                uniform_buffer.next(uniform_data).unwrap(),
                cs_uniform_buffer
                    .next(fs_cs::ty::UBO {
                        camera_pos: camera_pos.into(),
                        camera_look: view.row(2).truncate().into(),
                        camera_up: view.row(1).truncate().into(),
                        camera_right: view.row(0).truncate().into(),
                        camera_fov: aspect_ratio,
                        ug_size: ug_size,
                        ug_bins_count: ug_bins_count,
                        _dummy0: [0, 0, 0, 0],
                        _dummy1: [0, 0, 0, 0],
                        _dummy2: [0, 0, 0, 0],
                    })
                    .unwrap(),
            )
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

        tick(state);

        let command_buffer = if camera_moved {
            let vertices = state.pos.iter().cloned();
            let vertex_buffer =
                CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), vertices)
                    .unwrap();
            let mut edges: Vec<vec3> = Vec::new();
            for edge in &state.links {
                edges.push(state.pos[edge.0 as usize]);
                edges.push(state.pos[edge.1 as usize]);
            }
            let edges = edges.iter().cloned();
            let edges_buffer =
                CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), edges).unwrap();

            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap()
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    false,
                    vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()],
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
                .unwrap()
        } else {
            // let mut ug = UG::new(ug_size, ug_bins_count);
            // for (i, &pnt) in state.pos.iter().enumerate() {
            //     ug.put(pnt, i as u32);
            // }
            // let (bins, point_ids) = ug.pack();
            // let mut points: Vec<vec3> = Vec::new();
            // for &id in &point_ids {
            //     points.push(state.pos[id as usize]);
            // }
            // let points = points.iter().cloned();
            // let points_buffer =
            //     CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), points).unwrap();
            // let bins = bins.iter().cloned();
            // let bins_buffer =
            //     CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), bins).unwrap();

            AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
                .unwrap()
                .dispatch(
                    [(dimensions[0] + 15) / 16, (dimensions[1] + 15) / 16, 1],
                    cs_pipeline.clone(),
                    Arc::new(
                        PersistentDescriptorSet::start(cs_pipeline.clone(), 0)
                            .add_image(diffuse_buffer.clone())
                            .unwrap()
                            .add_buffer(cs_uniform_buffer_subbuffer.clone())
                            .unwrap()
                            .build()
                            .unwrap(),
                    ),
                    (),
                )
                .unwrap()
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    false,
                    vec![[0.0, 0.0, 0.0, 1.0].into(), 1f32.into()],
                )
                .unwrap()
                .draw(
                    fs_pipeline.clone().unwrap(),
                    &DynamicState::none(),
                    vec![fs_vertex_buffer.clone()],
                    Arc::new(
                        PersistentDescriptorSet::start(fs_pipeline.clone().unwrap(), 0)
                            .add_sampled_image(diffuse_buffer.clone(), sampler.clone())
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
                .unwrap()
        };

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
        camera_moved = false;
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
                    winit::WindowEvent::MouseWheel {
                        device_id,
                        delta,
                        phase,
                        modifiers,
                    },
                ..
            } => match &delta {
                winit::MouseScrollDelta::LineDelta(dx, dy) => {
                    camera_zoom += dy;
                    camera_moved = true;
                }
                winit::MouseScrollDelta::PixelDelta(pd) => {
                    std::panic!();
                }
            },
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
                    phi += (dx as f32) * 0.01;
                    theta -= (dy as f32) * 0.01;
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
                    camera_moved = true;
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
