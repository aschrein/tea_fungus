extern crate vulkano;
extern crate vulkano_shaders;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::sys::UnsafeCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::command_buffer::*;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::instance::{Instance, InstanceExtensions, PhysicalDevice};
use vulkano::pipeline::ComputePipeline;
use vulkano::sync;
use vulkano::sync::GpuFuture;

use std::sync::Arc;

use vulkano::query::*;

#[test]
fn mem_test_1_wrapper() {
    mem_test_1()
}

pub fn mem_test_1() {
    let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_compute())
        .unwrap();
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &DeviceExtensions::none(),
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();
    let data_buffer = {
        let data_iter = (0..65536u32).map(|n| n);
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), data_iter).unwrap()
    };
    let pipeline = Arc::new({
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: "
                    #version 450

                    layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

                    layout(set = 0, binding = 0) buffer Data {
                        uint data[];
                    } data;

                    void main() {
                        uint idx = gl_GlobalInvocationID.x;
                        data.data[idx] *= 12;
                    }"
            }
        }
        let shader = cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });
    let command_buffer =
        AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
            .unwrap()
            .dispatch(
                [64, 1, 1],
                pipeline.clone(),
                Arc::new(
                    PersistentDescriptorSet::start(pipeline.clone(), 0)
                        .add_buffer(data_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                )
                .clone(),
                (),
            )
            .unwrap()
            .build()
            .unwrap();
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
    let data_buffer_content = data_buffer.read().unwrap();
    for n in 0..65536u32 {
        assert_eq!(data_buffer_content[n as usize], n * 12);
    }
    println!("success");
}

#[test]
fn mem_test_2_wrapper() {
    mem_test_2();
}

pub fn mem_test_2() {
    let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_compute())
        .unwrap();
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &DeviceExtensions::none(),
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();

    let pipeline = Arc::new({
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: "
                    #version 450

                    layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

                    layout(set = 0, binding = 0) buffer Data {
                        uint data[];
                    } data;
                    layout(set = 0, binding = 1) buffer DynConstants {
                        uint counter;
                        uint payload_size;
                    } dc;
                    void main() {
                        while(true) {
                            uint id = atomicAdd(dc.counter, 64u);
                            uint limit = min(dc.payload_size, id + 64u);
                            for (; id < limit; id++)
                                data.data[id] *= 12;
                            if (limit >= dc.payload_size)
                                break;
                        }
                    }"
            }
        }
        let shader = cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });
    let data_buffer = {
        let data_iter = (0..65536u32).map(|n| n);
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), data_iter).unwrap()
    };
    let dynconst_buffer = {
        let data_iter = vec![0u32, 65536u32];
        CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            data_iter.iter().cloned(),
        )
        .unwrap()
    };
    let set = Arc::new(
        PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_buffer(data_buffer.clone())
            .unwrap()
            .add_buffer(dynconst_buffer.clone())
            .unwrap()
            .build()
            .unwrap(),
    );
    let query_pool = UnsafeQueryPool::new(device.clone(), QueryType::Timestamp, 256).unwrap();
    // let command_buffer = unsafe {
    //     // let pool = Device::standard_command_pool(&device, queue_family);
    //     // let command_builder = UnsafeCommandBufferBuilder::new(
    //     // &pool, Kind::Primary, Flags::OneTimeSubmit).unwrap();

    //     // command_builder.begin_query(query_pool.query(0).unwrap(), true);
    //     use sync::PipelineStages;
    //     let command_buffer =
    //         AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
    //             .unwrap()
    //             .dispatch([64, 1, 1], pipeline.clone(), set.clone(), ())
    //             .unwrap()
    //             .build()
    //             .unwrap();
    //     command_buffer.inner().write_timestamp(
    //         query_pool.query(0),
    //         PipelineStages {
    //             ..PipelineStages::none()
    //         },
    //     );
    //     command_buffer
    // };
    let command_buffer =
        AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
            .unwrap()
            .dispatch([1, 1, 1], pipeline.clone(), set.clone(), ())
            .unwrap()
            .build()
            .unwrap();
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
    let data_buffer_content = data_buffer.read().unwrap();
    for n in 0..65536u32 {
        assert_eq!(data_buffer_content[n as usize], n * 12);
    }
    println!("success");
}

pub fn mem_test_3() {
    let instance = Instance::new(None, &InstanceExtensions::none(), None).unwrap();
    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_compute())
        .unwrap();
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &DeviceExtensions::none(),
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();
    let queue = queues.next().unwrap();
    let data_buffer = {
        let data_iter = (0..65536u32).map(|n| n);
        CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), data_iter).unwrap()
    };
    let pipeline = Arc::new({
        mod cs {
            vulkano_shaders::shader! {
                ty: "compute",
                src: "
                    #version 450

                    layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

                    layout(set = 0, binding = 0) buffer Data {
                        uint data[];
                    } data;

                    void main() {
                        uint idx = gl_GlobalInvocationID.x;
                        data.data[idx] *= 12;
                    }"
            }
        }
        let shader = cs::Shader::load(device.clone()).unwrap();
        ComputePipeline::new(device.clone(), &shader.main_entry_point(), &()).unwrap()
    });
    let command_buffer =
        AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family())
            .unwrap()
            .dispatch(
                [64, 1, 1],
                pipeline.clone(),
                Arc::new(
                    PersistentDescriptorSet::start(pipeline.clone(), 0)
                        .add_buffer(data_buffer.clone())
                        .unwrap()
                        .build()
                        .unwrap(),
                )
                .clone(),
                (),
            )
            .unwrap()
            .build()
            .unwrap();
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()
        .unwrap();
    future.wait(None).unwrap();
    let data_buffer_content = data_buffer.read().unwrap();
    for n in 0..65536u32 {
        assert_eq!(data_buffer_content[n as usize], n * 12);
    }
    println!("success");
}