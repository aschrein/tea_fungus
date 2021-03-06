extern crate lzma;
use lzma::LzmaWriter;
extern crate bincode;
extern crate cgmath;
extern crate rand;
extern crate serde;

use bincode::{
    config, deserialize, deserialize_from, deserialize_in_place, serialize, serialized_size,
    ErrorKind, Result,
};
use cgmath::{Matrix3, Matrix4, MetricSpace, Point3, Rad, Vector3};
use serde::de::Deserializer;
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::prelude::*;
pub type vec3 = Vector3<f32>;
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct Sim_Params {
    pub rest_length: f32,
    pub spring_factor: f32,
    pub repell_factor: f32,
    pub planar_factor: f32,
    pub bulge_factor: f32,
    pub cell_radius: f32,
    pub cell_mass: f32,
    pub can_radius: f32,
}
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct Sim_State {
    pub params: Sim_Params,
    pub pos: Vec<vec3>,
    pub links: HashSet<(u32, u32)>,
}
impl Sim_State {
    pub fn new(params: Sim_Params) -> Sim_State {
        Sim_State {
            params: params,
            pos: Vec::new(),
            links: HashSet::new(),
        }
    }
    pub fn open(filename: &str) -> Sim_State {
        let mut file = File::open(filename).unwrap();
        let mut contents: Vec<u8> = Vec::new();
        file.read_to_end(&mut contents).unwrap();
        let contents = lzma::decompress(&mut contents).unwrap();
        deserialize(&contents).unwrap()
    }
    pub fn dump(&self, filename: &str) {
        let ser = serialize(self).unwrap();
        let mut file = File::create(filename).unwrap();
        let ser = lzma::compress(&ser, 6).unwrap();
        file.write_all(&ser).unwrap();
    }
}

// Uniform Grid
//  ____
// |    |}size
// |____|}size
//
pub struct UG {
    size: f32,
    bin_count: u32,
    bins: Vec<Vec<u32>>,
    bins_indices: Vec<u32>,
}

impl UG {
    pub fn new(size: f32, bin_count: u32) -> UG {
        let mut bins_indices: Vec<u32> = Vec::new();
        for i in 0..bin_count * bin_count * bin_count {
            bins_indices.push(0);
        }
        UG {
            size: size,
            bin_count: bin_count,
            bins: vec![Vec::new()],
            bins_indices: bins_indices,
        }
    }
    pub fn pack(&self) -> (Vec<u32>, Vec<u32>) {
        let mut bins: Vec<u32> = Vec::new();
        let mut points: Vec<u32> = Vec::new();
        points.push(0);
        for &bin_index in &self.bins_indices {
            if bin_index > 0 {
                let bin = &self.bins[bin_index as usize];
                bins.push(points.len() as u32);
                bins.push(bin.len() as u32);
                for &id in bin {
                    points.push(id);
                }
            } else {
                bins.push(0);
                bins.push(0);
            }
        }
        (bins, points)
    }
    pub fn put(&mut self, pos: vec3, index: u32) {
        if pos.x > self.size
            || pos.y > self.size
            || pos.z > self.size
            || pos.x < -self.size
            || pos.y < -self.size
            || pos.z < -self.size
        {
            std::panic!();
        }
        let bin_size = (2.0 * self.size) / self.bin_count as f32;
        // let idx = radius / bin_size;
        let bin_idx = ((pos.x + self.size) / bin_size) as u32;
        let bin_idy = ((pos.y + self.size) / bin_size) as u32;
        let bin_idz = ((pos.z + self.size) / bin_size) as u32;
        for dx in -1..2 {
            for dy in -1..2 {
                for dz in -1..2 {
                    let mut bin_idx = bin_idx as i32 + dx;
                    let mut bin_idy = bin_idy as i32 + dy;
                    let mut bin_idz = bin_idz as i32 + dz;
                    if bin_idx < 0
                        || bin_idy < 0
                        || bin_idz < 0
                        || bin_idx >= self.bin_count as i32
                        || bin_idy >= self.bin_count as i32
                        || bin_idz >= self.bin_count as i32
                    {
                        continue;
                    }
                    let flat_id =
                        bin_idx as u32 + bin_idy as u32 * self.bin_count + bin_idz as u32 * self.bin_count * self.bin_count;
                    let bin_id = &mut self.bins_indices[flat_id as usize];
                    if *bin_id == 0 {
                        self.bins.push(Vec::new());
                        *bin_id = self.bins.len() as u32 - 1;
                    }
                    self.bins[*bin_id as usize].push(index);
                }
            }
        }
        
    }
    pub fn fill_lines_render(&self, lines: &mut Vec<vec3>) {
        let bin_size = self.size * 2.0 / self.bin_count as f32;
        for dz in 0..self.bin_count {
            for dy in 0..self.bin_count {
                for dx in 0..self.bin_count {
                    let flat_id = dx + dy * self.bin_count + dz * self.bin_count * self.bin_count;
                    let bin_id = &self.bins_indices[flat_id as usize];
                    if *bin_id != 0 {
                        let bin_idx = bin_size * dx as f32 - self.size;
                        let bin_idy = bin_size * dy as f32 - self.size;
                        let bin_idz = bin_size * dz as f32 - self.size;
                        let iter_x = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0];
                        let iter_y = [0, 1, 1, 0, 0, 0, 0, 1, 1, 0];
                        let iter_z = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
                        for i in 0..9 {
                            lines.push(vec3 {
                                x: bin_idx + bin_size * iter_x[i] as f32,
                                y: bin_idy + bin_size * iter_y[i] as f32,
                                z: bin_idz + bin_size * iter_z[i] as f32,
                            });
                            lines.push(vec3 {
                                x: bin_idx + bin_size * iter_x[i + 1] as f32,
                                y: bin_idy + bin_size * iter_y[i + 1] as f32,
                                z: bin_idz + bin_size * iter_z[i + 1] as f32,
                            });
                        }
                        let iter_x = [0, 0, 1, 1, 1, 1,];
                        let iter_y = [1, 1, 1, 1, 0, 0,];
                        let iter_z = [0, 1, 0, 1, 0, 1,];
                        for i in 0..3 {
                            lines.push(vec3 {
                                x: bin_idx + bin_size * iter_x[i * 2] as f32,
                                y: bin_idy + bin_size * iter_y[i * 2] as f32,
                                z: bin_idz + bin_size * iter_z[i * 2] as f32,
                            });
                            lines.push(vec3 {
                                x: bin_idx + bin_size * iter_x[i * 2 + 1] as f32,
                                y: bin_idy + bin_size * iter_y[i * 2 + 1] as f32,
                                z: bin_idz + bin_size * iter_z[i * 2 + 1] as f32,
                            });
                        }
                    }
                }
            }
        }
    }
    pub fn traverse(&self, pos: vec3, radius: f32, hit_history: &mut HashSet<u32>) {
        if pos.x > self.size
            || pos.y > self.size
            || pos.z > self.size
            || pos.x < -self.size
            || pos.y < -self.size
            || pos.z < -self.size
        {
            std::panic!();
        }
        let bin_size = (2.0 * self.size) / self.bin_count as f32;
        let nr = (radius / bin_size) as i32 + 1;
        // assert!(nr > 0);
        let bin_idx = (self.bin_count as f32 * (pos.x + self.size) / (2.0 * self.size)) as i32;
        let bin_idy = (self.bin_count as f32 * (pos.y + self.size) / (2.0 * self.size)) as i32;
        let bin_idz = (self.bin_count as f32 * (pos.z + self.size) / (2.0 * self.size)) as i32;
        for dz in -nr..nr + 1 {
            for dy in -nr..nr + 1 {
                for dx in -nr..nr + 1 {
                    let bin_idx = bin_idx + dx;
                    let bin_idy = bin_idy + dy;
                    let bin_idz = bin_idz + dz;
                    if bin_idx < 0
                        || bin_idy < 0
                        || bin_idz < 0
                        || bin_idx >= self.bin_count as i32
                        || bin_idy >= self.bin_count as i32
                        || bin_idz >= self.bin_count as i32
                    {
                        continue;
                    }
                    let flat_id = bin_idx as u32
                        + bin_idy as u32 * self.bin_count
                        + bin_idz as u32 * self.bin_count * self.bin_count;
                    let bin_id = self.bins_indices[flat_id as usize];
                    if bin_id != 0 {
                        for item in &self.bins[bin_id as usize] {
                            hit_history.insert(*item);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn UG_test() {
    let mut state = Sim_State::new(Sim_Params {
        rest_length: 0.1,
        spring_factor: 0.1,
        repell_factor: 0.1,
        planar_factor: 0.1,
        bulge_factor: 0.1,
        cell_radius: 0.1,
        cell_mass: 0.1,
        can_radius: 20.0,
    });
    use rand::distributions::{Distribution, Uniform};
    let size = 20.0;
    let range = Uniform::new(-size, size);
    let mut rng = rand::thread_rng();
    let N = 100000;
    for i in 0..N {
        state.pos.push(vec3 {
            x: range.sample(&mut rng),
            y: range.sample(&mut rng),
            z: range.sample(&mut rng),
        });
        state.links.insert((i, (i + 1) % N));
    }
    let M = 100;
    let bin_size = size * 2.0 / M as f32;
    let mut ug = UG::new(size, M);
    for (i, &pnt) in state.pos.iter().enumerate() {
        ug.put(pnt, i as u32);
    }
    {
        let rand_point = vec3 {
            x: range.sample(&mut rng),
            y: range.sample(&mut rng),
            z: range.sample(&mut rng),
        };
        let mut hit_history: HashSet<u32> = HashSet::new();
        ug.traverse(rand_point, size * 2.0, &mut hit_history);
        assert_eq!(hit_history.len(), N as usize);
    }
    for i in 0..100 {
        let rand_point = vec3 {
            x: range.sample(&mut rng),
            y: range.sample(&mut rng),
            z: range.sample(&mut rng),
        };
        let mut hit_history: HashSet<u32> = HashSet::new();
        let range = Uniform::new(0.5, size);
        let dist = range.sample(&mut rng);
        let portion = (dist) * (dist) / (size * size);
        ug.traverse(rand_point, dist, &mut hit_history);
        let theoretical_num = (N as f32 * portion) as usize;
        let true_num = hit_history.len();
        println!("{} {}", theoretical_num, true_num);
        for j in &hit_history {
            let point = state.pos[*j as usize];
            assert!(point.distance(rand_point.clone()) < (dist + bin_size * 2.0) * 2.5);
        }
        assert!(
            hit_history.len() > 0
                && theoretical_num as f32 * 2.0 + 10.0 > true_num as f32
                && theoretical_num as f32 * 0.08 < true_num as f32 + 10.0
        );
    }
}

#[test]
fn serde_test() {
    extern crate rand;
    let mut state = Sim_State::new(Sim_Params {
        rest_length: 0.1,
        spring_factor: 0.1,
        repell_factor: 0.1,
        planar_factor: 0.1,
        bulge_factor: 0.1,
        cell_radius: 0.1,
        cell_mass: 0.1,
        can_radius: 20.0,
    });
    use rand::distributions::{Distribution, Uniform};
    let range = Uniform::new(0.0, 1.0);
    let mut rng = rand::thread_rng();
    for i in 0..1000 {
        state.pos.push(vec3 {
            x: range.sample(&mut rng),
            y: range.sample(&mut rng),
            z: range.sample(&mut rng),
        });
        state.pos.push(vec3 {
            x: range.sample(&mut rng),
            y: range.sample(&mut rng),
            z: range.sample(&mut rng),
        });
        state.links.insert((i, i + 1));
    }
    state.dump("foo");
    assert_eq!(Sim_State::open("foo"), state);
}
