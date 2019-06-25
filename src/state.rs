extern crate lzma;
use lzma::LzmaWriter;
extern crate cgmath;
extern crate bincode;
extern crate rand;
extern crate serde;

use bincode::{
    config, deserialize, deserialize_from, deserialize_in_place, serialize, serialized_size,
    ErrorKind, Result,
};
use cgmath::{Matrix3, Matrix4, Point3, Rad, Vector3};
use serde::de::Deserializer;
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;
pub type vec3 = Point3<f32>;
pub struct Sim_Params {
    rest_length: f32,
    spring_factor: f32,
    planar_factor: f32,
    bulge_factor: f32,
    cell_radius: f32,
    cell_mass: f32,
    can_radius: f32,
}
#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct Sim_State {
    pub pos: Vec<vec3>,
    pub links: Vec<(u32, u32)>,
}
impl Sim_State {
    pub fn new() -> Sim_State {
        Sim_State {
            pos: Vec::new(),
            links: Vec::new(),
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

#[test]
fn serde_test() {
    extern crate rand;
    let mut state = Sim_State::new();
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
        state.links.push((i, i + 1));
    }
    state.dump("foo");
    assert_eq!(Sim_State::open("foo"), state);
}