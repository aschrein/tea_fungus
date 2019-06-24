extern crate lzma;
use lzma::LzmaWriter;
extern crate bincode;
extern crate nalgebra as na;
extern crate rand;
extern crate serde;

use rand::Rng;

// #[macro_use]
// extern crate serde_derive;
// extern crate serde_bytes;
use bincode::{
    config, deserialize, deserialize_from, deserialize_in_place, serialize, serialized_size,
    ErrorKind, Result,
};
use na::{Rotation, Rotation3, Vector3};
use serde::de::Deserializer;
use serde::ser::{SerializeSeq, Serializer};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::prelude::*;
type vec3 = Vector3<f32>;
#[derive(Serialize, Deserialize)]
#[serde(remote = "vec3")]
struct vec3Def {
    x: f32,
    y: f32,
    z: f32,
}
fn vec_serialize<S>(vec: &Vec<vec3>, s: S) -> std::result::Result<S::Ok, S::Error>
where
    S: Serializer,
{
    #[derive(Serialize)]
    struct Wrapper(#[serde(with = "vec3Def")] vec3);
    Vec::serialize(&vec.into_iter().map(|a| Wrapper(*a)).collect(), s)
    // let mut proxy: Vec<f32> = Vec::new();
    // for item in vec.iter() {
    //     proxy.push(item.x);
    //     proxy.push(item.y);
    //     proxy.push(item.z);
    // }
    // s.serialize_struct("proxy", &proxy)
    //     let mut seq_ser = s.serialize_seq(Some(vec.len()))?;
    //         for item in vec.iter() {
    //             seq_ser.serialize_element(&item.x)?;
    //             seq_ser.serialize_element(&item.y)?;
    //             seq_ser.serialize_element(&item.z)?;
    //         }
    //         seq_ser.end()
}
fn vec_deserialize<'de, D>(d: D) -> std::result::Result<Vec<vec3>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct Wrapper(#[serde(with = "vec3Def")] vec3);
    let v = Vec::deserialize(d)?;
    Ok(v.into_iter().map(|Wrapper(a)| a).collect())
    // d.deserialize_struct(name: &'static str, fields: &'static [&'static str], visitor: V)
}

struct Sim_Params {
    rest_length: f32,
    spring_factor: f32,
    planar_factor: f32,
    bulge_factor: f32,
    cell_radius: f32,
    cell_mass: f32,
    can_radius: f32,
}
#[derive(Serialize, Deserialize, Debug, PartialEq)]
struct Sim_State {
    #[serde(serialize_with = "vec_serialize", deserialize_with = "vec_deserialize")]
    pos: Vec<vec3>,
    links: Vec<(u32, u32)>,
}
// impl Serialize for Sim_State {
//     fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
//     where
//         S: Serializer,
//     {
//         let mut seq_ser = serializer.serialize_seq(Some(self.pos.len()))?;
//         for item in &self.pos {
//             seq_ser.serialize_element(&item.x)?;
//             seq_ser.serialize_element(&item.y)?;
//             seq_ser.serialize_element(&item.z)?;
//         }
//         seq_ser.end()?;
//         let mut seq_ser = serializer.serialize_seq(Some(self.links.len()))?;
//         for item in &self.links {
//             seq_ser.serialize_element(&item.0)?;
//             seq_ser.serialize_element(&item.1)?;
//         }
//         seq_ser.end()
//     }
// }
impl Sim_State {
    fn new() -> Sim_State {
        Sim_State {
            pos: Vec::new(),
            links: Vec::new(),
        }
    }
    fn open(filename: &str) -> Sim_State {
        let mut file = File::open(filename).unwrap();
        let mut contents: Vec<u8> = Vec::new();
        file.read_to_end(&mut contents).unwrap();
        let contents = lzma::decompress(&mut contents).unwrap();
        deserialize(&contents).unwrap()
    }
    fn dump(&self, filename: &str) {
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
