#[macro_use]
extern crate vulkano;
pub mod state;
pub mod vulkan_tests;
use cgmath::*;
mod render;
use rand::distributions::{Distribution, Uniform};
use std::collections::HashSet;

fn main() {
    let mut state = state::Sim_State::new(state::Sim_Params {
        rest_length: 0.2,
        spring_factor: 1000.0,
        repell_factor: 1.0,
        planar_factor: 0.1,
        bulge_factor: 0.1,
        cell_radius: 0.1,
        cell_mass: 0.1,
        can_radius: 10.0,
    });
    let range = Uniform::new(-1.1, 1.1);
    let mut rng = rand::thread_rng();
    let N = 1;
    state.links.insert((0, 0));
         state.pos.push(state::vec3 {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        });
    for i in 0..N {
        state.pos.push(state::vec3 {
            x: range.sample(&mut rng),
            y: range.sample(&mut rng),
            z: range.sample(&mut rng),
        });
        //state.links.insert((i, (i + 1) % N));
    }
    // let mut state = state::Sim_State::open("blob");
    // state.pos.push(state::vec3 {
    //         x: 0.12412,
    //         y: 0.15153,
    //         z: 0.0,
    //     });
    //     state.pos.push(state::vec3 {
    //         x: -4.0,
    //         y: -4.0,
    //         z: -4.0,
    //     });
    render::render_main(
        &mut state,
        Box::new(|state: &mut state::Sim_State| {
            // return;
            let birth_range = Uniform::new(0, 10);
            let range = Uniform::new(-0.01, 0.01);
            let mut rng = rand::thread_rng();

            let mut size = 0.0;
            for (i, &pnt) in state.pos.iter().enumerate() {
                size = std::cmp::max(
                    size as u32,
                    std::cmp::max(
                        f32::abs(pnt.x) as u32,
                        std::cmp::max(f32::abs(pnt.y) as u32, f32::abs(pnt.z) as u32),
                    ),
                ) as f32;
                ;
            }
            size += 1.0;
            let M = (size / 0.025) as u32;
            let mut ug = state::UG::new(size, M);
            for (i, &pnt) in state.pos.iter().enumerate() {
                ug.put(pnt, i as u32);
            }

            let bin_size = size * 2.0 / M as f32;
            let dt = 1.0e-3;
            let mut hit_history: HashSet<u32> = HashSet::new();
            let mut force_history: Vec<f32> = Vec::new();
            let mut new_pos = state.pos.clone();
            // Repell
            for (i, &pnt) in state.pos.iter().enumerate() {
                hit_history.clear();
                ug.traverse(pnt, state.params.cell_radius * 1.0, &mut hit_history);
                let mut new_point = pnt;
                let mut acc_force = 0.0;
                for &id in &hit_history {
                    if id == i as u32 {
                        continue;
                    }
                    let pnt_1 = state.pos[id as usize];
                    let dist = pnt.distance(pnt_1);
                    if dist < state.params.rest_length * 0.9 && id > i as u32 {
                        state.links.insert((i as u32, id));
                    }
                    // if dist > state.params.cell_radius {
                    //     continue;
                    // }
                    let force =
                        state.params.repell_factor * state.params.cell_mass / (dist * dist + 1.0);
                    acc_force += f32::abs(force);
                    let vforce = (pnt - pnt_1) / (dist + 1.0) * force * dt;
                    new_point += vforce;
                }
                new_pos[i] = new_point;
                force_history.push(acc_force);
            }
            // Attract
            for &(i, j) in &state.links {
                if (i, j) == (j, i) {
                    continue;
                }
                let pnt_1 = state.pos[i as usize].clone();
                let pnt_2 = state.pos[j as usize].clone();
                let dist = state.params.rest_length - pnt_1.distance(pnt_2.clone());
                let force = state.params.spring_factor * state.params.cell_mass * dist;
                let vforce = dt * (pnt_1 - pnt_2) * force;
                force_history[i as usize] += f32::abs(force);
                force_history[j as usize] += f32::abs(force);
                new_pos[i as usize] += vforce;
                new_pos[j as usize] -= vforce;
            }
            // Planarization
            struct Planar_Target {
                p: state::vec3,
                n: u32,
            }
            let mut spring_target = Vec::<Planar_Target>::new();
            for (i, &pnt) in state.pos.iter().enumerate() {
                spring_target.push(Planar_Target {
                    p: state::vec3{x:0.0, y:0.0, z:0.0},
                    n: 0,
                });
            }
            for &(i, j) in &state.links {
                if (i, j) == (j, i) {
                    continue;
                }
                let pnt_1 = state.pos[i as usize].clone();
                let pnt_2 = state.pos[j as usize].clone();
                // let dr = (pnt_1 - pnt_2);
                // let drn = dr.normalize();
                spring_target[i as usize].p += pnt_2;// + state.params.rest_length * drn;
                spring_target[i as usize].n += 1;
                spring_target[j as usize].p += pnt_1;// - state.params.rest_length * drn;
                spring_target[j as usize].n += 1;
            }
            for (i, &pnt) in state.pos.iter().enumerate() {
                let st = &spring_target[i];
                if st.n == 0 {
                    continue;
                }
                let pt = st.p / st.n as f32;
                let pnt_1 = state.pos[i as usize].clone();
                let dist = state.params.rest_length - pnt_1.distance(pt.clone());
                let force = state.params.spring_factor * state.params.cell_mass * dist;
                let vforce = dt * (pnt_1 - pt) * force;
                force_history[i as usize] += f32::abs(force);
                new_pos[i as usize] += vforce;
            }
            // Division
            for (i, &pnt) in state.pos.iter().enumerate() {
                if birth_range.sample(&mut rng) == 0 && force_history[i as usize] < 20.0 {
                    new_pos.push(
                        new_pos[i as usize].clone()
                            + state::vec3 {
                                x: range.sample(&mut rng),
                                y: range.sample(&mut rng),
                                z: range.sample(&mut rng),
                            },
                    );
                }
            }

            // Force into the domain
            for (i, pnt) in new_pos.iter_mut().enumerate() {
                let dist = ((pnt.x * pnt.x) + (pnt.y * pnt.y)).sqrt();
                let diff = dist - state.params.can_radius;
                if diff > 0.0 {
                    let k = diff / dist;
                    pnt.x -= pnt.x * k;
                    pnt.y -= pnt.y * k;
                }
                // if pnt.z < 0.0 {
                //     pnt.z = 0.0;
                // }
                // let force = -pnt.z * 4.0;
                // force_history[i as usize] += f32::abs(force);
                // pnt.z += force * dt;
                if pnt.z > state.params.can_radius {
                    pnt.z = state.params.can_radius;
                }
                if pnt.z < -state.params.can_radius {
                    pnt.z = -state.params.can_radius;
                }
                
            }
            state.pos = new_pos;
        }),
    );
    // state.dump("blob");
}
