#[macro_use]
extern crate vulkano;
pub mod state;
use cgmath::*;
mod render;
use rand::distributions::{Distribution, Uniform};

fn main() {
    // let mut state = state::Sim_State::new(state::Sim_Params {
    //     rest_length: 0.3,
    //     spring_factor: 500.0,
    //     repell_factor: 1.0,
    //     planar_factor: 0.1,
    //     bulge_factor: 0.1,
    //     cell_radius: 0.1,
    //     cell_mass: 0.1,
    //     can_radius: 10.0,
    // });
    // let range = Uniform::new(-3.1, 3.1);
    // let mut rng = rand::thread_rng();
    // let N = 100;
    // state.links.insert((0, 0));
    // for i in 0..N {
    //     state.pos.push(state::vec3 {
    //         x: range.sample(&mut rng),
    //         y: range.sample(&mut rng),
    //         z: range.sample(&mut rng),
    //     });
    //     //state.links.insert((i, (i + 1) % N));
    // }
    let mut state = state::Sim_State::open("blob");
    
    render::render_main(
        &mut state,
        Box::new(|state: &mut state::Sim_State| {
            return;
            let birth_range = Uniform::new(0, 100);
            let range = Uniform::new(-0.01, 0.01);
            let mut rng = rand::thread_rng();
            let M = 40;
            let size = state.params.can_radius;
            let bin_size = size * 2.0 / M as f32;
            let dt = 1.0e-3;
            let mut ug = state::UG::new(size, M);
            for (i, &pnt) in state.pos.iter().enumerate() {
                ug.put(pnt, i as u32);
            }
            let mut hit_history: Vec<u32> = Vec::new();
            let mut force_history: Vec<f32> = Vec::new();
            let mut new_pos = state.pos.clone();
            // Repell
            for (i, &pnt) in state.pos.iter().enumerate() {
                hit_history.clear();
                ug.traverse(pnt, state.params.cell_radius * 10.0, &mut hit_history);
                let mut new_point = pnt;
                let mut acc_force = 0.0;
                for &id in &hit_history {
                    if id == i as u32 {
                        continue;
                    }
                    let pnt_1 = state.pos[id as usize];
                    let dist = pnt.distance(pnt_1);
                    if dist < state.params.rest_length * 1.0 && id > i as u32 {
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
                force_history[i as usize] += force;
                force_history[j as usize] -= force;
                new_pos[i as usize] += vforce;
                new_pos[j as usize] -= vforce;
            }
            // Division
            for (i, &pnt) in state.pos.iter().enumerate() {
                if birth_range.sample(&mut rng) == 0
                    && force_history[i as usize] < 300.0
                {
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
                let force = -pnt.z * 20.0;
                // force_history[i as usize] += f32::abs(force);
                pnt.z += force * dt;
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
    //state.dump("blob");
}
