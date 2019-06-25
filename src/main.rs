#[macro_use]
extern crate vulkano;
pub mod state;
use cgmath::*;
mod render;

fn main() {
    let mut state = state::Sim_State::new(state::Sim_Params {
        rest_length: 0.1,
        spring_factor: 0.1,
        planar_factor: 0.1,
        bulge_factor: 0.1,
        cell_radius: 0.1,
        cell_mass: 0.1,
        can_radius: 10.0,
    });
    use rand::distributions::{Distribution, Uniform};
    let range = Uniform::new(-1.0, 1.0);
    let mut rng = rand::thread_rng();
    let N = 100;
    state.links.push((0, 0));
    for i in 0..N {
        state.pos.push(state::vec3 {
            x: range.sample(&mut rng),
            y: range.sample(&mut rng),
            z: range.sample(&mut rng),
        });
        // state.links.push((i, (i + 1) % N));
    }
    render::render_main(
        &mut state,
        Box::new(|state: &mut state::Sim_State| {
            let M = 100;
            let size = state.params.can_radius;
            let bin_size = size * 2.0 / M as f32;
            let dt = 1.0e-3;
            let mut ug = state::UG::new(size, M);
            for (i, &pnt) in state.pos.iter().enumerate() {
                ug.put(pnt, i as u32);
            }
            let mut hit_history: Vec<u32> = Vec::new();
            let mut new_pos = state.pos.clone();
            for (i, &pnt) in state.pos.iter().enumerate() {
                hit_history.clear();
                ug.traverse(pnt, 1.0, &mut hit_history);
                let mut new_point = pnt;
                for &id in &hit_history {
                    let pnt_1 = state.pos[id as usize];
                    let dist = pnt.distance(pnt_1);
                    let force = dt / (dist * dist + 1.0);
                    let vforce = (pnt - pnt_1) * force;
                    new_point += vforce;
                }
                new_pos[i] = new_point;
            }
            // Force into the domain
            for pnt in &mut new_pos {
                let dist = ((pnt.x * pnt.x) + (pnt.y * pnt.y)).sqrt();
                let diff = dist - state.params.can_radius;
                if diff > 0.0 {
                    let k = diff / dist;
                    pnt.x -= pnt.x * k;
                    pnt.y -= pnt.y * k;
                }
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
}
