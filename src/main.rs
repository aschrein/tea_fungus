#[macro_use]
extern crate vulkano;
pub mod state;

mod render;

fn main() {
    let mut state = state::Sim_State::new();
    use rand::distributions::{Distribution, Uniform};
    let range = Uniform::new(-10.0, 10.0);
    let mut rng = rand::thread_rng();
    for i in 0..1000 {
        state.pos.push(state::vec3 {
            x: range.sample(&mut rng),
            y: range.sample(&mut rng),
            z: range.sample(&mut rng),
        });
        state.pos.push(state::vec3 {
            x: range.sample(&mut rng),
            y: range.sample(&mut rng),
            z: range.sample(&mut rng),
        });
        state.links.push((i, (i + 1) % 1000));
    }
    render::render_main(
        Box::new(move || {
            state.clone()
        })
    );
}