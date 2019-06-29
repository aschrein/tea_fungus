## Run
```console
glslangValidator --target-env vulkan1.1 -o src/shaders/compute.comp.spv src/shaders/compute.comp.glsl && cargo build --release && cargo run
```
## Links
http://www.andylomas.com/extra/andylomas_paper_cellular_forms_aisb50.pdf
https://gpuopen.com/optimizing-gpu-occupancy-resource-usage-large-thread-groups/
https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
https://www.scratchapixel.com/lessons/advanced-rendering/introduction-acceleration-structure/grid
## License
Licensed under either of
 * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
at your option.
