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

bool intersect_plane(vec3 p, vec3 n, vec3 ray, vec3 ray_origin, out vec3 hit) {
    vec3 dr = ray_origin - p;
    float proj = dot(n, dr);
    float ndv = dot(n, ray);
    if (proj * ndv > -1.0e-6) {
        return false;
    }
    float t = proj / ndv;
    hit = ray_origin - ray * t;
    return true;
}

void main() {
    ivec2 dim = imageSize(resultImage);
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / dim;
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
        return;
    vec3 ray_origin = g_ubo.camera_pos;
    vec2 xy = (-1.0 + 2.0 * uv) * vec2(1.0, 1.0/g_ubo.camera_fov);
    vec3 ray_dir = normalize(
        -g_ubo.camera_look
        +g_ubo.camera_up * xy.y
        + g_ubo.camera_right * xy.x
    );
    //float val = subgroupShuffle(ray_dir.x, 0);
    float val = 0.0;
    vec3 hit = vec3(0.0, 0.0, 0.0);
    if (intersect_plane(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0), ray_dir, ray_origin, hit)) {
        val = 1.0;
    }
    imageStore(resultImage, ivec2(gl_GlobalInvocationID.xy), vec4(hit.xyz, 1.0));
}