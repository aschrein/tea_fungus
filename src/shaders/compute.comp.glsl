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
layout(set = 0, binding = 2) buffer Bins {
    uint data[];
} g_bins;
layout(set = 0, binding = 3) buffer Particles {
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

bool intersect_box(
        vec3 box_min,
        vec3 box_max,
        vec3 ray_invdir,
        vec3 ray_origin,
        out float hit_min,
        out float hit_max
    ) {
    vec3 tbot = ray_invdir * (box_min - ray_origin);
    vec3 ttop = ray_invdir * (box_max - ray_origin);
    vec3 tmin = min(ttop, tbot);
    vec3 tmax = max(ttop, tbot);
    vec2 t = max(tmin.xx, tmin.yz);
    float t0 = max(t.x, t.y);
    t = min(tmax.xx, tmax.yz);
    float t1 = min(t.x, t.y);
    hit_min = t0;
    hit_max = t1;
    return t1 > max(t0, 0.0);
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
    float hit_min;
    float hit_max;
    vec3 color = vec3(0.0, 0.0, 0.0);
    if (intersect_box(
        vec3(-g_ubo.ug_size),
        vec3(g_ubo.ug_size),
        1.0/ray_dir, ray_origin, hit_min, hit_max)) {
        color = ray_origin + ray_dir * hit_min;
    }
    imageStore(resultImage, ivec2(gl_GlobalInvocationID.xy), vec4(color.xyz, 1.0));
}