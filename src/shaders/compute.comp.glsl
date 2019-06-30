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
    float ug_bin_size;
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

void iterate(
    vec3 ray_dir,
    vec3 ray_invdir,
    vec3 ray_origin,
    float hit_max,
    out uint iter,
    out vec3 out_val) {
    ivec3 exit, step, cell_id;
    vec3 axis_delta, cross_history;
    for (uint i = 0; i < 3; ++i) {
        // convert ray starting point to cell_id coordinates                                                                                                                                                 
        float ray_offset = ray_origin[i] + g_ubo.ug_size;
        cell_id[i] = int(clamp(floor(ray_offset / g_ubo.ug_bin_size), 0, float(g_ubo.ug_bins_count) - 1.0));
        // out_val[i] = cell_id[i];
        if (ray_dir[i] < 0) {
            axis_delta[i] = -g_ubo.ug_bin_size * ray_invdir[i];
            cross_history[i] = (cell_id[i] * g_ubo.ug_bin_size - ray_offset) * ray_invdir[i];
            // exit[i] = -1;
            step[i] = -1;
        }
        else {
            axis_delta[i] = g_ubo.ug_bin_size * ray_invdir[i];
            cross_history[i] = ((cell_id[i] + 1)  * g_ubo.ug_bin_size - ray_offset) * ray_invdir[i];
            // exit[i] = int();
            step[i] = 1;
        }
    }
    iter = 0;
    uint cell_id_offset = cell_id[2] * g_ubo.ug_bins_count * g_ubo.ug_bins_count
        + cell_id[1] * g_ubo.ug_bins_count + cell_id[0];
    int cell_id_cur = int(cell_id_offset);
    ivec3 cell_delta = step * ivec3(1, g_ubo.ug_bins_count, g_ubo.ug_bins_count * g_ubo.ug_bins_count);
    while (true) {
        uint o = cell_id_cur;
        uint bin_offset = g_bins.data[2 * o];
        if (bin_offset > 0) {
            uint pnt_cnt = g_bins.data[2 * o + 1];
            iter += pnt_cnt;
        }
        
        

        uint k =
            (uint(cross_history[0] < cross_history[1]) << 2) +
            (uint(cross_history[0] < cross_history[2]) << 1) +
            (uint(cross_history[1] < cross_history[2]));
        // uint k = k | (k << 3);
        // uint k = k | (k << 6);
        // uint k = k | (k << 12);
        // uvec3 k = uvec3(
        //     ,
        //     ,
            
        // );

        // const ivec3 map[8] = {
        //     ivec3(0, 0, 1), ivec3(0, 1, 0), ivec3(0, 0, 1), ivec3(0, 1, 0),
        //     ivec3(0, 0, 1), ivec3(0, 0, 1), ivec3(1, 0, 0), ivec3(1, 0, 0)};
        // const vec3 vmap[8] = {
        //     vec3(0, 0, 1), vec3(0, 1, 0), vec3(0, 0, 1), vec3(0, 1, 0),
        //     vec3(0, 0, 1), vec3(0, 0, 1), vec3(1, 0, 0), vec3(1, 0, 0)};
        // ivec3 axis = map[k];
        // vec3 vaxis = vmap[k];
        // if (hit_max < dot(vaxis, cross_history) - 1.0e-3) break;
        // cell_id_cur += axis.x * cell_delta.x + axis.y * cell_delta.y + axis.z * cell_delta.z;
        // cross_history += vaxis * axis_delta;

        const uint map[8] = {2, 1, 2, 1, 2, 2, 0, 0};
        uint axis = map[k];
        if (hit_max < cross_history[axis] - 1.0e-3) break;
        cell_id_cur += cell_delta[axis];
        cross_history[axis] += axis_delta[axis];


        // cell_id[axis] += step[axis];
        // if (
        //     cell_id_cur <= -1 ||
        //     cell_id_cur >= g_ubo.ug_bins_count * g_ubo.ug_bins_count * g_ubo.ug_bins_count
        // ) break;

        // out_val.x += axis_delta[axis];
    }
}

void main() {
    ivec2 dim = imageSize(resultImage);
    vec2 uv = vec2(gl_GlobalInvocationID.xy) / dim;
    if (gl_GlobalInvocationID.x > dim.x || gl_GlobalInvocationID.y > dim.y)
        return;
    vec3 ray_origin = g_ubo.camera_pos;
    vec2 xy = (-1.0 + 2.0 * uv) * vec2(g_ubo.camera_fov, 1.0);
    vec3 ray_dir = normalize(
        -g_ubo.camera_look
        +g_ubo.camera_up * xy.y
        + g_ubo.camera_right * xy.x
    );
    //float val = subgroupShuffle(ray_dir.x, 0);
    float hit_min;
    float hit_max;
    vec3 ray_invdir = 1.0/ray_dir;
    vec3 color = vec3(0.0, 0.0, 0.0);
    if (intersect_box(
        vec3(-g_ubo.ug_size),
        vec3(g_ubo.ug_size),
        ray_invdir, ray_origin, hit_min, hit_max)) {
        
        uint iter = 0;
        vec3 ray_box_hit = ray_origin + ray_dir * hit_min;
        vec3 out_val = vec3(0);
        iterate(ray_dir, ray_invdir, ray_box_hit, hit_max - hit_min, iter, out_val);
        if (iter > 0) {
            // vec3 ray_vox_hit = ray_box_hit + ray_dir * out_val.x;
            color = //ray_box_hit/g_ubo.ug_bin_size/128.0;
            //out_val/32.0;
            // ray_vox_hit*0.1 + 0.1;
            vec3(float(iter)/32.0);
        }
    }
    imageStore(resultImage, ivec2(gl_GlobalInvocationID.xy), vec4(color.xyz, 1.0));
}