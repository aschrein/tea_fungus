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
    // vec3 dt = ray_invdir * g_ubo.ug_bin_size;
    // vec3 box_min = floor((ray_origin - vec3(-g_ubo.ug_size)) / g_ubo.ug_bin_size);
    // vec3 ray_offset = ray_origin - vec3(-g_ubo.ug_size);
    // ivec3 orig_cell = ivec3(
    //                     clamp(
    //                         ray_offset / g_ubo.ug_bin_size,
    //                         0.0,
    //                         float(g_ubo.ug_bins_count) - 1.0
    //                         )
    //                     );
    // vec3 init_dt = ray_invdir * (ray_origin - init_dt * g_ubo.ug_bin_size)
    // vec3 drt = ray_invdir * g_ubo.ug_bin_size;
    ivec3 exit, step, cell;
    vec3 deltaT, nextCrossingT; 
    for (uint i = 0; i < 3; ++i) {
        // convert ray starting point to cell coordinates                                                                                                                                                 
        float rayOrigCell = ray_origin[i] + g_ubo.ug_size;
        
        cell[i] = int(clamp(floor(rayOrigCell / g_ubo.ug_bin_size), 0, float(g_ubo.ug_bins_count) - 1.0)); 
        // out_val[i] = cell[i];
        if (ray_dir[i] < 0) { 
            deltaT[i] = -g_ubo.ug_bin_size * ray_invdir[i]; 
            nextCrossingT[i] = (cell[i] * g_ubo.ug_bin_size - rayOrigCell) * ray_invdir[i]; 
            exit[i] = -1; 
            step[i] = -1; 
        } 
        else { 
            deltaT[i] = g_ubo.ug_bin_size * ray_invdir[i]; 
            nextCrossingT[i] = ((cell[i] + 1)  * g_ubo.ug_bin_size - rayOrigCell) * ray_invdir[i]; 
            exit[i] = int(g_ubo.ug_bins_count); 
            step[i] = 1; 
        } 
    }
    iter = 0;
    while (true) {
        uint o = cell[2] * g_ubo.ug_bins_count * g_ubo.ug_bins_count
        + cell[1] * g_ubo.ug_bins_count + cell[0];
        uint cnt = g_bins.data[2 * o];
        if (cnt > 0) {
            iter++;
        }
        uint k =
            (uint(nextCrossingT[0] < nextCrossingT[1]) << 2) +
            (uint(nextCrossingT[0] < nextCrossingT[2]) << 1) +
            (uint(nextCrossingT[1] < nextCrossingT[2]));
        const uint map[8] = {2, 1, 2, 1, 2, 2, 0, 0};
        uint axis = map[k];
        if (hit_max < nextCrossingT[axis]) break;
        cell[axis] += step[axis];
        if (cell[axis] == exit[axis]) break;
        nextCrossingT[axis] += deltaT[axis];
        // out_val.x += deltaT[axis];
    }

    // vec3 start = ray_origin + ray_dir * hit_min;
    // vec3 end = ray_origin + ray_dir * hit_max;
    // vec3 diff = start - end;
    // float inv_bin_size = 1.0/g_ubo.ug_bin_size;
    // int ix = int(diff.x*inv_bin_size);
    // int iy = int(diff.y*inv_bin_size);
    // int iz = int(diff.z*inv_bin_size);
    // uint ox = uint(abs((start.x - g_ubo.ug_size)*inv_bin_size));
    // uint oy = uint(abs((start.y - g_ubo.ug_size)*inv_bin_size));
    // uint oz = uint(abs((start.z - g_ubo.ug_size)*inv_bin_size));
    // int sx = sign(ix);
    // int sy = sign(iy);
    // int sz = sign(iz);
    // iter = 0;
    // for (int z = 0;; z += sz) {
    //     for (int y = 0;; y += sy) {
    //         for (int x = 0;; x += sx) {
    //             iter++;
    //             if (iter == 1000) {
    //                 return;
    //             }
    //             if (x == ix) {
    //                 break;
    //             }
    //         }
    //         if (y == iy) {
    //             break;
    //         }
    //     }
    //     if (z == ix) {
    //         break;
    //     }
    // }
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
            vec3(float(iter)/10.0);
        }
    }
    imageStore(resultImage, ivec2(gl_GlobalInvocationID.xy), vec4(color.xyz, 1.0));
}