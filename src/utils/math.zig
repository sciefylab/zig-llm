const std = @import("std");

pub const LANES: usize = 8;
pub const Vec = @Vector(LANES, f32);

inline fn load8(slice: []const f32, i: usize) Vec {
    return .{
        slice[i + 0], slice[i + 1], slice[i + 2], slice[i + 3],
        slice[i + 4], slice[i + 5], slice[i + 6], slice[i + 7],
    };
}

inline fn store8(slice: []f32, i: usize, v: Vec) void {
    slice[i + 0] = v[0];
    slice[i + 1] = v[1];
    slice[i + 2] = v[2];
    slice[i + 3] = v[3];
    slice[i + 4] = v[4];
    slice[i + 5] = v[5];
    slice[i + 6] = v[6];
    slice[i + 7] = v[7];
}

pub inline fn dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);

    var sum_vec: Vec = @splat(0.0);
    var i: usize = 0;

    while (i + LANES <= a.len) : (i += LANES) {
        const va = load8(a, i);
        const vb = load8(b, i);
        sum_vec += va * vb;
    }

    var s: f32 = @reduce(.Add, sum_vec);
    while (i < a.len) : (i += 1) {
        s += a[i] * b[i];
    }
    return s;
}

pub inline fn axpy(y: []f32, a: f32, x: []const f32) void {
    std.debug.assert(y.len == x.len);

    const va: Vec = @splat(a);
    var i: usize = 0;

    while (i + LANES <= y.len) : (i += LANES) {
        const vx = load8(x, i);
        const vy = load8(y, i);
        const res = vy + (va * vx);
        store8(y, i, res);
    }

    while (i < y.len) : (i += 1) {
        y[i] += a * x[i];
    }
}

pub inline fn add(dst: []f32, src: []const f32) void {
    std.debug.assert(dst.len == src.len);
    var i: usize = 0;

    while (i + LANES <= dst.len) : (i += LANES) {
        const vd = load8(dst, i);
        const vs = load8(src, i);
        store8(dst, i, vd + vs);
    }

    while (i < dst.len) : (i += 1) {
        dst[i] += src[i];
    }
}

pub inline fn reluInPlace(x: []f32) void {
    var i: usize = 0;

    while (i + LANES <= x.len) : (i += LANES) {
        var v = load8(x, i);
        inline for (0..LANES) |j| {
            if (v[j] < 0.0) v[j] = 0.0;
        }
        store8(x, i, v);
    }

    while (i < x.len) : (i += 1) {
        if (x[i] < 0.0) x[i] = 0.0;
    }
}

pub inline fn clipInPlace(x: []f32, min_val: f32, max_val: f32) void {
    var i: usize = 0;

    while (i + LANES <= x.len) : (i += LANES) {
        var v = load8(x, i);
        inline for (0..LANES) |j| {
            if (v[j] < min_val) v[j] = min_val;
            if (v[j] > max_val) v[j] = max_val;
        }
        store8(x, i, v);
    }

    while (i < x.len) : (i += 1) {
        if (x[i] < min_val) x[i] = min_val;
        if (x[i] > max_val) x[i] = max_val;
    }
}

pub inline fn maxAbs(x: []const f32) f32 {
    var m: f32 = 0.0;
    for (x) |v| {
        const a = if (v < 0.0) -v else v;
        if (a > m) m = a;
    }
    return m;
}

pub inline fn fill(dst: []f32, value: f32) void {
    @memset(dst, value);
}
