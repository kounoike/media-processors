const builtin = @import("builtin");
const std = @import("std");
const math = std.math;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const test_allocator = std.testing.allocator;

const vector_size = 32;

pub const Mask = struct {
    data: []u8,

    allocator: ?Allocator = null,

    const Self = @This();

    pub fn fromSlice(data: []u8) !Self {
        return .{ .data = data };
    }

    pub fn deinit(self: Self) void {
        if (self.allocator) |allocator| {
            allocator.free(self.data);
        }
    }
};

// RGBA
pub const Image = struct {
    data: []u8,
    allocator: ?Allocator = null,

    const Self = @This();

    pub fn fromSlice(data: []u8) !Self {
        if (data.len % 4 != 0) {
            return error.NotRgbaImageData;
        }
        return .{ .data = data };
    }

    pub fn deinit(self: Self) void {
        if (self.allocator) |allocator| {
            allocator.free(self.data);
        }
    }

    fn getRgb(self: Self, offset: usize) Rgb {
        return .{
            .r = self.data[offset],
            .g = self.data[offset + 1],
            .b = self.data[offset + 2],
        };
    }

    fn getRgbVector(self: Self, offset: usize) RgbVector {
        var r: @Vector(vector_size, u8) = undefined;
        var g: @Vector(vector_size, u8) = undefined;
        var b: @Vector(vector_size, u8) = undefined;
        for (0..vector_size) |i| {
            r[i] = self.data[offset + 4 * i + 0];
            g[i] = self.data[offset + 4 * i + 1];
            b[i] = self.data[offset + 4 * i + 2];
        }
        return RgbVector{ .r = r, .g = g, .b = b };
    }

    fn setRgb(self: Self, offset: usize, rgb: Rgb) void {
        self.data[offset] = rgb.r;
        self.data[offset + 1] = rgb.g;
        self.data[offset + 2] = rgb.b;
    }

    fn setRgbVector(self: Self, offset: usize, rgbVector: RgbVector) void {
        for (0..vector_size) |i| {
            self.data[offset + 4 * i + 0] = rgbVector.r[i];
            self.data[offset + 4 * i + 1] = rgbVector.g[i];
            self.data[offset + 4 * i + 2] = rgbVector.b[i];
        }
    }
};

pub const AgcwdOptions = struct {
    /// 論文中に出てくる α パラメータ
    alpha: f32 = 0.5,
    fusion: f32 = 0.5,
    min_intensity: u8 = 10,
    max_intensity: u8 = 245,
    entropy_threshold: f32 = 0.05,
};

/// 画像の明るさ調整処理を行うための構造体
///
/// 「Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution」論文がベース
// TODO: オリジナルとの違いを書く
// - セグメンテーションマスク
// - 明るさの底上げ
// - fusion
pub const Agcwd = struct {
    options: AgcwdOptions,
    entropy: f32 = -1.0,
    mapping_curve: [256]u8,

    const Self = @This();

    pub fn init(options: AgcwdOptions) Self {
        // updateState() が呼ばれるまでは入力画像と出力画像が一致するマッピングとなる
        // (実際には RGB <-> HSV 変換による誤差の影響によって微妙に変化する）
        var mapping_curve: [256]u8 = undefined;
        for (0..mapping_curve.len) |i| {
            mapping_curve[i] = @truncate(u8, i);
        }

        return .{ .options = options, .mapping_curve = mapping_curve };
    }

    pub fn isStateObsolete(self: Self, image: Image) bool {
        const pdf = Pdf.fromImage(image);
        const entropy = pdf.entropy();
        return @fabs(self.entropy - entropy) > self.options.entropy_threshold;
    }

    pub fn updateState(self: *Self, image: Image, mask: ?Mask) void {
        var pdf = Pdf.fromImage(image);
        self.entropy = pdf.entropy();

        pdf = Pdf.fromImageAndMask(image, mask);
        const cdf = Cdf.fromPdf(&pdf.toWeightingDistribution(self.options.alpha));
        self.mapping_curve = cdf.toIntensityMappingCurve(self.options);
    }

    pub fn enhanceImage(self: Self, image: *Image) void {
        var i: usize = 0;
        if (std.Target.wasm.featureSetHas(builtin.cpu.features, .simd128)) {
            while (i + 4 * vector_size <= image.data.len) : (i += 4 * vector_size) {
                const rgb_vector = image.getRgbVector(i);
                var hsv_vector = rgb_vector.toHsvVector();
                for (0..vector_size) |j| {
                    hsv_vector.v[j] = self.mapping_curve[hsv_vector.v[j]];
                }
                image.setRgbVector(i, hsv_vector.toRgbVector());
            }
        }
        while (i < image.data.len) : (i += 4) {
            const rgb = image.getRgb(i);
            var hsv = rgb.toHsv();
            hsv.v = self.mapping_curve[hsv.v];
            image.setRgb(i, hsv.toRgb());
        }
    }
};

const Pdf = struct {
    table: [256]f32,

    const Self = @This();

    fn fromImage(image: Image) Self {
        return Self.fromImageAndMask(image, null);
    }

    fn fromImageAndMask(image: Image, mask: ?Mask) Self {
        var histogram = [_]usize{0} ** 256;
        var total: usize = 0;
        {
            var i: usize = 0;
            while (i < image.data.len) : (i += 4) {
                const weight = if (mask) |non_null_mask| non_null_mask.data[i] else 255;
                histogram[image.getRgb(i).intensity()] += weight;
                total += weight;
            }
        }

        var table: [256]f32 = undefined;
        const n = @intToFloat(f32, total);

        for (histogram, 0..) |weight, i| {
            table[i] = @intToFloat(f32, weight) / n;
        }

        return .{ .table = table };
    }

    fn entropy(self: *const Self) f32 {
        var sum: f32 = 0;
        for (self.table) |intensity| {
            if (intensity > 0.0) {
                sum += intensity * @log(intensity);
            }
        }
        return -sum;
    }

    fn toWeightingDistribution(self: *const Self, alpha: f32) Self {
        var max_intensity = self.table[0];
        var min_intensity = self.table[0];
        for (self.table) |intensity| {
            max_intensity = @max(max_intensity, intensity);
            min_intensity = @min(min_intensity, intensity);
        }

        var table: [256]f32 = undefined;
        const range = max_intensity - min_intensity + math.floatEps(f32);

        for (self.table, 0..) |x, i| {
            table[i] = max_intensity * math.pow(f32, ((x - min_intensity) / range), alpha);
        }

        return .{ .table = table };
    }
};

const Cdf = struct {
    const Self = @This();

    table: [256]f32,

    fn fromPdf(pdf: *const Pdf) Self {
        var sum: f32 = 0.0;
        for (pdf.table) |intensity| {
            sum += intensity;
        }

        var table: [256]f32 = undefined;
        var acc: f32 = 0.0;
        for (pdf.table, 0..) |intensity, i| {
            acc += intensity;
            table[i] = acc / sum;
        }

        return .{ .table = table };
    }

    fn toIntensityMappingCurve(self: *const Self, options: AgcwdOptions) [256]u8 {
        var curve: [256]u8 = undefined;
        const min: f32 = @intToFloat(f32, options.min_intensity);
        const max: f32 = @intToFloat(f32, options.max_intensity);
        const range: f32 = @max(0.0, max - min);
        const fusion = options.fusion;

        for (self.table, 0..) |gamma, i| {
            const v0: f32 = @intToFloat(f32, i);
            const v1: f32 = range * math.pow(f32, v0 / 255.0, 1.0 - gamma) + min;
            curve[i] = @floatToInt(u8, math.round(v0 * (1.0 - fusion) + v1 * fusion));
        }
        return curve;
    }
};

// TODO: Sharpen

const Rgb = struct {
    r: u8,
    g: u8,
    b: u8,

    const Self = @This();

    fn intensity(self: Self) u8 {
        return @max(self.r, @max(self.g, self.b));
    }

    fn toHsv(self: Self) Hsv {
        const r: usize = self.r;
        const g: usize = self.g;
        const b: usize = self.b;
        const max = @max(r, @max(g, b));
        const min = @min(r, @min(g, b));
        const n = max - min;

        const s = if (max == 0) 0 else n * 255 / max;
        const v = max;
        var h: usize = undefined;
        if (n == 0) {
            h = 0;
        } else if (max == r) {
            if (g < b) {
                h = (6 * 255) + (g * 255 / n) - (b * 255 / n);
            } else {
                h = (g - b) * 255 / n;
            }
        } else if (max == g) {
            h = 2 * 255 + b * 255 / n - r * 255 / n;
        } else {
            h = 4 * 255 + r * 255 / n - g * 255 / n;
        }
        h /= 6;

        return .{ .h = @truncate(u8, h), .s = @truncate(u8, s), .v = @truncate(u8, v) };
    }
};

const Hsv = struct {
    h: u8,
    s: u8,
    v: u8,

    const Self = @This();

    fn toRgb(self: Self) Rgb {
        if (self.s == 0) {
            return .{ .r = self.v, .g = self.v, .b = self.v };
        }

        var r: usize = self.v;
        var g: usize = self.v;
        var b: usize = self.v;
        const s: usize = self.s;
        const h6: usize = @intCast(usize, self.h) * 6;

        const f = h6 % 255;
        switch (h6 / 255) {
            1 => {
                r = r * (255 * 255 - s * f) / (255 * 255);
                b = b * (255 - s) / 255;
            },
            2 => {
                r = r * (255 - s) / 255;
                b = b * (255 * 255 - s * (255 - f)) / (255 * 255);
            },
            3 => {
                r = r * (255 - s) / 255;
                g = g * (255 * 255 - s * f) / (255 * 255);
            },
            4 => {
                r = r * (255 * 255 - s * (255 - f)) / (255 * 255);
                g = g * (255 - s) / 255;
            },
            5 => {
                g = g * (255 - s) / 255;
                b = b * (255 * 255 - s * f) / (255 * 255);
            },
            else => {
                g = g * (255 * 255 - s * (255 - f)) / (255 * 255);
                b = b * (255 - s) / 255;
            },
        }

        return .{ .r = @truncate(u8, r), .g = @truncate(u8, g), .b = @truncate(u8, b) };
    }
};

const RgbVector = struct {
    const Self = @This();
    const Vu8 = @Vector(vector_size, u8);
    const V16 = @Vector(vector_size, i16);

    r: Vu8,
    g: Vu8,
    b: Vu8,

    fn toHsvVector(self: Self) HsvVector {
        const max = @max(self.r, @max(self.g, self.b));
        const min = @min(self.r, @min(self.g, self.b));
        const n = max - min;
        var v255max: Vu8 = undefined;
        var v255n: V16 = undefined;
        var r: V16 = undefined;
        var g: V16 = undefined;
        var b: V16 = undefined;
        for (0..vector_size) |i| {
            if (max[i] == 0) {
                v255max[i] = 0;
            } else {
                v255max[i] = @divExact(255, max[i]);
            }
            if (n[i] == 0) {
                v255n[i] = 0;
            } else {
                v255n[i] = @divExact(255, n[i]);
            }
            r[i] = @intCast(i16, self.r[i]);
            g[i] = @intCast(i16, self.g[i]);
            b[i] = @intCast(i16, self.b[i]);
        }

        const s = n * v255max;
        const v = max;
        const vu8_0 = @splat(vector_size, @intCast(u8, 0));
        const vi16_0 = @splat(vector_size, @intCast(i16, 0));
        const h = @select(
            i16,
            n == vu8_0,
            vi16_0,
            @select(
                i16,
                max == self.r,
                @select(
                    i16,
                    self.g < self.b,
                    @splat(vector_size, @intCast(i16, 6 * 255)) + (g * v255n) - (b * v255n),
                    (g - b) * v255n,
                ),
                @select(
                    i16,
                    max == self.g,
                    @splat(vector_size, @intCast(i16, 2 * 255)) + b * v255n - r * v255n,
                    @splat(vector_size, @intCast(i16, 4 * 255)) + r * v255n - g * v255n,
                ),
            ),
        );
        var hu8: Vu8 = undefined;
        for (0..vector_size) |i| {
            hu8[i] = @intCast(u8, @divTrunc(h[i], 6));
        }

        return .{ .h = hu8, .s = s, .v = v };
    }
};

const HsvVector = struct {
    const Self = @This();
    const Vu8 = @Vector(vector_size, u8);
    const Vu32 = @Vector(vector_size, u32);

    h: Vu8,
    s: Vu8,
    v: Vu8,

    fn toRgbVector(self: Self) RgbVector {
        const h: Vu32 = self.h;
        const s: Vu32 = self.s;
        const v: Vu32 = self.v;

        const h6: Vu32 = h * @splat(vector_size, @intCast(u32, 6));
        const v255 = @splat(vector_size, @intCast(u32, 255));

        const f = h6 % v255;
        const h6div255 = h6 / v255;

        const r = @select(
            u32,
            s == @splat(vector_size, @intCast(u32, 0)),
            v,
            @select(
                u32,
                h6div255 == @splat(vector_size, @intCast(u32, 1)),
                v * (v255 * v255 - s * f) / (v255 * v255),
                @select(
                    u32,
                    h6div255 == @splat(vector_size, @intCast(u32, 2)),
                    v * (v255 - s) / v255,
                    @select(
                        u32,
                        h6div255 == @splat(vector_size, @intCast(u32, 3)),
                        v * (v255 - s) / v255,
                        @select(
                            u32,
                            h6div255 == @splat(vector_size, @intCast(u32, 4)),
                            v * (v255 * v255 - s * (v255 - f)) / (v255 * v255),
                            v,
                        ),
                    ),
                ),
            ),
        );
        const g = @select(
            u32,
            s == @splat(vector_size, @intCast(u32, 0)),
            v,
            @select(
                u32,
                h6div255 == @splat(vector_size, @intCast(u32, 1)),
                v,
                @select(
                    u32,
                    h6div255 == @splat(vector_size, @intCast(u32, 2)),
                    v,
                    @select(
                        u32,
                        h6div255 == @splat(vector_size, @intCast(u32, 3)),
                        v * (v255 * v255 - s * f) / (v255 * v255),
                        @select(
                            u32,
                            h6div255 == @splat(vector_size, @intCast(u32, 4)),
                            v * (v255 - s) / v255,
                            @select(
                                u32,
                                h6div255 == @splat(vector_size, @intCast(u32, 5)),
                                v * (v255 - s) / v255,
                                v * (v255 * v255 - s * (v255 - f)) / (v255 * v255),
                            ),
                        ),
                    ),
                ),
            ),
        );
        const b = @select(
            u32,
            s == @splat(vector_size, @intCast(u32, 0)),
            v,
            @select(
                u32,
                h6div255 == @splat(vector_size, @intCast(u32, 1)),
                v * (v255 - s) / v255,
                @select(
                    u32,
                    h6div255 == @splat(vector_size, @intCast(u32, 2)),
                    v * (v255 * v255 - s * (v255 - f)) / (v255 * v255),
                    @select(
                        u32,
                        h6div255 == @splat(vector_size, @intCast(u32, 3)),
                        v,
                        @select(
                            u32,
                            h6div255 == @splat(vector_size, @intCast(u32, 4)),
                            v,
                            @select(
                                u32,
                                h6div255 == @splat(vector_size, @intCast(u32, 5)),
                                v * (v255 * v255 - s * f) / (v255 * v255),
                                v * (v255 - s) / v255,
                            ),
                        ),
                    ),
                ),
            ),
        );
        var ru8: Vu8 = undefined;
        var gu8: Vu8 = undefined;
        var bu8: Vu8 = undefined;
        for (0..vector_size) |i| {
            ru8[i] = @truncate(u8, r[i]);
            gu8[i] = @truncate(u8, g[i]);
            bu8[i] = @truncate(u8, b[i]);
        }

        return .{ .r = ru8, .g = gu8, .b = bu8 };
    }
};

test "RGB to HSV to RGB" {
    const inputs = .{
        Rgb{ .r = 255, .g = 0, .b = 0 },
        Rgb{ .r = 10, .g = 30, .b = 200 },
        Rgb{ .r = 222, .g = 222, .b = 222 },
    };

    inline for (inputs) |original_rgb| {
        const hsv = original_rgb.toHsv();
        const rgb = hsv.toRgb();
        try expect(try math.absInt(@intCast(i9, rgb.r) - @intCast(i9, original_rgb.r)) <= 2);
        try expect(try math.absInt(@intCast(i9, rgb.g) - @intCast(i9, original_rgb.g)) <= 2);
        try expect(try math.absInt(@intCast(i9, rgb.b) - @intCast(i9, original_rgb.b)) <= 2);
    }
}
test "RgbVector to HsvVector to RgbVector" {
    const original_rgb_vector = RgbVector{
        .r = @Vector(vector_size, u8){
            0,
            0,
            0,
            255,
            0,
            255,
            255,
            255,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        },
        .g = @Vector(vector_size, u8){
            0,
            0,
            255,
            0,
            255,
            0,
            255,
            255,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        },
        .b = @Vector(vector_size, u8){
            0,
            255,
            0,
            0,
            255,
            255,
            0,
            255,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        },
    };
    const hsv_vector = original_rgb_vector.toHsvVector();
    for (0..vector_size) |i| {
        const rgb = Rgb{ .r = original_rgb_vector.r[i], .g = original_rgb_vector.g[i], .b = original_rgb_vector.b[i] };
        const hsv = rgb.toHsv();
        try expect(hsv.h == hsv_vector.h[i]);
        try expect(hsv.s == hsv_vector.s[i]);
        try expect(hsv.v == hsv_vector.v[i]);
    }
    const rgb_vector = hsv_vector.toRgbVector();
    for (0..vector_size) |i| {
        const hsv = Hsv{ .h = hsv_vector.h[i], .s = hsv_vector.s[i], .v = hsv_vector.v[i] };
        const rgb = hsv.toRgb();
        try expect(rgb_vector.r[i] == rgb.r);
        try expect(rgb_vector.g[i] == rgb.g);
        try expect(rgb_vector.b[i] == rgb.b);
    }
}

test "Enhance image" {
    var data = [_]u8{ 1, 2, 3, 255, 4, 5, 6, 255 };
    var image = try Image.fromSlice(&data);
    defer image.deinit();

    var agcwd = Agcwd.init(.{});

    // 最初は常に状態の更新が必要
    try expect(agcwd.isStateObsolete(image));
    agcwd.updateState(image, null);

    // すでに更新済み
    try expect(!agcwd.isStateObsolete(image));

    // 画像を処理
    agcwd.enhanceImage(&image);
}
