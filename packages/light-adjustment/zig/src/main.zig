const std = @import("std");
const math = std.math;
const ArrayList = std.ArrayList;
const Allocator = std.mem.Allocator;
const expect = std.testing.expect;
const test_allocator = std.testing.allocator;

/// 明るさ調整用のオプション群。
pub const LightAdjustmentOptions = struct {
    /// AGCWD の論文中に出てくる α パラメータ。
    ///
    /// この値が大きいほどコントラストが強調される傾向がある。
    alpha: f32 = 0.5,

    /// AGCWD の論文中に出てくる映像のフレーム間の差異を検知する際の閾値。
    ///
    /// 新しいフレームが以前のものと差異がある、と判定された場合には、
    /// 明るさ調整用のテーブルが更新されることになる。
    ///
    /// 値が小さくなるほど、微細な変化でテーブルが更新されるようになるため、
    /// より適切な調整が行われやすくなるが、その分処理負荷は高くなる可能性がある。
    ///
    /// 0 以下の場合には毎フレームで更新されるようになる。
    entropy_threshold: f32 = 0.05,

    /// 処理後の画像の最低の明るさ (HSV の V の値）。
    min_intensity: u8 = 0,

    /// 処理後の画像の最大の明るさ (HSV の V の値）。
    max_intensity: u8 = 255,

    /// 明るさ調整処理の度合い。
    ///
    /// AGCWD による明るさ調整処理適用前後の画像の混合割合（パーセンテージ）で、
    /// 0 なら処理適用前の画像が、100 なら処理適用後の画像が、50 ならそれらの半々が、採用されることになる。
    adjustment_level: u8 = 50,

    /// 明るさ調整後に画像に適用するシャープネス処理の度合い。
    ///
    /// シャープネス処理適用前後の画像の混合割合（パーセンテージ）で、
    /// 0 なら処理適用前の画像が、100 なら処理適用後の画像が、50 ならそれらの半々が、採用されることになる。
    sharpness_level: u8 = 20,
};

/// 明るさ調整を行うための構造体。
///
/// 明るさ調整は「Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution」論文がベース。
/// 明るさ調整後には deconvolution filter に基づくシャープネス処理を適用する。
pub const LightAdjustment = struct {
    const Self = @This();

    options: LightAdjustmentOptions,
    agcwd: Agcwd,
    sharpener: Sharpener,
    image: RgbaImage,
    mask: FocusMask,

    pub fn init(allocator: Allocator, width: u32, height: u32) !Self {
        const image = try RgbaImage.init(allocator, width, height);
        errdefer image.deinit();

        const mask = try FocusMask.init(allocator, width, height);
        errdefer mask.deinit();

        const sharpener = try Sharpener.init(allocator, width, height);
        errdefer sharpener.deinit();

        return .{
            .options = .{},
            .agcwd = Agcwd.init(),
            .sharpener = sharpener,
            .image = image,
            .mask = mask,
        };
    }

    pub fn deinit(self: Self) void {
        self.image.deinit();
        self.mask.deinit();
        self.sharpener.deinit();
    }

    pub fn isStateObsolete(self: *const Self) bool {
        return self.agcwd.isStateObsolete(&self.image, &self.options);
    }

    pub fn updateState(self: *Self) void {
        self.agcwd.updateState(&self.image, &self.mask, &self.options);
    }

    pub fn processImage(self: *Self) void {
        self.agcwd.processImage(&self.image);
        self.sharpener.processImage(&self.image, &self.options);
    }

    pub fn resize(self: *Self, width: u32, height: u32) !void {
        self.deinit();

        self.image = try RgbaImage.init(self.image.allocator, width, height);
        errdefer self.image.deinit();

        self.mask = try FocusMask.init(self.mask.allocator, width, height);
        errdefer self.mask.deinit();

        self.sharpener = try Sharpener.init(self.sharpener.allocator, width, height);
        errdefer self.sharpener.deinit();
    }
};

/// AGCWD で明るさ調整を行う際に、画像のどの部分を基準に調整するかを指定するための構造体。
///
/// これによって「逆光時に画像全体ではなく人物部分の明るさを改善するように指定する」といったことが可能になる。
/// なお、このマスクの値によらず、明るさ調整処理自体は常に画像全体に対して適用される（その適用のされ方が変わるだけ）。
const FocusMask = struct {
    const Self = @This();

    /// 画像の各ピクセルの重み。
    /// 0 ならそのピクセルは無視され、値が大きくなるほど明るさ調整の際に重視されるようになる。
    data: []u8,

    allocator: Allocator,

    fn init(allocator: Allocator, width: u32, height: u32) !Self {
        const data = try allocator.alloc(u8, width * height);
        return .{ .data = data, .allocator = allocator };
    }

    fn deinit(self: Self) void {
        self.allocator.free(self.data);
    }
};

/// 処理対象の画像データを格納するための構造体。
const RgbaImage = struct {
    const Self = @This();

    width: u32,
    height: u32,

    /// RGBA 形式のピクセル列。
    data: []u8,

    allocator: Allocator,

    fn init(allocator: Allocator, width: u32, height: u32) !Self {
        const data = try allocator.alloc(u8, width * height * 4);
        return .{ .width = width, .height = height, .data = data, .allocator = allocator };
    }

    fn deinit(self: Self) void {
        self.allocator.free(self.data);
    }

    fn getRgb(self: Self, offset: usize) Rgb {
        return .{
            .r = self.data[offset],
            .g = self.data[offset + 1],
            .b = self.data[offset + 2],
        };
    }

    fn setRgb(self: Self, offset: usize, rgb: Rgb) void {
        self.data[offset] = rgb.r;
        self.data[offset + 1] = rgb.g;
        self.data[offset + 2] = rgb.b;
    }
};

/// SoA形式でRGB画像を表現する構造体。
const RgbImageSoA = struct {
    const Self = @This();

    width: u32,
    height: u32,

    r: []i16,
    g: []i16,
    b: []i16,

    allocator: Allocator,

    fn init(allocator: Allocator, width: u32, height: u32) !Self {
        const r = try allocator.alloc(i16, width * height);
        const g = try allocator.alloc(i16, width * height);
        const b = try allocator.alloc(i16, width * height);
        return .{ .width = width, .height = height, .r = r, .g = g, .b = b, .allocator = allocator };
    }

    fn deinit(self: Self) void {
        self.allocator.free(self.r);
        self.allocator.free(self.g);
        self.allocator.free(self.b);
    }

    fn fromRgbaImage(self: Self, image: *const RgbaImage) void {
        if (image.width != self.width or image.height != self.height) return;
        comptime var vec_size = std.simd.suggestVectorSize(u8) orelse unreachable;
        var idx: usize = 0;
        while (idx + vec_size < image.width * image.height) : (idx += vec_size) {
            const vec: @Vector(vec_size * 4, u8) = image.data[idx * 4 ..][0 .. vec_size * 4].*;
            const rgba = std.simd.deinterlace(4, vec);
            self.r[idx..][0..vec_size].* = @intCast(@Vector(vec_size, i16), rgba[0]);
            self.g[idx..][0..vec_size].* = @intCast(@Vector(vec_size, i16), rgba[1]);
            self.b[idx..][0..vec_size].* = @intCast(@Vector(vec_size, i16), rgba[2]);
        }
        while (idx < image.width * image.height) : (idx += 4) {
            self.r[idx] = @intCast(i16, image.data[idx * 4]);
            self.g[idx] = @intCast(i16, image.data[idx * 4 + 1]);
            self.b[idx] = @intCast(i16, image.data[idx * 4 + 2]);
        }
    }
};

const Rgb = struct {
    const Self = @This();

    r: u8,
    g: u8,
    b: u8,

    fn intensity(self: Self) u8 {
        return @max(self.r, @max(self.g, self.b));
    }
};

const RgbI32 = struct {
    const Self = @This();

    r: i32,
    g: i32,
    b: i32,

    fn init() Self {
        return .{ .r = 0, .g = 0, .b = 0 };
    }

    fn clamp(self: *Self) void {
        self.r = @max(0, @min(self.r, 255));
        self.g = @max(0, @min(self.g, 255));
        self.b = @max(0, @min(self.b, 255));
    }

    fn blend(self: Self, original: Rgb, level: u8) Rgb {
        return .{
            .r = @truncate(u8, (@intCast(u32, self.r) * level + @intCast(u32, original.r) * (64 - level)) >> 6),
            .g = @truncate(u8, (@intCast(u32, self.g) * level + @intCast(u32, original.g) * (64 - level)) >> 6),
            .b = @truncate(u8, (@intCast(u32, self.b) * level + @intCast(u32, original.b) * (64 - level)) >> 6),
        };
    }
};

/// AGCWD アルゴリズムに基づいて画像の明るさ調整処理を行うための構造体。
///
/// 論文: Efficient Contrast Enhancement Using Adaptive Gamma Correction With Weighting Distribution
const Agcwd = struct {
    const Self = @This();

    /// 明るさ調整用の変換テーブル。
    /// 形式: `table[変換前（ピクセル）の intensity][変換後の R or G or B 値] = 変換後の R or G or B 値`
    table: [256][256]u8,

    /// テーブルの更新が必要かどうかの判定に使われる画像のエントロピー値。
    entropy: f32 = -1.0,

    fn init() Self {
        // updateState() が呼ばれるまでは入力画像と出力画像が一致するマッピングとなる
        var table: [256][256]u8 = undefined;
        for (0..table.len) |i| {
            for (0..i + 1) |j| {
                table[i][j] = @truncate(u8, j);
            }
        }
        return .{ .table = table };
    }

    fn isStateObsolete(self: *const Self, image: *const RgbaImage, options: *const LightAdjustmentOptions) bool {
        const pdf = Pdf.fromImage(image);
        const entropy = pdf.entropy();
        return @fabs(self.entropy - entropy) > options.entropy_threshold;
    }

    fn updateState(self: *Self, image: *const RgbaImage, mask: *const FocusMask, options: *const LightAdjustmentOptions) void {
        // エントロピーは画像全体から求める
        var pdf = Pdf.fromImage(image);
        self.entropy = pdf.entropy();

        // 明るさ変換に使うテーブルはマスク領域だけから求める
        pdf = Pdf.fromImageAndMask(image, mask);
        const cdf = Cdf.fromPdf(&pdf.toWeightingDistribution(options.alpha));
        const mapping_curve = cdf.toIntensityMappingCurve(options);
        for (0..mapping_curve.len) |intensity| {
            const processed_intensity = @intCast(u16, mapping_curve[intensity]);
            for (0..intensity) |pixel_value| {
                self.table[intensity][pixel_value] = @truncate(u8, pixel_value * processed_intensity / intensity);
            }
            self.table[intensity][intensity] = @truncate(u8, processed_intensity);
        }
    }

    fn processImage(self: *const Self, image: *RgbaImage) void {
        var i: usize = 0;
        while (i < image.data.len) : (i += 4) {
            var rgb = image.getRgb(i);
            const v = rgb.intensity();
            rgb.r = self.table[v][rgb.r];
            rgb.g = self.table[v][rgb.g];
            rgb.b = self.table[v][rgb.b];
            image.setRgb(i, rgb);
        }
    }
};

/// Probability Density Function.
const Pdf = struct {
    const Self = @This();

    table: [256]f32,

    fn fromImage(image: *const RgbaImage) Self {
        return Self.fromImageAndMask(image, null);
    }

    fn fromImageAndMask(image: *const RgbaImage, mask: ?*const FocusMask) Self {
        var histogram = [_]usize{0} ** 256;
        var total: usize = 0;
        {
            var i: usize = 0;
            while (i < image.data.len) : (i += 4) {
                const weight = if (mask) |non_null_mask| non_null_mask.data[i / 4] else 255;
                histogram[image.getRgb(i).intensity()] += weight;
                total += weight;
            }
        }

        var table: [256]f32 = undefined;
        if (total > 0) {
            const n = @intToFloat(f32, total);

            for (histogram, 0..) |weight, i| {
                table[i] = @intToFloat(f32, weight) / n;
            }
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

/// Cumulative Distribution Function.
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

    fn toIntensityMappingCurve(self: *const Self, options: *const LightAdjustmentOptions) [256]u8 {
        var curve: [256]u8 = undefined;
        const min: f32 = @intToFloat(f32, options.min_intensity);
        const max: f32 = @intToFloat(f32, options.max_intensity);
        const range: f32 = @max(0.0, max - min);
        const ratio = @intToFloat(f32, options.adjustment_level) / 100.0;

        for (self.table, 0..) |gamma, i| {
            const v0: f32 = @intToFloat(f32, i);
            const v1: f32 = range * math.pow(f32, v0 / 255.0, 1.0 - gamma) + min;
            curve[i] = @floatToInt(u8, math.round(v0 * (1.0 - ratio) + v1 * ratio));
        }
        return curve;
    }
};

/// Deconvolution filter によって画像のシャープネス処理を行うための構造体。
const Sharpener = struct {
    const Self = @This();

    /// 処理中の一時データを保持するためのフィールド。
    temp_image: RgbImageSoA,

    allocator: Allocator,

    fn init(allocator: Allocator, width: u32, height: u32) !Self {
        const temp_image = try RgbImageSoA.init(allocator, width, height);
        return .{ .temp_image = temp_image, .allocator = allocator };
    }

    fn deinit(self: Self) void {
        self.temp_image.deinit();
    }

    // TODO(sile): 端の処理
    fn processImage(self: *const Self, image: *RgbaImage, options: *const LightAdjustmentOptions) void {
        const level = options.sharpness_level;
        if (level == 0) {
            return;
        }
        comptime var vec_size = std.simd.suggestVectorSize(i16) orelse unreachable;
        const Vec = @Vector(vec_size, i16);

        const filter = [_]i8{ 0, -1, 0, -1, 5, -1, 0, -1, 0 };
        // const filter16 = [_]i16{ 0, -1, 0, -1, 5, -1, 0, -1, 0 };
        self.temp_image.fromRgbaImage(image);

        for (1..image.height - 1) |y| {
            var x: usize = 1;

            while (x + vec_size < image.width - 1) : (x += vec_size) {
                const lr: Vec = self.temp_image.r[y * image.width + x - 1 ..][0..vec_size].*;
                const lg: Vec = self.temp_image.g[y * image.width + x - 1 ..][0..vec_size].*;
                const lb: Vec = self.temp_image.b[y * image.width + x - 1 ..][0..vec_size].*;
                const cr: Vec = self.temp_image.r[y * image.width + x ..][0..vec_size].*;
                const cg: Vec = self.temp_image.g[y * image.width + x ..][0..vec_size].*;
                const cb: Vec = self.temp_image.b[y * image.width + x ..][0..vec_size].*;
                const rr: Vec = self.temp_image.r[y * image.width + x + 1 ..][0..vec_size].*;
                const rg: Vec = self.temp_image.g[y * image.width + x + 1 ..][0..vec_size].*;
                const rb: Vec = self.temp_image.b[y * image.width + x + 1 ..][0..vec_size].*;
                const tr: Vec = self.temp_image.r[(y - 1) * image.width + x ..][0..vec_size].*;
                const tg: Vec = self.temp_image.g[(y - 1) * image.width + x ..][0..vec_size].*;
                const tb: Vec = self.temp_image.b[(y - 1) * image.width + x ..][0..vec_size].*;
                const br: Vec = self.temp_image.r[(y + 1) * image.width + x ..][0..vec_size].*;
                const bg: Vec = self.temp_image.g[(y + 1) * image.width + x ..][0..vec_size].*;
                const bb: Vec = self.temp_image.b[(y + 1) * image.width + x ..][0..vec_size].*;

                const vec5 = @splat(vec_size, @as(i16, 5));
                const processed_r = cr * vec5 - (lr + rr + tr + br);
                const processed_g = cg * vec5 - (lg + rg + tg + bg);
                const processed_b = cb * vec5 - (lb + rb + tb + bb);

                const vec255 = @splat(vec_size, @as(i16, 255));
                const vec0 = @splat(vec_size, @as(i16, 0));
                const clamped_r = @max(vec0, @min(vec255, processed_r));
                const clamped_g = @max(vec0, @min(vec255, processed_g));
                const clamped_b = @max(vec0, @min(vec255, processed_b));

                const level_vec = @splat(vec_size, @as(i16, level));
                const level2_vec = @splat(vec_size, @as(i16, 64 - level));
                const shift_vec = @splat(vec_size, @as(i16, 6));
                const blend_r = (clamped_r * level_vec + cr * level2_vec) >> shift_vec;
                const blend_g = (clamped_g * level_vec + cg * level2_vec) >> shift_vec;
                const blend_b = (clamped_b * level_vec + cb * level2_vec) >> shift_vec;

                const blend_r8 = @intCast(@Vector(vec_size, u8), blend_r);
                const blend_g8 = @intCast(@Vector(vec_size, u8), blend_g);
                const blend_b8 = @intCast(@Vector(vec_size, u8), blend_b);
                const blend_a8 = @splat(vec_size, @as(u8, 255));

                const blend = std.simd.interlace(.{ blend_r8, blend_g8, blend_b8, blend_a8 });

                image.data[y * image.width * 4 + x * 4 ..][0 .. vec_size * 4].* = blend;
            }
            while (x < image.width) : (x += 1) {
                var processed = RgbI32.init();
                for (0..3) |fy| {
                    if ((fy == 0 and y == 0) or (fy == 2 and y + 1 == image.height)) {
                        continue;
                    }

                    for (0..3) |fx| {
                        if ((fx == 0 and x == 0) or (fx == 2 and x + 1 == image.width)) {
                            continue;
                        }

                        const f = filter[fy * 3 + fx];
                        const original = image.getRgb(((y + fy - 1) * image.width + (x + fx - 1)) * 4);
                        processed.r += @intCast(i32, original.r) * f;
                        processed.g += @intCast(i32, original.g) * f;
                        processed.b += @intCast(i32, original.b) * f;
                    }
                }
                processed.clamp();

                const i = (y * image.width + x) * 4;
                const original = image.getRgb(i);
                image.setRgb(i, processed.blend(original, level));
            }
        }
    }
};

test "Process image" {
    var la = try LightAdjustment.init(test_allocator, 32, 32);
    defer la.deinit();

    std.mem.copy(u8, la.image.data, &([_]u8{ 1, 2, 3, 255, 4, 5, 6, 255, 7, 8, 9, 255, 10, 11, 12, 255 } ** 256));
    std.mem.copy(u8, la.mask.data, &([_]u8{ 0, 255, 10, 200 } ** 256));

    // 最初は常に状態の更新が必要
    try expect(la.isStateObsolete());
    la.updateState();

    // すでに更新済み
    try expect(!la.isStateObsolete());

    // 画像を処理
    la.processImage();
}
