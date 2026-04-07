const std = @import("std");

pub const ZigTokenizer = struct {
    allocator: std.mem.Allocator,
    memory_pool: []u32,

    vocab_size: u32,
    offsets: []const u32,
    blob: []const u8,

    vocab_map: std.StringHashMap(u32),

    pub fn load(allocator: std.mem.Allocator, file_path: []const u8) !ZigTokenizer {
        var file = try std.fs.cwd().openFile(file_path, .{});
        defer file.close();

        const file_size = try file.getEndPos();

        const u32_count = (file_size + 3) / 4;
        const memory_pool = try allocator.alloc(u32, u32_count);
        errdefer allocator.free(memory_pool);

        const raw_bytes = std.mem.sliceAsBytes(memory_pool)[0..file_size];
        _ = try file.readAll(raw_bytes);

        const aligned_buffer = @as([]align(4) u8, @alignCast(raw_bytes));

        var offset: usize = 0;

        const magic = aligned_buffer[offset .. offset + 4];
        if (!std.mem.eql(u8, magic, "ZDCT")) return error.InvalidFormat;
        offset += 4;

        const vocab_size = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
        offset += 4;
        const blob_size = std.mem.readInt(u32, aligned_buffer[offset .. offset + 4][0..4], .little);
        offset += 4;

        const offsets_bytes_len = (vocab_size + 1) * 4;
        const offsets_slice = @as([]align(4) u8, @alignCast(aligned_buffer[offset .. offset + offsets_bytes_len]));
        const offsets_array = std.mem.bytesAsSlice(u32, offsets_slice);
        offset += offsets_bytes_len;

        const blob_slice = aligned_buffer[offset .. offset + blob_size];

        var vocab_map = std.StringHashMap(u32).init(allocator);
        for (0..vocab_size) |i| {
            const start_idx = offsets_array[i];
            const end_idx = offsets_array[i + 1];
            const word = blob_slice[start_idx..end_idx];
            try vocab_map.put(word, @as(u32, @intCast(i)));
        }

        return ZigTokenizer{
            .allocator = allocator,
            .memory_pool = memory_pool,
            .vocab_size = vocab_size,
            .offsets = offsets_array,
            .blob = blob_slice,
            .vocab_map = vocab_map,
        };
    }

    pub fn getWord(self: *const ZigTokenizer, token_id: u32) []const u8 {
        if (token_id >= self.vocab_size) return "<|unknown|>";
        const start_idx = self.offsets[token_id];
        const end_idx = self.offsets[token_id + 1];
        return self.blob[start_idx..end_idx];
    }

    // FUNGSI ENCODE (VERSI RAW MEMORY SLICE: SANGAT CEPAT!)
    pub fn encode(self: *const ZigTokenizer, allocator: std.mem.Allocator, input_text: []const u8) ![]u32 {
        // Asumsi terburuk: 1 huruf = 1 token. Kita pesan memori seukuran panjang teks.
        var tokens = try allocator.alloc(u32, input_text.len);
        var token_count: usize = 0;
        var i: usize = 0;

        while (i < input_text.len) {
            var best_match_len: usize = 0;
            var best_token_id: u32 = 0;

            const max_lookahead = @min(input_text.len - i, 20);
            var len: usize = 1;

            while (len <= max_lookahead) : (len += 1) {
                const chunk = input_text[i .. i + len];
                if (self.vocab_map.get(chunk)) |token_id| {
                    best_match_len = len;
                    best_token_id = token_id;
                }
            }

            if (best_match_len > 0) {
                tokens[token_count] = best_token_id; // Masukkan ke dalam raw array
                token_count += 1;
                i += best_match_len;
            } else {
                i += 1;
            }
        }

        // Hanya kembalikan array sebesar jumlah kata yang benar-benar ditemukan
        return tokens[0..token_count];
    }

    pub fn deinit(self: *ZigTokenizer) void {
        self.vocab_map.deinit();
        self.allocator.free(self.memory_pool);
    }
};
