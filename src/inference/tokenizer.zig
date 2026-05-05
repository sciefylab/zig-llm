const std = @import("std");

pub const Tokenizer = struct {
    arena: std.heap.ArenaAllocator,
    id_to_token: [][]const u8,
    token_to_id: std.StringHashMap(u32),
    max_token_length: usize, // Sangat krusial untuk algoritma BPE/Subword

    /// Memuat Vocab JSON ke dalam RAM
    pub fn load(parent_allocator: std.mem.Allocator, json_path: []const u8) !Tokenizer {
        // Semua alokasi memori terkait tokenizer diikat ke dalam satu Arena
        var arena = std.heap.ArenaAllocator.init(parent_allocator);
        errdefer arena.deinit();
        const allocator = arena.allocator();

        const file = try std.fs.cwd().openFile(json_path, .{});
        defer file.close();

        // Alokasi memori untuk membaca file JSON
        const buf = try file.readToEndAlloc(allocator, 15 * 1024 * 1024); // Maksimal 15 MB
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, buf, .{});
        const obj = parsed.value.object;

        var max_id: usize = 0;
        var max_len: usize = 0;

        // Cari ID terbesar dan Panjang Karakter Subword terpanjang
        var it = obj.iterator();
        while (it.next()) |e| {
            const id: usize = @intCast(e.value_ptr.integer);
            if (id > max_id) max_id = id;
            if (e.key_ptr.*.len > max_len) max_len = e.key_ptr.*.len;
        }

        // Siapkan array pemetaan ID -> Subword
        const id_map = try allocator.alloc([]const u8, max_id + 1);
        @memset(id_map, "<UNK>");

        // Siapkan hashmap pemetaan Subword -> ID
        var word_map = std.StringHashMap(u32).init(allocator);
        try word_map.ensureTotalCapacity(@intCast(obj.count()));

        it = obj.iterator();
        while (it.next()) |e| {
            const id: u32 = @intCast(e.value_ptr.integer);
            const word_dup = try allocator.dupe(u8, e.key_ptr.*);
            id_map[id] = word_dup;
            try word_map.put(word_dup, id);
        }

        return Tokenizer{
            .arena = arena,
            .id_to_token = id_map,
            .token_to_id = word_map,
            .max_token_length = max_len,
        };
    }

    /// Membersihkan seluruh memori tokenizer dengan sekali sapu
    pub fn deinit(self: *Tokenizer) void {
        self.arena.deinit();
    }

    /// Mengubah ID angka kembali menjadi teks Subword
    pub fn decode(self: *const Tokenizer, id: u32) []const u8 {
        return if (id < self.id_to_token.len) self.id_to_token[id] else "";
    }

    /// 🚀 ALGORITMA GREEDY SUBWORD ENCODER (ZIG 0.15+ COMPLIANT)
    /// Memotong teks mencari pecahan terpanjang di kamus
    pub fn encode(self: *const Tokenizer, allocator: std.mem.Allocator, text: []const u8) ![]u32 {
        // 🚀 ZIG 0.15+: Gunakan .empty, bukan .init(allocator)
        var tokens: std.ArrayList(u32) = .empty;

        // 🚀 ZIG 0.15+: Lempar allocator secara eksplisit ke deinit
        errdefer tokens.deinit(allocator);

        var unk_count: usize = 0;
        var i: usize = 0;

        while (i < text.len) {
            var match_found = false;
            // Coba dari potongan terpanjang yang mungkin, mundur hingga 1 karakter
            var len: usize = @min(self.max_token_length, text.len - i);

            while (len > 0) : (len -= 1) {
                const sub_str = text[i .. i + len];
                if (self.token_to_id.get(sub_str)) |id| {
                    // 🚀 ZIG 0.15+: Masukkan allocator ke dalam append
                    try tokens.append(allocator, id);
                    i += len;
                    match_found = true;
                    break;
                }
            }

            if (!match_found) {
                // Jika karakter sama sekali tidak dikenali (bahkan sebagai 1 huruf), jadikan UNK (ID 1)
                try tokens.append(allocator, 1);
                unk_count += 1;
                i += 1; // Geser 1 byte dan coba lagi
            }
        }

        if (unk_count > 0) {
            std.debug.print("   ⚠️  {d} bytes tidak dapat dikenali (<UNK>)\n", .{unk_count});
        }

        // 🚀 ZIG 0.15+: Lempar allocator ke toOwnedSlice
        return tokens.toOwnedSlice(allocator);
    }
};
