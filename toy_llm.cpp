#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct Tokenizer {
    virtual ~Tokenizer() = default;
    virtual std::vector<int> encode(const std::string& text) const = 0;
    virtual std::string decode(const std::vector<int>& ids) const = 0;
    virtual int vocab_size() const = 0;
};

struct CharTokenizer : Tokenizer {
    std::vector<char> id_to_char;
    std::unordered_map<char, int> char_to_id;

    explicit CharTokenizer(const std::string& corpus) {
        for (char c : corpus) {
            if (char_to_id.find(c) == char_to_id.end()) {
                int id = static_cast<int>(id_to_char.size());
                char_to_id[c] = id;
                id_to_char.push_back(c);
            }
        }
        if (id_to_char.empty()) {
            throw std::runtime_error("Tokenizer corpus must not be empty");
        }
    }

    std::vector<int> encode(const std::string& text) const override {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (char c : text) {
            auto it = char_to_id.find(c);
            if (it == char_to_id.end()) {
                ids.push_back(0);
            } else {
                ids.push_back(it->second);
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const override {
        std::string out;
        out.reserve(ids.size());
        for (int id : ids) {
            if (id < 0 || id >= static_cast<int>(id_to_char.size())) {
                out.push_back('?');
            } else {
                out.push_back(id_to_char[static_cast<size_t>(id)]);
            }
        }
        return out;
    }

    int vocab_size() const override {
        return static_cast<int>(id_to_char.size());
    }
};

struct SubwordTokenizer : Tokenizer {
    std::vector<std::string> id_to_token;
    std::unordered_map<std::string, int> token_to_id;
    std::vector<int> ids_by_descending_token_length;
    int unk_id = 0;

    explicit SubwordTokenizer(const std::vector<std::string>& vocab_tokens) {
        if (vocab_tokens.empty()) {
            throw std::runtime_error("Subword vocab must not be empty");
        }

        id_to_token = vocab_tokens;
        for (size_t i = 0; i < id_to_token.size(); ++i) {
            token_to_id[id_to_token[i]] = static_cast<int>(i);
        }

        auto it = token_to_id.find("<unk>");
        if (it != token_to_id.end()) {
            unk_id = it->second;
        } else {
            unk_id = 0;
        }

        ids_by_descending_token_length.resize(id_to_token.size());
        for (size_t i = 0; i < id_to_token.size(); ++i) {
            ids_by_descending_token_length[i] = static_cast<int>(i);
        }
        std::sort(ids_by_descending_token_length.begin(), ids_by_descending_token_length.end(),
                  [&](int a, int b) {
                      return id_to_token[static_cast<size_t>(a)].size() >
                             id_to_token[static_cast<size_t>(b)].size();
                  });
    }

    static std::string parse_vocab_token_line(const std::string& line) {
        std::string out;
        out.reserve(line.size());
        for (size_t i = 0; i < line.size(); ++i) {
            char c = line[i];
            if (c == '\\' && i + 1 < line.size()) {
                char n = line[i + 1];
                if (n == 'n') {
                    out.push_back('\n');
                    ++i;
                    continue;
                }
                if (n == 't') {
                    out.push_back('\t');
                    ++i;
                    continue;
                }
                if (n == 'r') {
                    out.push_back('\r');
                    ++i;
                    continue;
                }
                if (n == 's') {
                    out.push_back(' ');
                    ++i;
                    continue;
                }
            }
            out.push_back(c);
        }
        return out;
    }

    static std::vector<std::string> load_vocab_file(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open vocab file: " + path);
        }

        std::vector<std::string> tokens;
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) continue;
            tokens.push_back(parse_vocab_token_line(line));
        }

        if (tokens.empty()) {
            throw std::runtime_error("Vocab file has no usable tokens: " + path);
        }
        return tokens;
    }

    std::vector<int> encode(const std::string& text) const override {
        std::vector<int> ids;
        size_t pos = 0;
        while (pos < text.size()) {
            int best_id = -1;
            size_t best_len = 0;

            for (int id : ids_by_descending_token_length) {
                const std::string& tok = id_to_token[static_cast<size_t>(id)];
                if (tok.empty()) continue;
                if (tok.size() <= best_len) break;
                if (pos + tok.size() > text.size()) continue;
                if (text.compare(pos, tok.size(), tok) == 0) {
                    best_id = id;
                    best_len = tok.size();
                }
            }

            if (best_id >= 0) {
                ids.push_back(best_id);
                pos += best_len;
            } else {
                ids.push_back(unk_id);
                ++pos;
            }
        }
        if (ids.empty()) ids.push_back(unk_id);
        return ids;
    }

    std::string decode(const std::vector<int>& ids) const override {
        std::string out;
        for (int id : ids) {
            if (id < 0 || id >= static_cast<int>(id_to_token.size())) {
                out += "<unk>";
            } else {
                out += id_to_token[static_cast<size_t>(id)];
            }
        }
        return out;
    }

    int vocab_size() const override {
        return static_cast<int>(id_to_token.size());
    }
};

[[maybe_unused]] static std::string load_text_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open dataset file: " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

[[maybe_unused]] static std::string load_dataset_from_files(const std::vector<std::string>& file_paths) {
    if (file_paths.empty()) {
        throw std::runtime_error("No dataset files provided");
    }

    std::string corpus;
    for (size_t i = 0; i < file_paths.size(); ++i) {
        std::string part = load_text_file(file_paths[i]);
        if (!part.empty()) {
            corpus += part;
            if (i + 1 < file_paths.size()) {
                corpus.push_back('\n');
            }
        }
    }
    if (corpus.empty()) {
        throw std::runtime_error("Loaded dataset is empty");
    }
    return corpus;
}

static std::pair<std::vector<int>, std::vector<int>> split_train_val_ids(const std::vector<int>& ids,
                                                                          float val_split) {
    if (val_split <= 0.0f || ids.size() < 4) {
        return {ids, {}};
    }
    float clamped = std::max(0.0f, std::min(0.9f, val_split));
    size_t val_n = static_cast<size_t>(static_cast<double>(ids.size()) * static_cast<double>(clamped));
    if (val_n < 2) {
        return {ids, {}};
    }
    if (val_n >= ids.size() - 1) {
        val_n = ids.size() - 2;
    }

    size_t split_idx = ids.size() - val_n;
    std::vector<int> train(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(split_idx));
    std::vector<int> val(ids.begin() + static_cast<std::ptrdiff_t>(split_idx), ids.end());
    return {train, val};
}

static std::string escape_vocab_token_line(const std::string& token) {
    std::string out;
    out.reserve(token.size() * 2);
    for (char c : token) {
        if (c == '\n') {
            out += "\\n";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c == '\r') {
            out += "\\r";
        } else if (c == ' ') {
            out += "\\s";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

static std::vector<std::string> build_vocab_from_corpus_frequency(const std::string& corpus,
                                                                   int target_vocab_size) {
    if (corpus.empty()) {
        throw std::runtime_error("Cannot build vocab from empty corpus");
    }
    int vocab_cap = std::max(8, target_vocab_size);

    std::unordered_map<std::string, int> word_freq;
    std::unordered_map<std::string, int> char_freq;

    std::string current_word;
    current_word.reserve(32);

    auto flush_word = [&]() {
        if (!current_word.empty()) {
            ++word_freq[current_word];
            current_word.clear();
        }
    };

    for (char c : corpus) {
        unsigned char uc = static_cast<unsigned char>(c);
        if (std::isalnum(uc) || c == '\'') {
            current_word.push_back(c);
        } else {
            flush_word();
            ++char_freq[std::string(1, c)];
        }
    }
    flush_word();

    for (char c : corpus) {
        ++char_freq[std::string(1, c)];
    }

    std::vector<std::pair<std::string, int>> words(word_freq.begin(), word_freq.end());
    std::sort(words.begin(), words.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });

    std::vector<std::pair<std::string, int>> chars(char_freq.begin(), char_freq.end());
    std::sort(chars.begin(), chars.end(), [](const auto& a, const auto& b) {
        if (a.second != b.second) return a.second > b.second;
        return a.first < b.first;
    });

    std::vector<std::string> vocab;
    vocab.reserve(static_cast<size_t>(vocab_cap));
    std::unordered_set<std::string> seen;

    auto push_token = [&](const std::string& tok) {
        if (tok.empty()) return;
        if (seen.insert(tok).second) {
            vocab.push_back(tok);
        }
    };

    push_token("<unk>");
    push_token(" ");
    push_token("\n");

    int budget_for_words = std::max(0, vocab_cap - 4);
    for (const auto& kv : words) {
        if (static_cast<int>(vocab.size()) >= budget_for_words) break;
        if (kv.first.size() <= 1) continue;
        push_token(kv.first);
    }

    for (const auto& kv : chars) {
        if (static_cast<int>(vocab.size()) >= vocab_cap) break;
        push_token(kv.first);
    }

    if (vocab.size() < 4) {
        throw std::runtime_error("Failed to build a usable vocab from corpus");
    }

    return vocab;
}

static void write_vocab_file(const std::string& path,
                             const std::vector<std::string>& vocab_tokens) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to write vocab file: " + path);
    }
    for (const std::string& tok : vocab_tokens) {
        out << escape_vocab_token_line(tok) << "\n";
    }
}

struct RunConfigJson {
    std::unordered_map<std::string, std::string> scalars;
    std::unordered_map<std::string, std::vector<std::string>> string_arrays;
};

struct EffectiveRunConfig {
    int num_layers = 1;
    std::vector<std::string> data;
    std::vector<std::string> warnings;
    std::string vocab;
    std::string build_vocab;
    std::string prompt;
    int vocab_size = 256;
    float val_split = 0.0f;
    float temperature = 0.9f;
    int top_k = 5;
    float top_p = 1.0f;
    float rep_penalty = 1.0f;
    int steps = 80;
    std::string resume;
    std::string save_ckpt;
    int epochs = 40;
    int ctx_window = 24;
    float lr = 0.04f;
    std::string optimizer = "sgd";
    int warmup = 0;
    float min_lr_ratio = 0.1f;
    float weight_decay = 0.0f;
    int batch_size = 1;
    float grad_clip = 0.0f;
    int report_every = 10;
    bool report_lr = false;
    bool report_grad = false;
    std::string save_config;
    bool dry_run = false;
    bool strict_config = false;
};

[[maybe_unused]] static bool cfg_is_true_like(const std::string& value) {
    return value == "true" || value == "1";
}

static bool is_known_config_scalar_key(const std::string& key) {
    static const std::unordered_set<std::string> keys = {
        "num_layers", "vocab", "build_vocab", "prompt", "vocab_size",
        "val_split", "temperature", "top_k", "top_p", "rep_penalty", "steps",
        "resume", "save_ckpt", "save_config", "epochs", "ctx_window", "lr",
        "optimizer", "warmup", "min_lr_ratio", "weight_decay", "batch_size",
        "grad_clip", "report_every", "report_lr", "report_grad", "dry_run",
        "strict_config"
    };
    return keys.find(key) != keys.end();
}

static bool is_known_config_array_key(const std::string& key) {
    return key == "data" || key == "warnings";
}

static std::vector<std::string> all_known_config_keys() {
    return {
        "num_layers", "vocab", "build_vocab", "prompt", "vocab_size",
        "val_split", "temperature", "top_k", "top_p", "rep_penalty", "steps",
        "resume", "save_ckpt", "save_config", "epochs", "ctx_window", "lr",
        "optimizer", "warmup", "min_lr_ratio", "weight_decay", "batch_size",
        "grad_clip", "report_every", "report_lr", "report_grad", "dry_run",
        "strict_config", "data"
    };
}

static int levenshtein_distance(const std::string& a, const std::string& b) {
    const size_t n = a.size();
    const size_t m = b.size();
    std::vector<int> prev(m + 1, 0);
    std::vector<int> cur(m + 1, 0);
    for (size_t j = 0; j <= m; ++j) prev[j] = static_cast<int>(j);

    for (size_t i = 1; i <= n; ++i) {
        cur[0] = static_cast<int>(i);
        for (size_t j = 1; j <= m; ++j) {
            int cost = (a[i - 1] == b[j - 1]) ? 0 : 1;
            int del = prev[j] + 1;
            int ins = cur[j - 1] + 1;
            int sub = prev[j - 1] + cost;
            cur[j] = std::min(del, std::min(ins, sub));
        }
        prev.swap(cur);
    }
    return prev[m];
}

static std::vector<std::string> suggest_config_keys(const std::string& bad_key,
                                                    size_t max_count = 3) {
    const std::vector<std::string> known = all_known_config_keys();
    std::vector<std::pair<int, std::string>> ranked;
    ranked.reserve(known.size());
    for (const std::string& k : known) {
        int d = levenshtein_distance(bad_key, k);
        ranked.push_back({d, k});
    }

    if (ranked.empty()) return {};
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        if (a.first != b.first) return a.first < b.first;
        return a.second < b.second;
    });

    int best_dist = ranked.front().first;

    int max_reasonable = 3;
    if (static_cast<int>(bad_key.size()) >= 10) {
        max_reasonable = 4;
    }
    if (best_dist > max_reasonable) return {};

    std::vector<std::string> out;
    for (const auto& item : ranked) {
        if (item.first > max_reasonable) break;
        if (item.first > best_dist + 1) break;
        out.push_back(item.second);
        if (out.size() >= max_count) break;
    }
    return out;
}

static std::string format_config_key_suggestions(const std::vector<std::string>& suggestions) {
    if (suggestions.empty()) return std::string();
    if (suggestions.size() == 1) {
        return ". Did you mean '" + suggestions[0] + "'?";
    }

    std::string out = ". Did you mean one of: ";
    for (size_t i = 0; i < suggestions.size(); ++i) {
        if (i > 0) out += ", ";
        out += "'" + suggestions[i] + "'";
    }
    out += "?";
    return out;
}

static std::vector<std::string> collect_non_strict_config_warnings(const RunConfigJson& cfg) {
    std::vector<std::string> warnings;
    for (const auto& kv : cfg.scalars) {
        if (is_known_config_scalar_key(kv.first)) continue;
        std::string msg = "Unknown config key (ignored): " + kv.first;
        msg += format_config_key_suggestions(suggest_config_keys(kv.first));
        warnings.push_back(msg);
    }
    for (const auto& kv : cfg.string_arrays) {
        if (is_known_config_array_key(kv.first)) continue;
        std::string msg = "Unknown config key (ignored): " + kv.first;
        msg += format_config_key_suggestions(suggest_config_keys(kv.first));
        warnings.push_back(msg);
    }
    return warnings;
}

static void validate_run_config_keys(const RunConfigJson& cfg, bool strict_mode) {
    if (!strict_mode) return;

    for (const auto& kv : cfg.scalars) {
        if (!is_known_config_scalar_key(kv.first)) {
            std::string msg = "Unknown config key in strict mode: " + kv.first;
            msg += format_config_key_suggestions(suggest_config_keys(kv.first));
            throw std::runtime_error(msg);
        }
    }
    for (const auto& kv : cfg.string_arrays) {
        if (!is_known_config_array_key(kv.first)) {
            std::string msg = "Unknown config key in strict mode: " + kv.first;
            msg += format_config_key_suggestions(suggest_config_keys(kv.first));
            throw std::runtime_error(msg);
        }
    }
}

static void cfg_skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) {
        ++i;
    }
}

static std::string cfg_parse_json_string(const std::string& s, size_t& i) {
    if (i >= s.size() || s[i] != '"') {
        throw std::runtime_error("Config JSON parse error: expected string");
    }
    ++i;
    std::string out;
    while (i < s.size()) {
        char c = s[i++];
        if (c == '"') {
            return out;
        }
        if (c == '\\') {
            if (i >= s.size()) {
                throw std::runtime_error("Config JSON parse error: bad escape");
            }
            char e = s[i++];
            if (e == 'n') out.push_back('\n');
            else if (e == 't') out.push_back('\t');
            else if (e == 'r') out.push_back('\r');
            else if (e == '"') out.push_back('"');
            else if (e == '\\') out.push_back('\\');
            else out.push_back(e);
            continue;
        }
        out.push_back(c);
    }
    throw std::runtime_error("Config JSON parse error: unterminated string");
}

static std::string cfg_parse_json_literal(const std::string& s, size_t& i) {
    size_t start = i;
    while (i < s.size()) {
        char c = s[i];
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '+' || c == '.' || c == '_') {
            ++i;
        } else {
            break;
        }
    }
    if (i == start) {
        throw std::runtime_error("Config JSON parse error: expected literal");
    }
    return s.substr(start, i - start);
}

static std::vector<std::string> cfg_parse_string_array(const std::string& s, size_t& i) {
    if (i >= s.size() || s[i] != '[') {
        throw std::runtime_error("Config JSON parse error: expected array");
    }
    ++i;
    std::vector<std::string> out;
    cfg_skip_ws(s, i);
    if (i < s.size() && s[i] == ']') {
        ++i;
        return out;
    }
    while (i < s.size()) {
        cfg_skip_ws(s, i);
        out.push_back(cfg_parse_json_string(s, i));
        cfg_skip_ws(s, i);
        if (i >= s.size()) break;
        if (s[i] == ',') {
            ++i;
            continue;
        }
        if (s[i] == ']') {
            ++i;
            return out;
        }
        throw std::runtime_error("Config JSON parse error: bad array separator");
    }
    throw std::runtime_error("Config JSON parse error: unterminated array");
}

static RunConfigJson load_run_config_json(const std::string& path) {
    std::string raw = load_text_file(path);
    size_t i = 0;
    cfg_skip_ws(raw, i);
    if (i >= raw.size() || raw[i] != '{') {
        throw std::runtime_error("Config JSON parse error: root must be object");
    }
    ++i;

    RunConfigJson cfg;
    while (i < raw.size()) {
        cfg_skip_ws(raw, i);
        if (i < raw.size() && raw[i] == '}') {
            ++i;
            break;
        }

        std::string key = cfg_parse_json_string(raw, i);
        cfg_skip_ws(raw, i);
        if (i >= raw.size() || raw[i] != ':') {
            throw std::runtime_error("Config JSON parse error: expected ':'");
        }
        ++i;
        cfg_skip_ws(raw, i);

        if (i >= raw.size()) {
            throw std::runtime_error("Config JSON parse error: missing value");
        }

        if (raw[i] == '"') {
            cfg.scalars[key] = cfg_parse_json_string(raw, i);
        } else if (raw[i] == '[') {
            cfg.string_arrays[key] = cfg_parse_string_array(raw, i);
        } else {
            cfg.scalars[key] = cfg_parse_json_literal(raw, i);
        }

        cfg_skip_ws(raw, i);
        if (i < raw.size() && raw[i] == ',') {
            ++i;
            continue;
        }
        if (i < raw.size() && raw[i] == '}') {
            ++i;
            break;
        }
    }

    cfg_skip_ws(raw, i);
    if (i != raw.size()) {
        throw std::runtime_error("Config JSON parse error: trailing content");
    }
    return cfg;
}

static std::string json_escape_string(const std::string& in) {
    std::string out;
    out.reserve(in.size() + in.size() / 4 + 8);
    for (char c : in) {
        if (c == '"') out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\t') out += "\\t";
        else if (c == '\r') out += "\\r";
        else out.push_back(c);
    }
    return out;
}

static std::string float_to_json(float v) {
    std::ostringstream oss;
    oss << v;
    return oss.str();
}

static std::string effective_run_config_to_json_string(const EffectiveRunConfig& cfg) {
    std::ostringstream out;
    out << "{\n";
    out << "  \"num_layers\": " << cfg.num_layers << ",\n";
    out << "  \"data\": [";
    for (size_t i = 0; i < cfg.data.size(); ++i) {
        if (i > 0) out << ", ";
        out << "\"" << json_escape_string(cfg.data[i]) << "\"";
    }
    out << "],\n";
    out << "  \"warnings\": [";
    for (size_t i = 0; i < cfg.warnings.size(); ++i) {
        if (i > 0) out << ", ";
        out << "\"" << json_escape_string(cfg.warnings[i]) << "\"";
    }
    out << "],\n";
    out << "  \"vocab\": \"" << json_escape_string(cfg.vocab) << "\",\n";
    out << "  \"build_vocab\": \"" << json_escape_string(cfg.build_vocab) << "\",\n";
    out << "  \"prompt\": \"" << json_escape_string(cfg.prompt) << "\",\n";
    out << "  \"vocab_size\": " << cfg.vocab_size << ",\n";
    out << "  \"val_split\": " << float_to_json(cfg.val_split) << ",\n";
    out << "  \"temperature\": " << float_to_json(cfg.temperature) << ",\n";
    out << "  \"top_k\": " << cfg.top_k << ",\n";
    out << "  \"top_p\": " << float_to_json(cfg.top_p) << ",\n";
    out << "  \"rep_penalty\": " << float_to_json(cfg.rep_penalty) << ",\n";
    out << "  \"steps\": " << cfg.steps << ",\n";
    out << "  \"resume\": \"" << json_escape_string(cfg.resume) << "\",\n";
    out << "  \"save_ckpt\": \"" << json_escape_string(cfg.save_ckpt) << "\",\n";
    out << "  \"epochs\": " << cfg.epochs << ",\n";
    out << "  \"ctx_window\": " << cfg.ctx_window << ",\n";
    out << "  \"lr\": " << float_to_json(cfg.lr) << ",\n";
    out << "  \"optimizer\": \"" << json_escape_string(cfg.optimizer) << "\",\n";
    out << "  \"warmup\": " << cfg.warmup << ",\n";
    out << "  \"min_lr_ratio\": " << float_to_json(cfg.min_lr_ratio) << ",\n";
    out << "  \"weight_decay\": " << float_to_json(cfg.weight_decay) << ",\n";
    out << "  \"batch_size\": " << cfg.batch_size << ",\n";
    out << "  \"grad_clip\": " << float_to_json(cfg.grad_clip) << ",\n";
    out << "  \"report_every\": " << cfg.report_every << ",\n";
    out << "  \"report_lr\": " << (cfg.report_lr ? "true" : "false") << ",\n";
    out << "  \"report_grad\": " << (cfg.report_grad ? "true" : "false") << ",\n";
    out << "  \"save_config\": \"" << json_escape_string(cfg.save_config) << "\",\n";
    out << "  \"dry_run\": " << (cfg.dry_run ? "true" : "false") << ",\n";
    out << "  \"strict_config\": " << (cfg.strict_config ? "true" : "false") << "\n";
    out << "}\n";
    return out.str();
}

static void write_effective_run_config_json(const std::string& path,
                                            const EffectiveRunConfig& cfg) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output config file: " + path);
    }
    out << effective_run_config_to_json_string(cfg);
}

struct TinyTransformer {
    int vocab;
    int d_model;
    int d_ff;
    int max_seq;
    int num_layers;
    bool use_layernorm;
    int num_heads;

    // Embeddings.
    std::vector<float> token_emb; // [vocab, d_model]
    std::vector<float> pos_emb;   // [max_seq, d_model]

    // Single-head attention projections.
    std::vector<float> wq; // [d_model, d_model]
    std::vector<float> wk;
    std::vector<float> wv;
    std::vector<float> wo;

    // Feed-forward.
    std::vector<float> w1; // [d_model, d_ff]
    std::vector<float> w2; // [d_ff, d_model]

    // Output projection.
    std::vector<float> out_proj; // [d_model, vocab]
    std::vector<float> out_bias; // [vocab]

    // Persistent optimizer/training state for resume support.
    int32_t train_global_step = 0;
    std::vector<float> adam_m_out_bias;
    std::vector<float> adam_v_out_bias;
    std::vector<float> adam_m_out_proj;
    std::vector<float> adam_v_out_proj;
    std::vector<float> adam_m_w1;
    std::vector<float> adam_v_w1;
    std::vector<float> adam_m_w2;
    std::vector<float> adam_v_w2;
    std::vector<float> adam_m_wo;
    std::vector<float> adam_v_wo;
    std::vector<float> adam_m_wq;
    std::vector<float> adam_v_wq;
    std::vector<float> adam_m_wk;
    std::vector<float> adam_v_wk;
    std::vector<float> adam_m_wv;
    std::vector<float> adam_v_wv;
    std::vector<float> adam_m_tok;
    std::vector<float> adam_v_tok;
    std::vector<float> adam_m_pos;
    std::vector<float> adam_v_pos;

    struct LayerCache {
        int seq = 0;                        // sequence length
        std::vector<float> x_in;         // [seq, d_model] input to this layer
        std::vector<float> q;            // [seq, d_model]
        std::vector<float> k;            // [seq, d_model]
        std::vector<float> v;            // [seq, d_model]
        std::vector<float> attn_out;     // [seq, d_model]
        std::vector<float> h1;           // [seq, d_model] (after attention + residual)
        std::vector<float> ff1_lin;      // [seq, d_ff]
        std::vector<float> ff1_act;      // [seq, d_ff]
        std::vector<float> h2;           // [seq, d_model] (after FFN + residual)
        std::vector<float> attn_weights; // [seq, seq]
        std::vector<float> attn_input;   // [seq, d_model] (pre-norm attention input)
        std::vector<float> ffn_input;    // [seq, d_model] (pre-norm FFN input)
    };

    struct ForwardCache {
        int seq = 0;
        std::vector<LayerCache> layers;  // [num_layers]
        std::vector<float> logits_all;   // [seq, vocab] final output logits
        
        // Convenience accessors for backward compat (single-layer access)
        const std::vector<float>& x() const { return layers.back().x_in; }
        const std::vector<float>& q() const { return layers.back().q; }
        const std::vector<float>& k() const { return layers.back().k; }
        const std::vector<float>& v() const { return layers.back().v; }
        const std::vector<float>& attn_out() const { return layers.back().attn_out; }
        const std::vector<float>& h1() const { return layers.back().h1; }
        const std::vector<float>& h2() const { return layers.back().h2; }
        const std::vector<float>& ff1_lin() const { return layers.back().ff1_lin; }
        const std::vector<float>& ff1_act() const { return layers.back().ff1_act; }
        const std::vector<float>& attn_weights() const { return layers.back().attn_weights; }
        const std::vector<float>& attn_input() const { return layers.back().attn_input; }
        const std::vector<float>& ffn_input() const { return layers.back().ffn_input; }
    };

    struct OutputHeadBackward {
        float loss = 0.0f;
        std::vector<float> grad_out_proj; // [d_model, vocab]
        std::vector<float> grad_out_bias; // [vocab]
        std::vector<float> grad_h2;       // [seq, d_model]
    };

    struct FFNBackward {
        std::vector<float> grad_w2; // [d_ff, d_model]
        std::vector<float> grad_w1; // [d_model, d_ff]
        std::vector<float> grad_h1; // [seq, d_model]
    };

    struct AttentionProjBackward {
        std::vector<float> grad_wo;       // [d_model, d_model]
        std::vector<float> grad_attn_out; // [seq, d_model]
    };

    struct AttentionCoreBackward {
        std::vector<float> grad_wq; // [d_model, d_model]
        std::vector<float> grad_wk; // [d_model, d_model]
        std::vector<float> grad_wv; // [d_model, d_model]
        std::vector<float> grad_x;  // [seq, d_model]
    };

    struct EmbeddingBackward {
        std::vector<float> grad_token_emb; // [vocab, d_model]
        std::vector<float> grad_pos_emb;   // [max_seq, d_model]
    };

    explicit TinyTransformer(int vocab_size,
                             int d_model_ = 24,
                             int d_ff_ = 48,
                             int max_seq_ = 64,
                                                         uint32_t seed = 1337,
                                                         int num_layers_ = 1,
                                                         bool use_layernorm_ = false,
                                                         int num_heads_ = 1)
        : vocab(vocab_size),
          d_model(d_model_),
          d_ff(d_ff_),
          max_seq(max_seq_),
                    num_layers(std::max(1, num_layers_)),
                    use_layernorm(use_layernorm_),
                    num_heads(std::max(1, num_heads_)),
          token_emb(static_cast<size_t>(vocab_size * d_model_)),
          pos_emb(static_cast<size_t>(max_seq_ * d_model_)),
          wq(static_cast<size_t>(d_model_ * d_model_)),
          wk(static_cast<size_t>(d_model_ * d_model_)),
          wv(static_cast<size_t>(d_model_ * d_model_)),
          wo(static_cast<size_t>(d_model_ * d_model_)),
          w1(static_cast<size_t>(d_model_ * d_ff_)),
          w2(static_cast<size_t>(d_ff_ * d_model_)),
          out_proj(static_cast<size_t>(d_model_ * vocab_size)),
          out_bias(static_cast<size_t>(vocab_size), 0.0f) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> nd(0.0f, 0.08f);

        auto init = [&](std::vector<float>& vec) {
            for (float& x : vec) x = nd(rng);
        };

        init(token_emb);
        init(pos_emb);
        init(wq);
        init(wk);
        init(wv);
        init(wo);
        init(w1);
        init(w2);
        init(out_proj);
    }

    void clear_optimizer_state() {
        train_global_step = 0;
        adam_m_out_bias.clear();
        adam_v_out_bias.clear();
        adam_m_out_proj.clear();
        adam_v_out_proj.clear();
        adam_m_w1.clear();
        adam_v_w1.clear();
        adam_m_w2.clear();
        adam_v_w2.clear();
        adam_m_wo.clear();
        adam_v_wo.clear();
        adam_m_wq.clear();
        adam_v_wq.clear();
        adam_m_wk.clear();
        adam_v_wk.clear();
        adam_m_wv.clear();
        adam_v_wv.clear();
        adam_m_tok.clear();
        adam_v_tok.clear();
        adam_m_pos.clear();
        adam_v_pos.clear();
    }

    static void layernorm_rows(const std::vector<float>& in,
                               int rows,
                               int cols,
                               std::vector<float>& out,
                               float eps = 1e-5f) {
        out.assign(in.size(), 0.0f);
        for (int r = 0; r < rows; ++r) {
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) {
                mean += in[static_cast<size_t>(r * cols + c)];
            }
            mean /= static_cast<float>(cols);

            float var = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float d = in[static_cast<size_t>(r * cols + c)] - mean;
                var += d * d;
            }
            var /= static_cast<float>(cols);
            float inv_std = 1.0f / std::sqrt(var + eps);

            for (int c = 0; c < cols; ++c) {
                out[static_cast<size_t>(r * cols + c)] =
                    (in[static_cast<size_t>(r * cols + c)] - mean) * inv_std;
            }
        }
    }

    static void layernorm_rows_backward(const std::vector<float>& in,
                                        const std::vector<float>& grad_out,
                                        int rows,
                                        int cols,
                                        std::vector<float>& grad_in,
                                        float eps = 1e-5f) {
        grad_in.assign(in.size(), 0.0f);
        for (int r = 0; r < rows; ++r) {
            float mean = 0.0f;
            for (int c = 0; c < cols; ++c) {
                mean += in[static_cast<size_t>(r * cols + c)];
            }
            mean /= static_cast<float>(cols);

            float var = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float d = in[static_cast<size_t>(r * cols + c)] - mean;
                var += d * d;
            }
            var /= static_cast<float>(cols);
            float std_val = std::sqrt(var + eps);
            float inv_std = 1.0f / std_val;

            float sum_grad = 0.0f;
            float sum_grad_x = 0.0f;
            for (int c = 0; c < cols; ++c) {
                float x_norm = (in[static_cast<size_t>(r * cols + c)] - mean) * inv_std;
                sum_grad += grad_out[static_cast<size_t>(r * cols + c)];
                sum_grad_x += grad_out[static_cast<size_t>(r * cols + c)] * x_norm;
            }

            float c_cols = static_cast<float>(cols);
            for (int c = 0; c < cols; ++c) {
                float x_norm = (in[static_cast<size_t>(r * cols + c)] - mean) * inv_std;
                float grad_x_norm = grad_out[static_cast<size_t>(r * cols + c)] -
                                    (sum_grad / c_cols) -
                                    x_norm * (sum_grad_x / c_cols);
                grad_in[static_cast<size_t>(r * cols + c)] = grad_x_norm * inv_std;
            }
        }
    }

    template <typename T>
    static bool write_pod(std::ofstream& out, const T& value) {
        out.write(reinterpret_cast<const char*>(&value), static_cast<std::streamsize>(sizeof(T)));
        return out.good();
    }

    template <typename T>
    static bool read_pod(std::ifstream& in, T& value) {
        in.read(reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)));
        return in.good();
    }

    static bool write_vector(std::ofstream& out, const std::vector<float>& v) {
        uint64_t sz = static_cast<uint64_t>(v.size());
        if (!write_pod(out, sz)) return false;
        if (sz == 0) return true;
        out.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(sizeof(float) * v.size()));
        return out.good();
    }

    static bool read_vector(std::ifstream& in, std::vector<float>& v, size_t expected_size) {
        uint64_t sz = 0;
        if (!read_pod(in, sz)) return false;
        if (sz != static_cast<uint64_t>(expected_size)) return false;
        v.resize(expected_size);
        if (expected_size == 0) return true;
        in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(sizeof(float) * v.size()));
        return in.good();
    }

    bool save_checkpoint(const std::string& path) const {
        std::ofstream out(path, std::ios::binary);
        if (!out.is_open()) return false;

        const char magic[8] = {'T', 'L', 'L', 'M', 'C', 'K', '2', '\0'};
        out.write(magic, 8);
        if (!out.good()) return false;

        int32_t v_vocab = vocab;
        int32_t v_d_model = d_model;
        int32_t v_d_ff = d_ff;
        int32_t v_max_seq = max_seq;
        int32_t v_num_layers = num_layers;
        int32_t v_use_layernorm = use_layernorm ? 1 : 0;
        int32_t v_num_heads = num_heads;
        if (!write_pod(out, v_vocab) || !write_pod(out, v_d_model) ||
            !write_pod(out, v_d_ff) || !write_pod(out, v_max_seq) ||
            !write_pod(out, v_num_layers) || !write_pod(out, v_use_layernorm) ||
            !write_pod(out, v_num_heads)) {
            return false;
        }

         if (!(write_vector(out, token_emb) &&
            write_vector(out, pos_emb) &&
            write_vector(out, wq) &&
            write_vector(out, wk) &&
            write_vector(out, wv) &&
            write_vector(out, wo) &&
            write_vector(out, w1) &&
            write_vector(out, w2) &&
            write_vector(out, out_proj) &&
            write_vector(out, out_bias))) {
             return false;
         }

        bool has_optimizer_state =
            adam_m_out_bias.size() == out_bias.size() && adam_v_out_bias.size() == out_bias.size() &&
            adam_m_out_proj.size() == out_proj.size() && adam_v_out_proj.size() == out_proj.size() &&
            adam_m_w1.size() == w1.size() && adam_v_w1.size() == w1.size() &&
            adam_m_w2.size() == w2.size() && adam_v_w2.size() == w2.size() &&
            adam_m_wo.size() == wo.size() && adam_v_wo.size() == wo.size() &&
            adam_m_wq.size() == wq.size() && adam_v_wq.size() == wq.size() &&
            adam_m_wk.size() == wk.size() && adam_v_wk.size() == wk.size() &&
            adam_m_wv.size() == wv.size() && adam_v_wv.size() == wv.size() &&
            adam_m_tok.size() == token_emb.size() && adam_v_tok.size() == token_emb.size() &&
            adam_m_pos.size() == pos_emb.size() && adam_v_pos.size() == pos_emb.size();
        int32_t has_optimizer_state_i32 = has_optimizer_state ? 1 : 0;

        if (!write_pod(out, train_global_step) || !write_pod(out, has_optimizer_state_i32)) {
            return false;
        }

        if (!has_optimizer_state) {
            return true;
        }

        return write_vector(out, adam_m_out_bias) && write_vector(out, adam_v_out_bias) &&
               write_vector(out, adam_m_out_proj) && write_vector(out, adam_v_out_proj) &&
               write_vector(out, adam_m_w1) && write_vector(out, adam_v_w1) &&
               write_vector(out, adam_m_w2) && write_vector(out, adam_v_w2) &&
               write_vector(out, adam_m_wo) && write_vector(out, adam_v_wo) &&
               write_vector(out, adam_m_wq) && write_vector(out, adam_v_wq) &&
               write_vector(out, adam_m_wk) && write_vector(out, adam_v_wk) &&
               write_vector(out, adam_m_wv) && write_vector(out, adam_v_wv) &&
               write_vector(out, adam_m_tok) && write_vector(out, adam_v_tok) &&
               write_vector(out, adam_m_pos) && write_vector(out, adam_v_pos);
    }

    bool load_checkpoint(const std::string& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) return false;

        char magic[8] = {0};
        in.read(magic, 8);
        if (!in.good()) return false;
        const char expected_v1[8] = {'T', 'L', 'L', 'M', 'C', 'K', '1', '\0'};
        const char expected_v2[8] = {'T', 'L', 'L', 'M', 'C', 'K', '2', '\0'};
        bool is_v1 = true;
        bool is_v2 = true;
        for (int i = 0; i < 8; ++i) {
            if (magic[i] != expected_v1[i]) is_v1 = false;
            if (magic[i] != expected_v2[i]) is_v2 = false;
        }
        if (!is_v1 && !is_v2) return false;

        int32_t v_vocab = 0;
        int32_t v_d_model = 0;
        int32_t v_d_ff = 0;
        int32_t v_max_seq = 0;
        int32_t v_num_layers = 0;
        int32_t v_use_layernorm = 0;
        int32_t v_num_heads = 0;
        if (!read_pod(in, v_vocab) || !read_pod(in, v_d_model) ||
            !read_pod(in, v_d_ff) || !read_pod(in, v_max_seq) ||
            !read_pod(in, v_num_layers) || !read_pod(in, v_use_layernorm) ||
            !read_pod(in, v_num_heads)) {
            return false;
        }

        if (v_vocab != vocab || v_d_model != d_model || v_d_ff != d_ff ||
            v_max_seq != max_seq || v_num_layers != num_layers ||
            v_use_layernorm != (use_layernorm ? 1 : 0) ||
            v_num_heads != num_heads) {
            return false;
        }

                if (!(read_vector(in, token_emb, token_emb.size()) &&
                            read_vector(in, pos_emb, pos_emb.size()) &&
                            read_vector(in, wq, wq.size()) &&
                            read_vector(in, wk, wk.size()) &&
                            read_vector(in, wv, wv.size()) &&
                            read_vector(in, wo, wo.size()) &&
                            read_vector(in, w1, w1.size()) &&
                            read_vector(in, w2, w2.size()) &&
                            read_vector(in, out_proj, out_proj.size()) &&
                            read_vector(in, out_bias, out_bias.size()))) {
                        return false;
                }

                if (is_v1) {
                        clear_optimizer_state();
                        return true;
                }

                int32_t has_optimizer_state_i32 = 0;
                if (!read_pod(in, train_global_step) || !read_pod(in, has_optimizer_state_i32)) return false;

                if (has_optimizer_state_i32 == 0) {
                    adam_m_out_bias.clear();
                    adam_v_out_bias.clear();
                    adam_m_out_proj.clear();
                    adam_v_out_proj.clear();
                    adam_m_w1.clear();
                    adam_v_w1.clear();
                    adam_m_w2.clear();
                    adam_v_w2.clear();
                    adam_m_wo.clear();
                    adam_v_wo.clear();
                    adam_m_wq.clear();
                    adam_v_wq.clear();
                    adam_m_wk.clear();
                    adam_v_wk.clear();
                    adam_m_wv.clear();
                    adam_v_wv.clear();
                    adam_m_tok.clear();
                    adam_v_tok.clear();
                    adam_m_pos.clear();
                    adam_v_pos.clear();
                    return true;
                }

                if (!(read_vector(in, adam_m_out_bias, out_bias.size()) && read_vector(in, adam_v_out_bias, out_bias.size()) &&
                      read_vector(in, adam_m_out_proj, out_proj.size()) && read_vector(in, adam_v_out_proj, out_proj.size()) &&
                      read_vector(in, adam_m_w1, w1.size()) && read_vector(in, adam_v_w1, w1.size()) &&
                      read_vector(in, adam_m_w2, w2.size()) && read_vector(in, adam_v_w2, w2.size()) &&
                      read_vector(in, adam_m_wo, wo.size()) && read_vector(in, adam_v_wo, wo.size()) &&
                      read_vector(in, adam_m_wq, wq.size()) && read_vector(in, adam_v_wq, wq.size()) &&
                      read_vector(in, adam_m_wk, wk.size()) && read_vector(in, adam_v_wk, wk.size()) &&
                      read_vector(in, adam_m_wv, wv.size()) && read_vector(in, adam_v_wv, wv.size()) &&
                      read_vector(in, adam_m_tok, token_emb.size()) && read_vector(in, adam_v_tok, token_emb.size()) &&
                      read_vector(in, adam_m_pos, pos_emb.size()) && read_vector(in, adam_v_pos, pos_emb.size()))) {
                    return false;
                }

                return true;
    }

    static float dot_row(const std::vector<float>& a, int a_row, int a_cols,
                         const std::vector<float>& b, int b_row, int b_cols) {
        float s = 0.0f;
        for (int i = 0; i < a_cols; ++i) {
            s += a[static_cast<size_t>(a_row * a_cols + i)] * b[static_cast<size_t>(b_row * b_cols + i)];
        }
        return s;
    }

    static std::vector<float> matmul(const std::vector<float>& a, int a_rows, int a_cols,
                                     const std::vector<float>& b, int b_cols) {
        // b is [a_cols, b_cols]
        std::vector<float> out(static_cast<size_t>(a_rows * b_cols), 0.0f);
        for (int r = 0; r < a_rows; ++r) {
            for (int c = 0; c < b_cols; ++c) {
                float s = 0.0f;
                for (int k = 0; k < a_cols; ++k) {
                    s += a[static_cast<size_t>(r * a_cols + k)] * b[static_cast<size_t>(k * b_cols + c)];
                }
                out[static_cast<size_t>(r * b_cols + c)] = s;
            }
        }
        return out;
    }

    ForwardCache forward_cache(const std::vector<int>& ids) const {
        const int seq = static_cast<int>(ids.size());
        if (seq <= 0 || seq > max_seq) {
            throw std::runtime_error("Input sequence length is invalid for TinyTransformer");
        }

        ForwardCache cache;
        cache.seq = seq;
        cache.layers.resize(static_cast<size_t>(num_layers));

        // 1) Embedding + positional encoding. X: [seq, d_model]
        std::vector<float> x_cur(static_cast<size_t>(seq * d_model), 0.0f);
        for (int t = 0; t < seq; ++t) {
            int tok = ids[static_cast<size_t>(t)];
            for (int i = 0; i < d_model; ++i) {
                x_cur[static_cast<size_t>(t * d_model + i)] =
                    token_emb[static_cast<size_t>(tok * d_model + i)] +
                    pos_emb[static_cast<size_t>(t * d_model + i)];
            }
        }

        if (d_model % num_heads != 0) {
            throw std::runtime_error("d_model must be divisible by num_heads");
        }
        const int head_dim = d_model / num_heads;

        for (int layer = 0; layer < num_layers; ++layer) {
            LayerCache& layer_cache = cache.layers[static_cast<size_t>(layer)];
            layer_cache.seq = seq;
            layer_cache.x_in = x_cur;

            std::vector<float> attn_input;
            if (use_layernorm) {
                layernorm_rows(x_cur, seq, d_model, attn_input);
            } else {
                attn_input = x_cur;
            }
            layer_cache.attn_input = attn_input;

            layer_cache.q = matmul(attn_input, seq, d_model, wq, d_model); // [seq, d]
            layer_cache.k = matmul(attn_input, seq, d_model, wk, d_model); // [seq, d]
            layer_cache.v = matmul(attn_input, seq, d_model, wv, d_model); // [seq, d]

            layer_cache.attn_out.assign(static_cast<size_t>(seq * d_model), 0.0f);
            layer_cache.attn_weights.assign(static_cast<size_t>(seq * seq), 0.0f);

            for (int h = 0; h < num_heads; ++h) {
                const int d0 = h * head_dim;
                for (int t = 0; t < seq; ++t) {
                    std::vector<float> scores(static_cast<size_t>(t + 1), 0.0f);
                    float max_score = -1e30f;
                    for (int j = 0; j <= t; ++j) {
                        float s = 0.0f;
                        for (int d = 0; d < head_dim; ++d) {
                            s += layer_cache.q[static_cast<size_t>(t * d_model + d0 + d)] *
                                 layer_cache.k[static_cast<size_t>(j * d_model + d0 + d)];
                        }
                        s *= (1.0f / std::sqrt(static_cast<float>(head_dim)));
                        scores[static_cast<size_t>(j)] = s;
                        max_score = std::max(max_score, s);
                    }

                    float exp_sum = 0.0f;
                    for (int j = 0; j <= t; ++j) {
                        float e = std::exp(scores[static_cast<size_t>(j)] - max_score);
                        scores[static_cast<size_t>(j)] = e;
                        exp_sum += e;
                    }

                    for (int j = 0; j <= t; ++j) {
                        float a = scores[static_cast<size_t>(j)] / exp_sum;
                        layer_cache.attn_weights[static_cast<size_t>(t * seq + j)] += a / static_cast<float>(num_heads);
                        for (int d = 0; d < head_dim; ++d) {
                            layer_cache.attn_out[static_cast<size_t>(t * d_model + d0 + d)] +=
                                a * layer_cache.v[static_cast<size_t>(j * d_model + d0 + d)];
                        }
                    }
                }
            }

            std::vector<float> attn_proj = matmul(layer_cache.attn_out, seq, d_model, wo, d_model);

            // Residual 1.
            layer_cache.h1.assign(static_cast<size_t>(seq * d_model), 0.0f);
            for (size_t i = 0; i < layer_cache.h1.size(); ++i) {
                layer_cache.h1[i] = x_cur[i] + attn_proj[i];
            }

            std::vector<float> ffn_input;
            if (use_layernorm) {
                layernorm_rows(layer_cache.h1, seq, d_model, ffn_input);
            } else {
                ffn_input = layer_cache.h1;
            }
            layer_cache.ffn_input = ffn_input;

            layer_cache.ff1_lin = matmul(ffn_input, seq, d_model, w1, d_ff);
            layer_cache.ff1_act = layer_cache.ff1_lin;
            for (float& z : layer_cache.ff1_act) {
                z = std::max(0.0f, z);
            }
            std::vector<float> ff2 = matmul(layer_cache.ff1_act, seq, d_ff, w2, d_model);

            layer_cache.h2.assign(static_cast<size_t>(seq * d_model), 0.0f);
            for (size_t i = 0; i < layer_cache.h2.size(); ++i) {
                layer_cache.h2[i] = layer_cache.h1[i] + ff2[i];
            }

            x_cur = layer_cache.h2;
        }

        // 4) Output logits for all positions.
        cache.logits_all.assign(static_cast<size_t>(seq * vocab), 0.0f);
        std::vector<float>& final_h2 = cache.layers.back().h2;
        for (int t = 0; t < seq; ++t) {
            for (int tok = 0; tok < vocab; ++tok) {
                float s = out_bias[static_cast<size_t>(tok)];
                for (int d = 0; d < d_model; ++d) {
                    s += final_h2[static_cast<size_t>(t * d_model + d)] *
                         out_proj[static_cast<size_t>(d * vocab + tok)];
                }
                cache.logits_all[static_cast<size_t>(t * vocab + tok)] = s;
            }
        }

        return cache;
    }

    std::vector<float> forward(const std::vector<int>& ids) const {
        ForwardCache cache = forward_cache(ids);
        std::vector<float> logits(static_cast<size_t>(vocab), 0.0f);
        const int last = cache.seq - 1;
        for (int tok = 0; tok < vocab; ++tok) {
            logits[static_cast<size_t>(tok)] = cache.logits_all[static_cast<size_t>(last * vocab + tok)];
        }
        return logits;
    }

    static void stable_softmax(const std::vector<float>& logits, std::vector<float>& probs) {
        float max_logit = -std::numeric_limits<float>::infinity();
        for (float x : logits) max_logit = std::max(max_logit, x);
        probs.resize(logits.size());
        float sum = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            float e = std::exp(logits[i] - max_logit);
            probs[i] = e;
            sum += e;
        }
        if (sum <= 0.0f) {
            float u = 1.0f / static_cast<float>(probs.size());
            for (float& p : probs) p = u;
            return;
        }
        for (float& p : probs) p /= sum;
    }

    OutputHeadBackward backprop_output_head(const ForwardCache& cache,
                                            const std::vector<int>& targets) const {
        if (static_cast<int>(targets.size()) != cache.seq) {
            throw std::runtime_error("Target length must match cache sequence length");
        }

        OutputHeadBackward grads;
        grads.grad_out_proj.assign(static_cast<size_t>(d_model * vocab), 0.0f);
        grads.grad_out_bias.assign(static_cast<size_t>(vocab), 0.0f);
        grads.grad_h2.assign(static_cast<size_t>(cache.seq * d_model), 0.0f);

        for (int p = 0; p < cache.seq; ++p) {
            std::vector<float> logits(static_cast<size_t>(vocab), 0.0f);
            for (int tok = 0; tok < vocab; ++tok) {
                logits[static_cast<size_t>(tok)] = cache.logits_all[static_cast<size_t>(p * vocab + tok)];
            }

            std::vector<float> probs;
            stable_softmax(logits, probs);

            int y = targets[static_cast<size_t>(p)];
            grads.loss += -std::log(std::max(probs[static_cast<size_t>(y)], 1e-8f));

            probs[static_cast<size_t>(y)] -= 1.0f;

            for (int tok = 0; tok < vocab; ++tok) {
                float g = probs[static_cast<size_t>(tok)];
                grads.grad_out_bias[static_cast<size_t>(tok)] += g;
                for (int d = 0; d < d_model; ++d) {
                    grads.grad_out_proj[static_cast<size_t>(d * vocab + tok)] +=
                        cache.h2()[static_cast<size_t>(p * d_model + d)] * g;
                    grads.grad_h2[static_cast<size_t>(p * d_model + d)] +=
                        out_proj[static_cast<size_t>(d * vocab + tok)] * g;
                }
            }
        }

        grads.loss /= static_cast<float>(std::max(1, cache.seq));
        return grads;
    }

    FFNBackward backprop_ffn(const ForwardCache& cache,
                             const std::vector<float>& grad_h2) const {
        if (grad_h2.size() != cache.h2().size()) {
            throw std::runtime_error("grad_h2 shape mismatch in backprop_ffn");
        }

        FFNBackward grads;
        grads.grad_w2.assign(static_cast<size_t>(d_ff * d_model), 0.0f);
        grads.grad_w1.assign(static_cast<size_t>(d_model * d_ff), 0.0f);
        grads.grad_h1.assign(static_cast<size_t>(cache.seq * d_model), 0.0f);

        // Residual path: h2 = h1 + ff2.
        // grad_h1 gets direct skip contribution and FFN contribution.
        std::vector<float> grad_ff2 = grad_h2;
        for (size_t i = 0; i < grads.grad_h1.size(); ++i) {
            grads.grad_h1[i] = grad_h2[i];
        }

        // ff2 = ff1_act @ w2.
        for (int f = 0; f < d_ff; ++f) {
            for (int d = 0; d < d_model; ++d) {
                float acc = 0.0f;
                for (int t = 0; t < cache.seq; ++t) {
                    acc += cache.ff1_act()[static_cast<size_t>(t * d_ff + f)] *
                           grad_ff2[static_cast<size_t>(t * d_model + d)];
                }
                grads.grad_w2[static_cast<size_t>(f * d_model + d)] = acc;
            }
        }

        std::vector<float> grad_ff1_act(static_cast<size_t>(cache.seq * d_ff), 0.0f);
        for (int t = 0; t < cache.seq; ++t) {
            for (int f = 0; f < d_ff; ++f) {
                float acc = 0.0f;
                for (int d = 0; d < d_model; ++d) {
                    acc += grad_ff2[static_cast<size_t>(t * d_model + d)] *
                           w2[static_cast<size_t>(f * d_model + d)];
                }
                grad_ff1_act[static_cast<size_t>(t * d_ff + f)] = acc;
            }
        }

        std::vector<float> grad_ff1_lin(static_cast<size_t>(cache.seq * d_ff), 0.0f);
        for (size_t i = 0; i < grad_ff1_lin.size(); ++i) {
            grad_ff1_lin[i] = cache.ff1_lin()[i] > 0.0f ? grad_ff1_act[i] : 0.0f;
        }

        // ff1_lin = h1 @ w1 (or ffn_input @ w1 if LayerNorm is used).
        std::vector<float> grad_ffn_input(static_cast<size_t>(cache.seq * d_model), 0.0f);
        for (int d = 0; d < d_model; ++d) {
            for (int f = 0; f < d_ff; ++f) {
                float acc = 0.0f;
                for (int t = 0; t < cache.seq; ++t) {
                    acc += cache.ffn_input()[static_cast<size_t>(t * d_model + d)] *
                           grad_ff1_lin[static_cast<size_t>(t * d_ff + f)];
                }
                grads.grad_w1[static_cast<size_t>(d * d_ff + f)] = acc;
            }
        }

        std::vector<float> grad_h1_from_ffn(static_cast<size_t>(cache.seq * d_model), 0.0f);
        for (int t = 0; t < cache.seq; ++t) {
            for (int d = 0; d < d_model; ++d) {
                float acc = 0.0f;
                for (int f = 0; f < d_ff; ++f) {
                    acc += grad_ff1_lin[static_cast<size_t>(t * d_ff + f)] *
                           w1[static_cast<size_t>(d * d_ff + f)];
                }
                grad_ffn_input[static_cast<size_t>(t * d_model + d)] = acc;
            }
        }

        // Apply LayerNorm backward if needed.
        if (use_layernorm) {
            std::vector<float> grad_h1_ln;
            layernorm_rows_backward(cache.h1(), grad_ffn_input, cache.seq, d_model, grad_h1_ln);
            for (size_t i = 0; i < grad_h1_from_ffn.size(); ++i) {
                grad_h1_from_ffn[i] = grad_h1_ln[i];
            }
        } else {
            grad_h1_from_ffn = grad_ffn_input;
        }

        for (size_t i = 0; i < grads.grad_h1.size(); ++i) {
            grads.grad_h1[i] += grad_h1_from_ffn[i];
        }

        return grads;
    }

    AttentionProjBackward backprop_attention_proj(const ForwardCache& cache,
                                                  const std::vector<float>& grad_h1) const {
        if (grad_h1.size() != cache.h1().size()) {
            throw std::runtime_error("grad_h1 shape mismatch in backprop_attention_proj");
        }

        AttentionProjBackward grads;
        grads.grad_wo.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_attn_out.assign(static_cast<size_t>(cache.seq * d_model), 0.0f);

        // h1 = x + attn_proj and attn_proj = attn_out @ wo.
        // grad_attn_proj is grad_h1 (residual passthrough).
        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_model; ++j) {
                float acc = 0.0f;
                for (int t = 0; t < cache.seq; ++t) {
                    acc += cache.attn_out()[static_cast<size_t>(t * d_model + i)] *
                           grad_h1[static_cast<size_t>(t * d_model + j)];
                }
                grads.grad_wo[static_cast<size_t>(i * d_model + j)] = acc;
            }
        }

        for (int t = 0; t < cache.seq; ++t) {
            for (int i = 0; i < d_model; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < d_model; ++j) {
                    acc += grad_h1[static_cast<size_t>(t * d_model + j)] *
                           wo[static_cast<size_t>(i * d_model + j)];
                }
                grads.grad_attn_out[static_cast<size_t>(t * d_model + i)] = acc;
            }
        }

        return grads;
    }

    AttentionCoreBackward backprop_attention_core(const ForwardCache& cache,
                                                  const std::vector<float>& grad_attn_out) const {
        if (grad_attn_out.size() != cache.attn_out().size()) {
            throw std::runtime_error("grad_attn_out shape mismatch in backprop_attention_core");
        }
        if (d_model % num_heads != 0) {
            throw std::runtime_error("d_model must be divisible by num_heads");
        }

        AttentionCoreBackward grads;
        grads.grad_wq.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_wk.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_wv.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_x.assign(static_cast<size_t>(cache.seq * d_model), 0.0f);

        std::vector<float> grad_q(static_cast<size_t>(cache.seq * d_model), 0.0f);
        std::vector<float> grad_k(static_cast<size_t>(cache.seq * d_model), 0.0f);
        std::vector<float> grad_v(static_cast<size_t>(cache.seq * d_model), 0.0f);
        const int head_dim = d_model / num_heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Backprop attention independently per head.
        for (int h = 0; h < num_heads; ++h) {
            const int d0 = h * head_dim;
            for (int t = 0; t < cache.seq; ++t) {
                std::vector<float> scores(static_cast<size_t>(t + 1), 0.0f);
                float max_score = -1e30f;
                for (int j = 0; j <= t; ++j) {
                    float s = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        s += cache.q()[static_cast<size_t>(t * d_model + d0 + d)] *
                             cache.k()[static_cast<size_t>(j * d_model + d0 + d)];
                    }
                    s *= scale;
                    scores[static_cast<size_t>(j)] = s;
                    max_score = std::max(max_score, s);
                }

                float exp_sum = 0.0f;
                std::vector<float> a(static_cast<size_t>(t + 1), 0.0f);
                for (int j = 0; j <= t; ++j) {
                    float e = std::exp(scores[static_cast<size_t>(j)] - max_score);
                    a[static_cast<size_t>(j)] = e;
                    exp_sum += e;
                }
                for (int j = 0; j <= t; ++j) {
                    a[static_cast<size_t>(j)] /= exp_sum;
                }

                std::vector<float> grad_a(static_cast<size_t>(t + 1), 0.0f);
                for (int j = 0; j <= t; ++j) {
                    float acc = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        acc += grad_attn_out[static_cast<size_t>(t * d_model + d0 + d)] *
                               cache.v()[static_cast<size_t>(j * d_model + d0 + d)];
                        grad_v[static_cast<size_t>(j * d_model + d0 + d)] +=
                            a[static_cast<size_t>(j)] *
                            grad_attn_out[static_cast<size_t>(t * d_model + d0 + d)];
                    }
                    grad_a[static_cast<size_t>(j)] = acc;
                }

                float dot_ga_a = 0.0f;
                for (int j = 0; j <= t; ++j) {
                    dot_ga_a += grad_a[static_cast<size_t>(j)] * a[static_cast<size_t>(j)];
                }

                for (int j = 0; j <= t; ++j) {
                    float grad_s = a[static_cast<size_t>(j)] * (grad_a[static_cast<size_t>(j)] - dot_ga_a);
                    for (int d = 0; d < head_dim; ++d) {
                        grad_q[static_cast<size_t>(t * d_model + d0 + d)] +=
                            grad_s * scale * cache.k()[static_cast<size_t>(j * d_model + d0 + d)];
                        grad_k[static_cast<size_t>(j * d_model + d0 + d)] +=
                            grad_s * scale * cache.q()[static_cast<size_t>(t * d_model + d0 + d)];
                    }
                }
            }
        }

        // q = x @ wq, k = x @ wk, v = x @ wv.
        for (int i = 0; i < d_model; ++i) {
            for (int o = 0; o < d_model; ++o) {
                float gq = 0.0f;
                float gk = 0.0f;
                float gv = 0.0f;
                for (int t = 0; t < cache.seq; ++t) {
                    gq += cache.attn_input()[static_cast<size_t>(t * d_model + i)] *
                          grad_q[static_cast<size_t>(t * d_model + o)];
                    gk += cache.attn_input()[static_cast<size_t>(t * d_model + i)] *
                          grad_k[static_cast<size_t>(t * d_model + o)];
                    gv += cache.attn_input()[static_cast<size_t>(t * d_model + i)] *
                          grad_v[static_cast<size_t>(t * d_model + o)];
                }
                grads.grad_wq[static_cast<size_t>(i * d_model + o)] = gq;
                grads.grad_wk[static_cast<size_t>(i * d_model + o)] = gk;
                grads.grad_wv[static_cast<size_t>(i * d_model + o)] = gv;
            }
        }

        std::vector<float> grad_attn_input(static_cast<size_t>(cache.seq * d_model), 0.0f);
        for (int t = 0; t < cache.seq; ++t) {
            for (int i = 0; i < d_model; ++i) {
                float acc = 0.0f;
                for (int o = 0; o < d_model; ++o) {
                    acc += grad_q[static_cast<size_t>(t * d_model + o)] * wq[static_cast<size_t>(i * d_model + o)];
                    acc += grad_k[static_cast<size_t>(t * d_model + o)] * wk[static_cast<size_t>(i * d_model + o)];
                    acc += grad_v[static_cast<size_t>(t * d_model + o)] * wv[static_cast<size_t>(i * d_model + o)];
                }
                grad_attn_input[static_cast<size_t>(t * d_model + i)] = acc;
            }
        }

        // Apply LayerNorm backward if needed.
        if (use_layernorm) {
            layernorm_rows_backward(cache.x(), grad_attn_input, cache.seq, d_model, grads.grad_x);
        } else {
            grads.grad_x = grad_attn_input;
        }

        return grads;
    }

    EmbeddingBackward backprop_embeddings(const std::vector<int>& ids,
                                          const std::vector<float>& grad_x) const {
        if (grad_x.size() != ids.size() * static_cast<size_t>(d_model)) {
            throw std::runtime_error("grad_x shape mismatch in backprop_embeddings");
        }

        EmbeddingBackward grads;
        grads.grad_token_emb.assign(static_cast<size_t>(vocab * d_model), 0.0f);
        grads.grad_pos_emb.assign(static_cast<size_t>(max_seq * d_model), 0.0f);

        for (size_t t = 0; t < ids.size(); ++t) {
            int tok = ids[t];
            if (tok < 0 || tok >= vocab) {
                throw std::runtime_error("Token id out of range in backprop_embeddings");
            }
            for (int d = 0; d < d_model; ++d) {
                float g = grad_x[t * static_cast<size_t>(d_model) + static_cast<size_t>(d)];
                grads.grad_token_emb[static_cast<size_t>(tok * d_model + d)] += g;
                grads.grad_pos_emb[static_cast<size_t>(t * d_model + d)] += g;
            }
        }

        return grads;
    }

    // Internal backprop functions that work with LayerCache directly (for multi-layer support)
    FFNBackward backprop_ffn_internal(const LayerCache& layer_cache,
                                      const std::vector<float>& grad_h2) const {
        if (grad_h2.size() != layer_cache.h2.size()) {
            throw std::runtime_error("grad_h2 shape mismatch in backprop_ffn_internal");
        }

        FFNBackward grads;
        grads.grad_w2.assign(static_cast<size_t>(d_ff * d_model), 0.0f);
        grads.grad_w1.assign(static_cast<size_t>(d_model * d_ff), 0.0f);
        grads.grad_h1.assign(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);

        std::vector<float> grad_ff2 = grad_h2;
        for (size_t i = 0; i < grads.grad_h1.size(); ++i) {
            grads.grad_h1[i] = grad_h2[i];
        }

        // ff2 = ff1_act @ w2
        for (int f = 0; f < d_ff; ++f) {
            for (int d = 0; d < d_model; ++d) {
                float acc = 0.0f;
                for (int t = 0; t < layer_cache.seq; ++t) {
                    acc += layer_cache.ff1_act[static_cast<size_t>(t * d_ff + f)] *
                           grad_ff2[static_cast<size_t>(t * d_model + d)];
                }
                grads.grad_w2[static_cast<size_t>(f * d_model + d)] = acc;
            }
        }

        std::vector<float> grad_ff1_act(static_cast<size_t>(layer_cache.seq * d_ff), 0.0f);
        for (int t = 0; t < layer_cache.seq; ++t) {
            for (int f = 0; f < d_ff; ++f) {
                float acc = 0.0f;
                for (int d = 0; d < d_model; ++d) {
                    acc += grad_ff2[static_cast<size_t>(t * d_model + d)] *
                           w2[static_cast<size_t>(f * d_model + d)];
                }
                grad_ff1_act[static_cast<size_t>(t * d_ff + f)] = acc;
            }
        }

        std::vector<float> grad_ff1_lin(static_cast<size_t>(layer_cache.seq * d_ff), 0.0f);
        for (size_t i = 0; i < grad_ff1_lin.size(); ++i) {
            grad_ff1_lin[i] = layer_cache.ff1_lin[i] > 0.0f ? grad_ff1_act[i] : 0.0f;
        }

        // ff1_lin = ffn_input @ w1 (ffn_input already has LayerNorm applied if needed)
        for (int d = 0; d < d_model; ++d) {
            for (int f = 0; f < d_ff; ++f) {
                float acc = 0.0f;
                for (int t = 0; t < layer_cache.seq; ++t) {
                    acc += layer_cache.ffn_input[static_cast<size_t>(t * d_model + d)] *
                           grad_ff1_lin[static_cast<size_t>(t * d_ff + f)];
                }
                grads.grad_w1[static_cast<size_t>(d * d_ff + f)] = acc;
            }
        }

        std::vector<float> grad_h1_from_ffn(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);
        for (int t = 0; t < layer_cache.seq; ++t) {
            for (int d = 0; d < d_model; ++d) {
                float acc = 0.0f;
                for (int f = 0; f < d_ff; ++f) {
                    acc += grad_ff1_lin[static_cast<size_t>(t * d_ff + f)] *
                           w1[static_cast<size_t>(d * d_ff + f)];
                }
                grad_h1_from_ffn[static_cast<size_t>(t * d_model + d)] = acc;
            }
        }

        // Apply LayerNorm backward if needed
        if (use_layernorm) {
            std::vector<float> grad_h1_ln;
            layernorm_rows_backward(layer_cache.h1, grad_h1_from_ffn, layer_cache.seq, d_model, grad_h1_ln);
            for (size_t i = 0; i < grad_h1_from_ffn.size(); ++i) {
                grad_h1_from_ffn[i] = grad_h1_ln[i];
            }
        }

        for (size_t i = 0; i < grads.grad_h1.size(); ++i) {
            grads.grad_h1[i] += grad_h1_from_ffn[i];
        }

        return grads;
    }

    AttentionProjBackward backprop_attention_proj_internal(const LayerCache& layer_cache,
                                                           const std::vector<float>& grad_h1) const {
        if (grad_h1.size() != layer_cache.h1.size()) {
            throw std::runtime_error("grad_h1 shape mismatch in backprop_attention_proj_internal");
        }

        AttentionProjBackward grads;
        grads.grad_wo.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_attn_out.assign(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);

        // h1 = x + attn_proj and attn_proj = attn_out @ wo
        for (int i = 0; i < d_model; ++i) {
            for (int j = 0; j < d_model; ++j) {
                float acc = 0.0f;
                for (int t = 0; t < layer_cache.seq; ++t) {
                    acc += layer_cache.attn_out[static_cast<size_t>(t * d_model + i)] *
                           grad_h1[static_cast<size_t>(t * d_model + j)];
                }
                grads.grad_wo[static_cast<size_t>(i * d_model + j)] = acc;
            }
        }

        for (int t = 0; t < layer_cache.seq; ++t) {
            for (int i = 0; i < d_model; ++i) {
                float acc = 0.0f;
                for (int j = 0; j < d_model; ++j) {
                    acc += grad_h1[static_cast<size_t>(t * d_model + j)] *
                           wo[static_cast<size_t>(i * d_model + j)];
                }
                grads.grad_attn_out[static_cast<size_t>(t * d_model + i)] = acc;
            }
        }

        return grads;
    }

    AttentionCoreBackward backprop_attention_core_internal(const LayerCache& layer_cache,
                                                           const std::vector<float>& grad_attn_out) const {
        if (grad_attn_out.size() != layer_cache.attn_out.size()) {
            throw std::runtime_error("grad_attn_out shape mismatch in backprop_attention_core_internal");
        }
        if (d_model % num_heads != 0) {
            throw std::runtime_error("d_model must be divisible by num_heads");
        }

        AttentionCoreBackward grads;
        grads.grad_wq.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_wk.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_wv.assign(static_cast<size_t>(d_model * d_model), 0.0f);
        grads.grad_x.assign(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);

        std::vector<float> grad_q(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);
        std::vector<float> grad_k(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);
        std::vector<float> grad_v(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);
        const int head_dim = d_model / num_heads;
        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

        // Backprop attention per head
        for (int h = 0; h < num_heads; ++h) {
            const int d0 = h * head_dim;
            for (int t = 0; t < layer_cache.seq; ++t) {
                std::vector<float> scores(static_cast<size_t>(t + 1), 0.0f);
                float max_score = -1e30f;
                for (int j = 0; j <= t; ++j) {
                    float s = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        s += layer_cache.q[static_cast<size_t>(t * d_model + d0 + d)] *
                             layer_cache.k[static_cast<size_t>(j * d_model + d0 + d)];
                    }
                    s *= scale;
                    scores[static_cast<size_t>(j)] = s;
                    max_score = std::max(max_score, s);
                }

                float exp_sum = 0.0f;
                std::vector<float> a(static_cast<size_t>(t + 1), 0.0f);
                for (int j = 0; j <= t; ++j) {
                    float e = std::exp(scores[static_cast<size_t>(j)] - max_score);
                    a[static_cast<size_t>(j)] = e;
                    exp_sum += e;
                }
                for (int j = 0; j <= t; ++j) {
                    a[static_cast<size_t>(j)] /= exp_sum;
                }

                std::vector<float> grad_a(static_cast<size_t>(t + 1), 0.0f);
                for (int j = 0; j <= t; ++j) {
                    float acc = 0.0f;
                    for (int d = 0; d < head_dim; ++d) {
                        acc += grad_attn_out[static_cast<size_t>(t * d_model + d0 + d)] *
                               layer_cache.v[static_cast<size_t>(j * d_model + d0 + d)];
                        grad_v[static_cast<size_t>(j * d_model + d0 + d)] +=
                            a[static_cast<size_t>(j)] *
                            grad_attn_out[static_cast<size_t>(t * d_model + d0 + d)];
                    }
                    grad_a[static_cast<size_t>(j)] = acc;
                }

                float dot_ga_a = 0.0f;
                for (int j = 0; j <= t; ++j) {
                    dot_ga_a += grad_a[static_cast<size_t>(j)] * a[static_cast<size_t>(j)];
                }

                for (int j = 0; j <= t; ++j) {
                    float grad_s = a[static_cast<size_t>(j)] * (grad_a[static_cast<size_t>(j)] - dot_ga_a);
                    for (int d = 0; d < head_dim; ++d) {
                        grad_q[static_cast<size_t>(t * d_model + d0 + d)] +=
                            grad_s * scale * layer_cache.k[static_cast<size_t>(j * d_model + d0 + d)];
                        grad_k[static_cast<size_t>(j * d_model + d0 + d)] +=
                            grad_s * scale * layer_cache.q[static_cast<size_t>(t * d_model + d0 + d)];
                    }
                }
            }
        }

        // q = x @ wq, k = x @ wk, v = x @ wv
        for (int i = 0; i < d_model; ++i) {
            for (int o = 0; o < d_model; ++o) {
                float gq = 0.0f;
                float gk = 0.0f;
                float gv = 0.0f;
                for (int t = 0; t < layer_cache.seq; ++t) {
                    gq += layer_cache.attn_input[static_cast<size_t>(t * d_model + i)] *
                          grad_q[static_cast<size_t>(t * d_model + o)];
                    gk += layer_cache.attn_input[static_cast<size_t>(t * d_model + i)] *
                          grad_k[static_cast<size_t>(t * d_model + o)];
                    gv += layer_cache.attn_input[static_cast<size_t>(t * d_model + i)] *
                          grad_v[static_cast<size_t>(t * d_model + o)];
                }
                grads.grad_wq[static_cast<size_t>(i * d_model + o)] = gq;
                grads.grad_wk[static_cast<size_t>(i * d_model + o)] = gk;
                grads.grad_wv[static_cast<size_t>(i * d_model + o)] = gv;
            }
        }

        std::vector<float> grad_attn_input(static_cast<size_t>(layer_cache.seq * d_model), 0.0f);
        for (int t = 0; t < layer_cache.seq; ++t) {
            for (int i = 0; i < d_model; ++i) {
                float acc = 0.0f;
                for (int o = 0; o < d_model; ++o) {
                    acc += grad_q[static_cast<size_t>(t * d_model + o)] * wq[static_cast<size_t>(i * d_model + o)];
                    acc += grad_k[static_cast<size_t>(t * d_model + o)] * wk[static_cast<size_t>(i * d_model + o)];
                    acc += grad_v[static_cast<size_t>(t * d_model + o)] * wv[static_cast<size_t>(i * d_model + o)];
                }
                grad_attn_input[static_cast<size_t>(t * d_model + i)] = acc;
            }
        }

        // Apply LayerNorm backward if needed (attn_input = LN(x_in))
        if (use_layernorm) {
            layernorm_rows_backward(layer_cache.x_in, grad_attn_input, layer_cache.seq, d_model, grads.grad_x);
        } else {
            grads.grad_x = grad_attn_input;
        }

        return grads;
    }

    EmbeddingBackward backprop_embeddings_internal(const std::vector<int>& ids,
                                                   const std::vector<float>& grad_x) const {
        if (grad_x.size() != ids.size() * static_cast<size_t>(d_model)) {
            throw std::runtime_error("grad_x shape mismatch in backprop_embeddings_internal");
        }

        EmbeddingBackward grads;
        grads.grad_token_emb.assign(static_cast<size_t>(vocab * d_model), 0.0f);
        grads.grad_pos_emb.assign(static_cast<size_t>(max_seq * d_model), 0.0f);

        for (size_t t = 0; t < ids.size(); ++t) {
            int tok = ids[t];
            if (tok < 0 || tok >= vocab) {
                throw std::runtime_error("Token id out of range in backprop_embeddings_internal");
            }
            for (int d = 0; d < d_model; ++d) {
                float g = grad_x[t * static_cast<size_t>(d_model) + static_cast<size_t>(d)];
                grads.grad_token_emb[static_cast<size_t>(tok * d_model + d)] += g;
                grads.grad_pos_emb[static_cast<size_t>(t * d_model + d)] += g;
            }
        }

        return grads;
    }

    float evaluate_next_token_loss(const std::vector<int>& corpus_ids,
                                   int context_window = 24) const {
        if (corpus_ids.size() < 2) {
            throw std::runtime_error("Corpus must contain at least 2 tokens");
        }
        const int cw = std::max(2, std::min(context_window, max_seq));

        float total_loss = 0.0f;
        size_t total_targets = 0;

        for (size_t start = 0; start + 1 < corpus_ids.size(); ++start) {
            size_t end = std::min(start + static_cast<size_t>(cw), corpus_ids.size() - 1);
            if (end <= start) continue;

            std::vector<int> ctx(corpus_ids.begin() + static_cast<std::ptrdiff_t>(start),
                                 corpus_ids.begin() + static_cast<std::ptrdiff_t>(end));
            std::vector<int> tgt(corpus_ids.begin() + static_cast<std::ptrdiff_t>(start + 1),
                                 corpus_ids.begin() + static_cast<std::ptrdiff_t>(end + 1));

            ForwardCache cache = forward_cache(ctx);
            for (size_t p = 0; p < ctx.size(); ++p) {
                std::vector<float> row(static_cast<size_t>(vocab), 0.0f);
                for (int tok = 0; tok < vocab; ++tok) {
                    row[static_cast<size_t>(tok)] = cache.logits_all[static_cast<size_t>(p * vocab + tok)];
                }
                std::vector<float> probs;
                stable_softmax(row, probs);
                int y = tgt[p];
                total_loss += -std::log(std::max(probs[static_cast<size_t>(y)], 1e-8f));
                ++total_targets;
            }
        }

        return total_loss / static_cast<float>(std::max<size_t>(1, total_targets));
    }

    float train_next_token(const std::vector<int>& corpus_ids,
                           int epochs = 40,
                           int context_window = 24,
                           float lr = 0.04f,
                           int report_every = 10,
                           const std::string& optimizer = "sgd",
                           int warmup_steps = 0,
                           float min_lr_ratio = 0.1f,
                           float weight_decay = 0.0f,
                           int batch_size = 1,
                           float grad_clip_norm = 0.0f,
                           bool report_lr = false,
                           bool report_grad_norm = false,
                           float val_split = 0.0f,
                           bool report_val_perplexity = false,
                           bool resume_optimizer_state = false) {
        if (batch_size <= 0) {
            throw std::runtime_error("batch_size must be >= 1");
        }
        if (corpus_ids.size() < 2) {
            throw std::runtime_error("Corpus must contain at least 2 tokens");
        }

        std::vector<int> train_ids;
        std::vector<int> val_ids;
        {
            auto split = split_train_val_ids(corpus_ids, val_split);
            train_ids = std::move(split.first);
            val_ids = std::move(split.second);
        }
        if (train_ids.size() < 2) {
            throw std::runtime_error("Training split must contain at least 2 tokens");
        }

        const int cw = std::max(2, std::min(context_window, max_seq));

        struct AdamState {
            std::vector<float>* m = nullptr;
            std::vector<float>* v = nullptr;
        };

        if (!resume_optimizer_state) {
            clear_optimizer_state();
        }

        AdamState st_out_bias{&adam_m_out_bias, &adam_v_out_bias};
        AdamState st_out_proj{&adam_m_out_proj, &adam_v_out_proj};
        AdamState st_w1{&adam_m_w1, &adam_v_w1};
        AdamState st_w2{&adam_m_w2, &adam_v_w2};
        AdamState st_wo{&adam_m_wo, &adam_v_wo};
        AdamState st_wq{&adam_m_wq, &adam_v_wq};
        AdamState st_wk{&adam_m_wk, &adam_v_wk};
        AdamState st_wv{&adam_m_wv, &adam_v_wv};
        AdamState st_tok{&adam_m_tok, &adam_v_tok};
        AdamState st_pos{&adam_m_pos, &adam_v_pos};

        auto ensure_state = [](AdamState& st, size_t n) {
            if (st.m == nullptr || st.v == nullptr) {
                throw std::runtime_error("AdamState vectors must be initialized");
            }
            if (st.m->size() != n) {
                st.m->assign(n, 0.0f);
                st.v->assign(n, 0.0f);
            }
        };

        const float beta1 = 0.9f;
        const float beta2 = 0.999f;
        const float eps = 1e-8f;
        int global_step = train_global_step;

        int updates_per_epoch = 0;
        for (size_t start = 0; start + 1 < train_ids.size(); ++start) {
            size_t end = std::min(start + static_cast<size_t>(cw), train_ids.size() - 1);
            if (end > start) ++updates_per_epoch;
        }
        updates_per_epoch = std::max(1, updates_per_epoch);
        const int steps_per_epoch = std::max(1, (updates_per_epoch + batch_size - 1) / batch_size);
        const int total_steps = std::max(1, epochs * steps_per_epoch);

        auto scheduled_lr = [&](int step) {
            if (warmup_steps > 0 && step <= warmup_steps) {
                return lr * (static_cast<float>(step) / static_cast<float>(warmup_steps));
            }
            int denom = std::max(1, total_steps - warmup_steps);
            float p = static_cast<float>(step - warmup_steps) / static_cast<float>(denom);
            p = std::min(1.0f, std::max(0.0f, p));
            float c = 0.5f * (1.0f + std::cos(3.1415926535f * p));
            return lr * (min_lr_ratio + (1.0f - min_lr_ratio) * c);
        };

        auto apply_update = [&](std::vector<float>& param,
                                const std::vector<float>& grad,
                                AdamState& st,
                                float lr_scale,
                                int step) {
            if (optimizer == "adamw") {
                ensure_state(st, param.size());
                float lr_eff = scheduled_lr(step) * lr_scale;
                float b1_corr = 1.0f - std::pow(beta1, static_cast<float>(step));
                float b2_corr = 1.0f - std::pow(beta2, static_cast<float>(step));
                for (size_t i = 0; i < param.size(); ++i) {
                    float g = grad[i] + weight_decay * param[i];
                    (*st.m)[i] = beta1 * (*st.m)[i] + (1.0f - beta1) * g;
                    (*st.v)[i] = beta2 * (*st.v)[i] + (1.0f - beta2) * g * g;
                    float m_hat = (*st.m)[i] / b1_corr;
                    float v_hat = (*st.v)[i] / b2_corr;
                    param[i] -= lr_eff * (m_hat / (std::sqrt(v_hat) + eps));
                }
            } else {
                float lr_eff = scheduled_lr(step) * lr_scale;
                for (size_t i = 0; i < param.size(); ++i) {
                    float g = grad[i] + weight_decay * param[i];
                    param[i] -= lr_eff * g;
                }
            }
        };

        std::vector<float> acc_out_bias(out_bias.size(), 0.0f);
        std::vector<float> acc_out_proj(out_proj.size(), 0.0f);
        std::vector<float> acc_w2(w2.size(), 0.0f);
        std::vector<float> acc_w1(w1.size(), 0.0f);
        std::vector<float> acc_wo(wo.size(), 0.0f);
        std::vector<float> acc_wq(wq.size(), 0.0f);
        std::vector<float> acc_wk(wk.size(), 0.0f);
        std::vector<float> acc_wv(wv.size(), 0.0f);
        std::vector<float> acc_tok(token_emb.size(), 0.0f);
        std::vector<float> acc_pos(pos_emb.size(), 0.0f);

        auto zero_vec = [](std::vector<float>& v) {
            std::fill(v.begin(), v.end(), 0.0f);
        };
        auto add_inplace = [](std::vector<float>& dst, const std::vector<float>& src) {
            for (size_t i = 0; i < dst.size(); ++i) {
                dst[i] += src[i];
            }
        };
        auto scale_inplace = [](std::vector<float>& v, float s) {
            for (float& x : v) {
                x *= s;
            }
        };

        auto clear_accumulators = [&]() {
            zero_vec(acc_out_bias);
            zero_vec(acc_out_proj);
            zero_vec(acc_w2);
            zero_vec(acc_w1);
            zero_vec(acc_wo);
            zero_vec(acc_wq);
            zero_vec(acc_wk);
            zero_vec(acc_wv);
            zero_vec(acc_tok);
            zero_vec(acc_pos);
        };

        auto global_grad_norm = [&]() {
            double sum_sq = 0.0;
            auto add_sq = [&](const std::vector<float>& v) {
                for (float x : v) {
                    sum_sq += static_cast<double>(x) * static_cast<double>(x);
                }
            };
            add_sq(acc_out_bias);
            add_sq(acc_out_proj);
            add_sq(acc_w2);
            add_sq(acc_w1);
            add_sq(acc_wo);
            add_sq(acc_wq);
            add_sq(acc_wk);
            add_sq(acc_wv);
            add_sq(acc_tok);
            add_sq(acc_pos);
            return static_cast<float>(std::sqrt(sum_sq));
        };

        auto apply_clipping_if_needed = [&]() {
            if (grad_clip_norm <= 0.0f) return;
            float norm = global_grad_norm();
            if (norm <= grad_clip_norm || norm <= 0.0f) return;

            float s = grad_clip_norm / norm;
            scale_inplace(acc_out_bias, s);
            scale_inplace(acc_out_proj, s);
            scale_inplace(acc_w2, s);
            scale_inplace(acc_w1, s);
            scale_inplace(acc_wo, s);
            scale_inplace(acc_wq, s);
            scale_inplace(acc_wk, s);
            scale_inplace(acc_wv, s);
            scale_inplace(acc_tok, s);
            scale_inplace(acc_pos, s);
        };

        float final_loss = 0.0f;
        for (int epoch = 1; epoch <= epochs; ++epoch) {
            float total_loss = 0.0f;
            size_t total_targets = 0;
            int batch_count = 0;
            float epoch_lr_sum = 0.0f;
            float epoch_grad_norm_sum = 0.0f;
            int epoch_update_steps = 0;

            clear_accumulators();

            auto flush_batch = [&]() {
                if (batch_count <= 0) return;

                float inv_bs = 1.0f / static_cast<float>(batch_count);
                scale_inplace(acc_out_bias, inv_bs);
                scale_inplace(acc_out_proj, inv_bs);
                scale_inplace(acc_w2, inv_bs);
                scale_inplace(acc_w1, inv_bs);
                scale_inplace(acc_wo, inv_bs);
                scale_inplace(acc_wq, inv_bs);
                scale_inplace(acc_wk, inv_bs);
                scale_inplace(acc_wv, inv_bs);
                scale_inplace(acc_tok, inv_bs);
                scale_inplace(acc_pos, inv_bs);

                float grad_norm_before_clip = global_grad_norm();

                apply_clipping_if_needed();

                ++global_step;
                float lr_step = scheduled_lr(global_step);
                epoch_lr_sum += lr_step;
                epoch_grad_norm_sum += grad_norm_before_clip;
                ++epoch_update_steps;

                apply_update(out_bias, acc_out_bias, st_out_bias, 1.0f, global_step);
                apply_update(out_proj, acc_out_proj, st_out_proj, 1.0f, global_step);
                apply_update(w2, acc_w2, st_w2, 0.02f, global_step);
                apply_update(w1, acc_w1, st_w1, 0.02f, global_step);
                apply_update(wo, acc_wo, st_wo, 0.01f, global_step);
                apply_update(wq, acc_wq, st_wq, 0.005f, global_step);
                apply_update(wk, acc_wk, st_wk, 0.005f, global_step);
                apply_update(wv, acc_wv, st_wv, 0.005f, global_step);
                apply_update(token_emb, acc_tok, st_tok, 0.002f, global_step);
                apply_update(pos_emb, acc_pos, st_pos, 0.002f, global_step);

                batch_count = 0;
                clear_accumulators();
            };

            for (size_t start = 0; start + 1 < train_ids.size(); ++start) {
                size_t end = std::min(start + static_cast<size_t>(cw), train_ids.size() - 1);
                if (end <= start) continue;

                std::vector<int> ctx(train_ids.begin() + static_cast<std::ptrdiff_t>(start),
                                     train_ids.begin() + static_cast<std::ptrdiff_t>(end));
                std::vector<int> tgt(train_ids.begin() + static_cast<std::ptrdiff_t>(start + 1),
                                     train_ids.begin() + static_cast<std::ptrdiff_t>(end + 1));

                ForwardCache cache = forward_cache(ctx);
                OutputHeadBackward grads = backprop_output_head(cache, tgt);
                total_loss += grads.loss * static_cast<float>(ctx.size());
                total_targets += ctx.size();
                add_inplace(acc_out_bias, grads.grad_out_bias);
                add_inplace(acc_out_proj, grads.grad_out_proj);

                // Multi-layer backprop: process layers in reverse
                std::vector<float> grad_h2_cur = grads.grad_h2;
                
                for (int layer = num_layers - 1; layer >= 0; --layer) {
                    LayerCache& layer_cache = cache.layers[static_cast<size_t>(layer)];
                    
                    // FFN backward for this layer
                    FFNBackward ffn_grads = backprop_ffn_internal(layer_cache, grad_h2_cur);
                    add_inplace(acc_w2, ffn_grads.grad_w2);
                    add_inplace(acc_w1, ffn_grads.grad_w1);

                    // Attention projection backward for this layer
                    AttentionProjBackward attn_grads = backprop_attention_proj_internal(layer_cache, ffn_grads.grad_h1);
                    add_inplace(acc_wo, attn_grads.grad_wo);

                    // Attention core backward for this layer
                    AttentionCoreBackward attn_core_grads = backprop_attention_core_internal(layer_cache, attn_grads.grad_attn_out);
                    add_inplace(acc_wq, attn_core_grads.grad_wq);
                    add_inplace(acc_wk, attn_core_grads.grad_wk);
                    add_inplace(acc_wv, attn_core_grads.grad_wv);

                    // If this is the first layer, backprop through embeddings
                    if (layer == 0) {
                        EmbeddingBackward emb_grads = backprop_embeddings_internal(ctx, attn_core_grads.grad_x);
                        add_inplace(acc_tok, emb_grads.grad_token_emb);
                        add_inplace(acc_pos, emb_grads.grad_pos_emb);
                    } else {
                        // For non-first layers, grad_x becomes grad_h2 for the previous layer
                        grad_h2_cur = attn_core_grads.grad_x;
                    }
                }

                ++batch_count;
                if (batch_count >= batch_size) {
                    flush_batch();
                }
            }

            flush_batch();

            final_loss = total_loss / static_cast<float>(std::max<size_t>(1, total_targets));
            if (report_every > 0 && (epoch % report_every == 0 || epoch == 1 || epoch == epochs)) {
                std::cout << "epoch=" << epoch << " loss=" << final_loss;
                if (report_lr && epoch_update_steps > 0) {
                    std::cout << " avg_lr=" << (epoch_lr_sum / static_cast<float>(epoch_update_steps));
                }
                if (report_grad_norm && epoch_update_steps > 0) {
                    std::cout << " avg_grad_norm=" << (epoch_grad_norm_sum / static_cast<float>(epoch_update_steps));
                }
                if (report_val_perplexity && val_ids.size() >= 2) {
                    float val_loss = evaluate_next_token_loss(val_ids, cw);
                    float val_ppl = std::exp(val_loss);
                    std::cout << " val_ppl=" << val_ppl;
                }
                std::cout << "\n";
            }
        }

        train_global_step = global_step;

        return final_loss;
    }

    std::vector<int> generate(const std::vector<int>& prompt,
                              int steps,
                              int context_window = 32,
                              float temperature = 1.0f,
                              int top_k = 0,
                              uint32_t seed = 2026,
                              float top_p = 1.0f,
                              float repetition_penalty = 1.0f) const {
        if (prompt.empty()) {
            throw std::runtime_error("Prompt must not be empty");
        }
        std::vector<int> ids = prompt;
        ids.reserve(prompt.size() + static_cast<size_t>(steps));
        std::mt19937 rng(seed);

        for (int s = 0; s < steps; ++s) {
            int start = 0;
            if (static_cast<int>(ids.size()) > context_window) {
                start = static_cast<int>(ids.size()) - context_window;
            }
            std::vector<int> ctx(ids.begin() + start, ids.end());
            std::vector<float> logits = forward(ctx);

            if (temperature > 0.0f) {
                for (float& x : logits) {
                    x /= temperature;
                }
            }

            if (repetition_penalty > 1.0f) {
                std::vector<char> seen(static_cast<size_t>(vocab), 0);
                for (int id : ids) {
                    if (id >= 0 && id < vocab) {
                        seen[static_cast<size_t>(id)] = 1;
                    }
                }
                for (int i = 0; i < vocab; ++i) {
                    if (!seen[static_cast<size_t>(i)]) continue;
                    float& logit = logits[static_cast<size_t>(i)];
                    if (logit > 0.0f) {
                        logit /= repetition_penalty;
                    } else {
                        logit *= repetition_penalty;
                    }
                }
            }

            if (top_k > 0 && top_k < vocab) {
                std::vector<int> idx(static_cast<size_t>(vocab), 0);
                for (int i = 0; i < vocab; ++i) idx[static_cast<size_t>(i)] = i;
                std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [&](int a, int b) {
                    return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)];
                });
                std::vector<char> keep(static_cast<size_t>(vocab), 0);
                for (int i = 0; i < top_k; ++i) keep[static_cast<size_t>(idx[static_cast<size_t>(i)])] = 1;
                for (int i = 0; i < vocab; ++i) {
                    if (!keep[static_cast<size_t>(i)]) {
                        logits[static_cast<size_t>(i)] = -std::numeric_limits<float>::infinity();
                    }
                }
            }

            if (top_p > 0.0f && top_p < 1.0f) {
                std::vector<float> probs_for_filter;
                stable_softmax(logits, probs_for_filter);

                std::vector<int> sorted_idx(static_cast<size_t>(vocab), 0);
                for (int i = 0; i < vocab; ++i) sorted_idx[static_cast<size_t>(i)] = i;
                std::sort(sorted_idx.begin(), sorted_idx.end(), [&](int a, int b) {
                    return probs_for_filter[static_cast<size_t>(a)] > probs_for_filter[static_cast<size_t>(b)];
                });

                std::vector<char> keep(static_cast<size_t>(vocab), 0);
                float cumulative = 0.0f;
                for (int id : sorted_idx) {
                    keep[static_cast<size_t>(id)] = 1;
                    cumulative += probs_for_filter[static_cast<size_t>(id)];
                    if (cumulative >= top_p) break;
                }

                for (int i = 0; i < vocab; ++i) {
                    if (!keep[static_cast<size_t>(i)]) {
                        logits[static_cast<size_t>(i)] = -std::numeric_limits<float>::infinity();
                    }
                }
            }

            int next_id = 0;
            if (temperature <= 0.0f || top_k == 1) {
                float best = logits[0];
                for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
                    if (logits[static_cast<size_t>(i)] > best) {
                        best = logits[static_cast<size_t>(i)];
                        next_id = i;
                    }
                }
            } else {
                std::vector<float> probs;
                stable_softmax(logits, probs);
                std::discrete_distribution<int> dist(probs.begin(), probs.end());
                next_id = dist(rng);
            }
            ids.push_back(next_id);
        }

        return ids;
    }
};

#ifndef TOY_LLM_NO_MAIN
int main(int argc, char* argv[]) {
    const std::string default_corpus =
        "hello world. this is a tiny transformer demo for character generation. "
        "hello world. this is a tiny transformer demo for character generation. ";

    int num_layers = 1;
    std::vector<std::string> data_files;
    std::string vocab_path;
    std::string build_vocab_out_path;
    std::string prompt = "hello ";
    int vocab_size_target = 256;
    float val_split = 0.0f;
    float temperature = 0.9f;
    int top_k = 5;
    float top_p = 1.0f;
    float repetition_penalty = 1.0f;
    int generation_steps = 80;
    std::string load_checkpoint_path;
    std::string save_checkpoint_path;
    std::string save_config_path;
    int train_epochs = 40;
    int train_context_window = 24;
    float train_lr = 0.04f;
    int report_every = 10;
    std::string optimizer = "sgd";
    int warmup_steps = 0;
    float min_lr_ratio = 0.1f;
    float weight_decay = 0.0f;
    int batch_size = 1;
    float grad_clip_norm = 0.0f;
    bool report_lr = false;
    bool report_grad_norm = false;
    bool dry_run = false;
    bool strict_config = false;
    std::vector<std::string> config_warnings;
    std::unordered_set<std::string> warning_set;

    auto apply_scalar = [&](const std::string& key, const std::string& value) {
        if (key == "num_layers") {
            num_layers = std::max(1, std::atoi(value.c_str()));
        } else if (key == "vocab") {
            vocab_path = value;
        } else if (key == "build_vocab") {
            build_vocab_out_path = value;
        } else if (key == "prompt") {
            prompt = value;
        } else if (key == "vocab_size") {
            vocab_size_target = std::max(8, std::atoi(value.c_str()));
        } else if (key == "val_split") {
            val_split = std::stof(value);
        } else if (key == "temperature") {
            temperature = std::stof(value);
        } else if (key == "top_k") {
            top_k = std::max(0, std::atoi(value.c_str()));
        } else if (key == "top_p") {
            top_p = std::stof(value);
        } else if (key == "rep_penalty") {
            repetition_penalty = std::stof(value);
        } else if (key == "steps") {
            generation_steps = std::max(1, std::atoi(value.c_str()));
        } else if (key == "resume") {
            load_checkpoint_path = value;
        } else if (key == "save_ckpt") {
            save_checkpoint_path = value;
        } else if (key == "save_config") {
            save_config_path = value;
        } else if (key == "epochs") {
            train_epochs = std::max(1, std::atoi(value.c_str()));
        } else if (key == "ctx_window") {
            train_context_window = std::max(2, std::atoi(value.c_str()));
        } else if (key == "lr") {
            train_lr = std::stof(value);
        } else if (key == "optimizer") {
            optimizer = value;
        } else if (key == "warmup") {
            warmup_steps = std::max(0, std::atoi(value.c_str()));
        } else if (key == "min_lr_ratio") {
            min_lr_ratio = std::stof(value);
        } else if (key == "weight_decay") {
            weight_decay = std::stof(value);
        } else if (key == "batch_size") {
            batch_size = std::max(1, std::atoi(value.c_str()));
        } else if (key == "grad_clip") {
            grad_clip_norm = std::stof(value);
        } else if (key == "report_every") {
            report_every = std::max(0, std::atoi(value.c_str()));
        } else if (key == "report_lr") {
            report_lr = cfg_is_true_like(value);
        } else if (key == "report_grad") {
            report_grad_norm = cfg_is_true_like(value);
        } else if (key == "dry_run") {
            dry_run = cfg_is_true_like(value);
        } else if (key == "strict_config") {
            strict_config = cfg_is_true_like(value);
        }
    };

    // Pre-pass: detect strict mode from CLI so config loading can enforce key checks.
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--strict-config") {
            strict_config = true;
        }
    }

    // First pass: load config file (if present) to set defaults.
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        std::string config_path;
        if (arg == "--config") {
            if (i + 1 >= argc) throw std::runtime_error("--config requires a file path");
            config_path = argv[++i];
        } else if (arg.rfind("--config=", 0) == 0) {
            config_path = arg.substr(9);
        }
        if (!config_path.empty()) {
            RunConfigJson cfg = load_run_config_json(config_path);
            bool strict_for_this_cfg = strict_config;
            auto it_strict = cfg.scalars.find("strict_config");
            if (it_strict != cfg.scalars.end() && cfg_is_true_like(it_strict->second)) {
                strict_for_this_cfg = true;
            }
            validate_run_config_keys(cfg, strict_for_this_cfg);
            if (!strict_for_this_cfg) {
                std::vector<std::string> ws = collect_non_strict_config_warnings(cfg);
                for (const std::string& w : ws) {
                    if (warning_set.insert(w).second) {
                        config_warnings.push_back(w);
                    }
                }
            }
            for (const auto& kv : cfg.scalars) {
                apply_scalar(kv.first, kv.second);
            }
            auto it_data = cfg.string_arrays.find("data");
            if (it_data != cfg.string_arrays.end()) {
                data_files = it_data->second;
            }
        }
    }

    // Second pass: parse CLI args and override config values.
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: toy_llm [num_layers] [--data FILE]... [--vocab VOCAB_FILE] [--prompt TEXT]\n";
            std::cout << "  num_layers       Optional positional integer (default: 1)\n";
            std::cout << "  --data FILE      Add a dataset text file (can be repeated)\n";
            std::cout << "  --config FILE    Load run settings from JSON file\n";
            std::cout << "  --vocab FILE     Enable subword mode with vocab from FILE\n";
            std::cout << "  --build-vocab FILE   Build vocab from dataset and write to FILE\n";
            std::cout << "  --vocab-size N       Target size for --build-vocab mode (default: 256)\n";
            std::cout << "  --prompt TEXT    Prompt for generation\n";
            std::cout << "  --val-split R    Validation split ratio in [0,0.9], e.g. 0.1\n";
            std::cout << "  --temperature T  Sampling temperature (default: 0.9)\n";
            std::cout << "  --top-k K        Top-k sampling cutoff (default: 5)\n";
            std::cout << "  --top-p P        Top-p nucleus threshold (default: 1.0)\n";
            std::cout << "  --rep-penalty R  Repetition penalty (>1 discourages repeats)\n";
            std::cout << "  --steps N        Number of generation steps (default: 80)\n";
            std::cout << "  --resume FILE    Load model + optimizer state from checkpoint\n";
            std::cout << "  --save-ckpt FILE Save model + optimizer state after training\n";
            std::cout << "  --save-config FILE Save fully resolved run config JSON\n";
            std::cout << "  --epochs N       Training epochs (default: 40)\n";
            std::cout << "  --ctx-window N   Training context window (default: 24)\n";
            std::cout << "  --lr F           Base learning rate (default: 0.04)\n";
            std::cout << "  --optimizer S    Optimizer: sgd or adamw (default: sgd)\n";
            std::cout << "  --warmup N       LR warmup steps (default: 0)\n";
            std::cout << "  --min-lr-ratio F Cosine floor ratio (default: 0.1)\n";
            std::cout << "  --weight-decay F Weight decay (default: 0.0)\n";
            std::cout << "  --batch-size N   Minibatch accumulation size (default: 1)\n";
            std::cout << "  --grad-clip F    Global gradient clip norm (default: 0.0)\n";
            std::cout << "  --report-every N Print every N epochs (default: 10)\n";
            std::cout << "  --report-lr      Print average LR each report\n";
            std::cout << "  --report-grad    Print average grad norm each report\n";
            std::cout << "  --dry-run        Validate/resolve settings and print effective config only\n";
            std::cout << "  --strict-config  Fail if config JSON contains unknown keys\n";
            return 0;
        }

        if (arg == "--data") {
            if (i + 1 >= argc) throw std::runtime_error("--data requires a file path");
            data_files.push_back(argv[++i]);
            continue;
        }
        if (arg.rfind("--data=", 0) == 0) {
            data_files.push_back(arg.substr(7));
            continue;
        }
        if (arg == "--config") {
            if (i + 1 >= argc) throw std::runtime_error("--config requires a file path");
            ++i;
            continue;
        }
        if (arg.rfind("--config=", 0) == 0) {
            continue;
        }
        if (arg == "--vocab") {
            if (i + 1 >= argc) throw std::runtime_error("--vocab requires a file path");
            vocab_path = argv[++i];
            continue;
        }
        if (arg.rfind("--vocab=", 0) == 0) {
            vocab_path = arg.substr(8);
            continue;
        }
        if (arg == "--build-vocab") {
            if (i + 1 >= argc) throw std::runtime_error("--build-vocab requires a file path");
            build_vocab_out_path = argv[++i];
            continue;
        }
        if (arg.rfind("--build-vocab=", 0) == 0) {
            build_vocab_out_path = arg.substr(14);
            continue;
        }
        if (arg == "--vocab-size") {
            if (i + 1 >= argc) throw std::runtime_error("--vocab-size requires a value");
            vocab_size_target = std::max(8, std::atoi(argv[++i]));
            continue;
        }
        if (arg.rfind("--vocab-size=", 0) == 0) {
            vocab_size_target = std::max(8, std::atoi(arg.substr(13).c_str()));
            continue;
        }
        if (arg == "--prompt") {
            if (i + 1 >= argc) throw std::runtime_error("--prompt requires text");
            prompt = argv[++i];
            continue;
        }
        if (arg.rfind("--prompt=", 0) == 0) {
            prompt = arg.substr(9);
            continue;
        }
        if (arg == "--val-split") {
            if (i + 1 >= argc) throw std::runtime_error("--val-split requires a value");
            val_split = std::stof(argv[++i]);
            continue;
        }
        if (arg.rfind("--val-split=", 0) == 0) {
            val_split = std::stof(arg.substr(12));
            continue;
        }
        if (arg == "--temperature") {
            if (i + 1 >= argc) throw std::runtime_error("--temperature requires a value");
            temperature = std::stof(argv[++i]);
            continue;
        }
        if (arg.rfind("--temperature=", 0) == 0) {
            temperature = std::stof(arg.substr(14));
            continue;
        }
        if (arg == "--top-k") {
            if (i + 1 >= argc) throw std::runtime_error("--top-k requires a value");
            top_k = std::max(0, std::atoi(argv[++i]));
            continue;
        }
        if (arg.rfind("--top-k=", 0) == 0) {
            top_k = std::max(0, std::atoi(arg.substr(8).c_str()));
            continue;
        }
        if (arg == "--top-p") {
            if (i + 1 >= argc) throw std::runtime_error("--top-p requires a value");
            top_p = std::stof(argv[++i]);
            continue;
        }
        if (arg.rfind("--top-p=", 0) == 0) {
            top_p = std::stof(arg.substr(8));
            continue;
        }
        if (arg == "--rep-penalty") {
            if (i + 1 >= argc) throw std::runtime_error("--rep-penalty requires a value");
            repetition_penalty = std::stof(argv[++i]);
            continue;
        }
        if (arg.rfind("--rep-penalty=", 0) == 0) {
            repetition_penalty = std::stof(arg.substr(14));
            continue;
        }
        if (arg == "--steps") {
            if (i + 1 >= argc) throw std::runtime_error("--steps requires a value");
            generation_steps = std::max(1, std::atoi(argv[++i]));
            continue;
        }
        if (arg.rfind("--steps=", 0) == 0) {
            generation_steps = std::max(1, std::atoi(arg.substr(8).c_str()));
            continue;
        }
        if (arg == "--resume") {
            if (i + 1 >= argc) throw std::runtime_error("--resume requires a file path");
            load_checkpoint_path = argv[++i];
            continue;
        }
        if (arg.rfind("--resume=", 0) == 0) {
            load_checkpoint_path = arg.substr(9);
            continue;
        }
        if (arg == "--save-ckpt") {
            if (i + 1 >= argc) throw std::runtime_error("--save-ckpt requires a file path");
            save_checkpoint_path = argv[++i];
            continue;
        }
        if (arg.rfind("--save-ckpt=", 0) == 0) {
            save_checkpoint_path = arg.substr(12);
            continue;
        }
        if (arg == "--save-config") {
            if (i + 1 >= argc) throw std::runtime_error("--save-config requires a file path");
            save_config_path = argv[++i];
            continue;
        }
        if (arg.rfind("--save-config=", 0) == 0) {
            save_config_path = arg.substr(14);
            continue;
        }
        if (arg == "--epochs") {
            if (i + 1 >= argc) throw std::runtime_error("--epochs requires a value");
            train_epochs = std::max(1, std::atoi(argv[++i]));
            continue;
        }
        if (arg.rfind("--epochs=", 0) == 0) {
            train_epochs = std::max(1, std::atoi(arg.substr(9).c_str()));
            continue;
        }
        if (arg == "--ctx-window") {
            if (i + 1 >= argc) throw std::runtime_error("--ctx-window requires a value");
            train_context_window = std::max(2, std::atoi(argv[++i]));
            continue;
        }
        if (arg.rfind("--ctx-window=", 0) == 0) {
            train_context_window = std::max(2, std::atoi(arg.substr(13).c_str()));
            continue;
        }
        if (arg == "--lr") {
            if (i + 1 >= argc) throw std::runtime_error("--lr requires a value");
            train_lr = std::stof(argv[++i]);
            continue;
        }
        if (arg.rfind("--lr=", 0) == 0) {
            train_lr = std::stof(arg.substr(5));
            continue;
        }
        if (arg == "--optimizer") {
            if (i + 1 >= argc) throw std::runtime_error("--optimizer requires a value");
            optimizer = argv[++i];
            continue;
        }
        if (arg.rfind("--optimizer=", 0) == 0) {
            optimizer = arg.substr(12);
            continue;
        }
        if (arg == "--warmup") {
            if (i + 1 >= argc) throw std::runtime_error("--warmup requires a value");
            warmup_steps = std::max(0, std::atoi(argv[++i]));
            continue;
        }
        if (arg.rfind("--warmup=", 0) == 0) {
            warmup_steps = std::max(0, std::atoi(arg.substr(9).c_str()));
            continue;
        }
        if (arg == "--min-lr-ratio") {
            if (i + 1 >= argc) throw std::runtime_error("--min-lr-ratio requires a value");
            min_lr_ratio = std::stof(argv[++i]);
            continue;
        }
        if (arg.rfind("--min-lr-ratio=", 0) == 0) {
            min_lr_ratio = std::stof(arg.substr(15));
            continue;
        }
        if (arg == "--weight-decay") {
            if (i + 1 >= argc) throw std::runtime_error("--weight-decay requires a value");
            weight_decay = std::stof(argv[++i]);
            continue;
        }
        if (arg.rfind("--weight-decay=", 0) == 0) {
            weight_decay = std::stof(arg.substr(15));
            continue;
        }
        if (arg == "--batch-size") {
            if (i + 1 >= argc) throw std::runtime_error("--batch-size requires a value");
            batch_size = std::max(1, std::atoi(argv[++i]));
            continue;
        }
        if (arg.rfind("--batch-size=", 0) == 0) {
            batch_size = std::max(1, std::atoi(arg.substr(13).c_str()));
            continue;
        }
        if (arg == "--grad-clip") {
            if (i + 1 >= argc) throw std::runtime_error("--grad-clip requires a value");
            grad_clip_norm = std::stof(argv[++i]);
            continue;
        }
        if (arg.rfind("--grad-clip=", 0) == 0) {
            grad_clip_norm = std::stof(arg.substr(12));
            continue;
        }
        if (arg == "--report-every") {
            if (i + 1 >= argc) throw std::runtime_error("--report-every requires a value");
            report_every = std::max(0, std::atoi(argv[++i]));
            continue;
        }
        if (arg.rfind("--report-every=", 0) == 0) {
            report_every = std::max(0, std::atoi(arg.substr(15).c_str()));
            continue;
        }
        if (arg == "--report-lr") {
            report_lr = true;
            continue;
        }
        if (arg == "--report-grad") {
            report_grad_norm = true;
            continue;
        }
        if (arg == "--dry-run") {
            dry_run = true;
            continue;
        }
        if (arg == "--strict-config") {
            strict_config = true;
            continue;
        }

        bool is_integer = !arg.empty();
        for (char ch : arg) {
            if (!std::isdigit(static_cast<unsigned char>(ch)) && ch != '-') {
                is_integer = false;
                break;
            }
        }
        if (is_integer) {
            num_layers = std::max(1, std::atoi(arg.c_str()));
            continue;
        }

        throw std::runtime_error("Unknown argument: " + arg);
    }

    auto build_effective_config = [&]() {
        EffectiveRunConfig cfg;
        cfg.num_layers = num_layers;
        cfg.data = data_files;
        cfg.warnings = config_warnings;
        cfg.vocab = vocab_path;
        cfg.build_vocab = build_vocab_out_path;
        cfg.prompt = prompt;
        cfg.vocab_size = vocab_size_target;
        cfg.val_split = val_split;
        cfg.temperature = temperature;
        cfg.top_k = top_k;
        cfg.top_p = top_p;
        cfg.rep_penalty = repetition_penalty;
        cfg.steps = generation_steps;
        cfg.resume = load_checkpoint_path;
        cfg.save_ckpt = save_checkpoint_path;
        cfg.epochs = train_epochs;
        cfg.ctx_window = train_context_window;
        cfg.lr = train_lr;
        cfg.optimizer = optimizer;
        cfg.warmup = warmup_steps;
        cfg.min_lr_ratio = min_lr_ratio;
        cfg.weight_decay = weight_decay;
        cfg.batch_size = batch_size;
        cfg.grad_clip = grad_clip_norm;
        cfg.report_every = report_every;
        cfg.report_lr = report_lr;
        cfg.report_grad = report_grad_norm;
        cfg.save_config = save_config_path;
        cfg.dry_run = dry_run;
        cfg.strict_config = strict_config;
        return cfg;
    };

    if (!save_config_path.empty()) {
        EffectiveRunConfig cfg = build_effective_config();
        write_effective_run_config_json(save_config_path, cfg);
    }

    if (dry_run) {
        EffectiveRunConfig cfg = build_effective_config();
        std::cout << effective_run_config_to_json_string(cfg);
        return 0;
    }

    for (const std::string& w : config_warnings) {
        std::cerr << "[config-warning] " << w << "\n";
    }

    std::string corpus = data_files.empty() ? default_corpus : load_dataset_from_files(data_files);

    if (!build_vocab_out_path.empty()) {
        std::vector<std::string> built = build_vocab_from_corpus_frequency(corpus, vocab_size_target);
        write_vocab_file(build_vocab_out_path, built);
        std::cout << "Wrote vocab file: " << build_vocab_out_path << " tokens=" << built.size() << "\n";
        return 0;
    }

    std::unique_ptr<Tokenizer> tokenizer;
    std::string tokenizer_mode = "char";
    if (!vocab_path.empty()) {
        std::vector<std::string> vocab_tokens = SubwordTokenizer::load_vocab_file(vocab_path);
        tokenizer = std::make_unique<SubwordTokenizer>(vocab_tokens);
        tokenizer_mode = "subword";
    } else {
        tokenizer = std::make_unique<CharTokenizer>(corpus);
    }

    Tokenizer& tok = *tokenizer;
    TinyTransformer model(tok.vocab_size(), 24, 48, 64, 1234, num_layers);

    bool resume_training = false;
    if (!load_checkpoint_path.empty()) {
        if (!model.load_checkpoint(load_checkpoint_path)) {
            throw std::runtime_error("Failed to load checkpoint: " + load_checkpoint_path);
        }
        resume_training = true;
    }

    std::vector<int> train_ids = tok.encode(corpus);
    auto split = split_train_val_ids(train_ids, val_split);
    const std::vector<int>& train_only_ids = split.first;
    const std::vector<int>& val_only_ids = split.second;

    float start_loss = model.evaluate_next_token_loss(train_only_ids, 24);
    float end_loss = model.train_next_token(train_ids,
                                            train_epochs,
                                            train_context_window,
                                            train_lr,
                                            report_every,
                                            optimizer,
                                            warmup_steps,
                                            min_lr_ratio,
                                            weight_decay,
                                            batch_size,
                                            grad_clip_norm,
                                            report_lr,
                                            report_grad_norm,
                                            val_split,
                                            !val_only_ids.empty(),
                                            resume_training);

    if (!save_checkpoint_path.empty()) {
        if (!model.save_checkpoint(save_checkpoint_path)) {
            throw std::runtime_error("Failed to save checkpoint: " + save_checkpoint_path);
        }
    }

    float val_loss = -1.0f;
    float val_ppl = -1.0f;
    if (val_only_ids.size() >= 2) {
        val_loss = model.evaluate_next_token_loss(val_only_ids, 24);
        val_ppl = std::exp(val_loss);
    }

    std::vector<int> prompt_ids = tok.encode(prompt);
    std::vector<int> greedy_ids = model.generate(prompt_ids, generation_steps, 32, 0.0f, 1, 7, 1.0f, 1.0f);
    std::vector<int> sampled_ids = model.generate(prompt_ids,
                                                  generation_steps,
                                                  32,
                                                  temperature,
                                                  top_k,
                                                  7,
                                                  top_p,
                                                  repetition_penalty);

    std::cout << "Tiny Transformer milestone (single-head, one-block)\n";
    std::cout << "tokenizer=" << tokenizer_mode << " vocab_size=" << tok.vocab_size() << " d_model=24 heads=1 blocks=1\n";
    std::cout << "Model has " << model.num_layers << " layer(s)\n";
    std::cout << "train_tokens=" << train_only_ids.size() << "";
    if (!val_only_ids.empty()) {
        std::cout << " val_tokens=" << val_only_ids.size();
    }
    std::cout << "\n";
    std::cout << "train_loss: start=" << start_loss << " end=" << end_loss << "\n";
    if (val_ppl > 0.0f) {
        std::cout << "val_loss=" << val_loss << " val_ppl=" << val_ppl << "\n";
    }
    std::cout << "prompt:    " << prompt << "\n";
    std::cout << "greedy:    " << tok.decode(greedy_ids) << "\n";
    std::cout << "sampled:   " << tok.decode(sampled_ids) << "\n";

    return 0;
}
#endif
