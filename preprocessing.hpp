#pragma once
#include "tensor.hpp"
#include <sstream>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>
#include <iostream>
using namespace std;

class SimpleTokenizer {
public:
    unordered_map<string, int> vocab_to_idx;
    unordered_map<int,string> idx_to_vocab;

    SimpleTokenizer() {
        add_token("<unk>"); // unknown token
        add_token("<pad>"); // pad token
    }

    int add_token(const string& token) {
        if(vocab_to_idx.find(token) == vocab_to_idx.end()) {
            int idx = vocab_to_idx.size();
            vocab_to_idx[token] = idx;
            idx_to_vocab[idx] = token;
        }
        return vocab_to_idx[token];
    }

    Tensor<int,1> encode(const string& text) {
        Tensor<int,1> output;
        vector<int> idxs;
        stringstream ss(text);
        string token;
        while(ss >> token) {
            if(vocab_to_idx.find(token) == vocab_to_idx.end()) {
                idxs.push_back(vocab_to_idx["<unk>"]);
            } else {
                idxs.push_back(vocab_to_idx[token]);
            }
        }
        output.get_data_ref() = idxs;
        return output;
    }   

    string decode(const Tensor<int,1>& idxs) {
        string result;
        for(size_t i = 0; i < idxs.get_data().size(); i++) {
            result += idx_to_vocab[idxs.get_data_ref()[i]];
            if(i != idxs.get_data().size()-1) result += " ";
        }
        return result;
    }
};

class BPE {
private:
    int next_token_id = 256;
    int vocab_limit = 1000;

    map<pair<int, int>, int> merges;             // Pair → new token ID
    unordered_map<int, pair<int, int>> reverse_merges;  // Token ID → pair (for decoding)
    unordered_map<int, string> id_to_token;      // Token ID → string
    unordered_map<string, int> token_to_id;      // string → token ID

public:
    BPE(int vocab_size_limit = 1000) : vocab_limit(vocab_size_limit) {
        // Initialize basic tokens (ASCII)
        for (int i = 32; i < 126; ++i) {
            string s(1, static_cast<char>(i));
            token_to_id[s] = i;
            id_to_token[i] = s;
        }
    }

    // Tokenize a string into vector of ints (initial)
    vector<int> tokenize_string(const string& str) {
        vector<int> tokens;
        for (char c : str)
            tokens.push_back((int)(unsigned char)c);
        return tokens;
    }

    // Convert vector<int> back to string
    string decode(const vector<int>& tokens) {
        string result;
        for (int id : tokens)
            result += decode_token(id);
        return result;
    }

    string decode_token(int token_id) {
        if (token_id < 256)
            return string(1, static_cast<char>(token_id));

        auto it = reverse_merges.find(token_id);
        if (it == reverse_merges.end())
            return id_to_token[token_id];

        string left = decode_token(it->second.first);
        string right = decode_token(it->second.second);
        return left + right;
    }

    void train(const vector<string>& corpus) {
        vector<vector<int>> tokenized_corpus;
        for (const auto& s : corpus)
            tokenized_corpus.push_back(tokenize_string(s));

        while (next_token_id < vocab_limit) {
            map<pair<int, int>, int> pair_freq;

            for (const auto& sentence : tokenized_corpus) {
                for (size_t i = 0; i + 1 < sentence.size(); ++i)
                    pair_freq[{sentence[i], sentence[i + 1]}]++;
            }

            if (pair_freq.empty())
                break;

            // Find most frequent pair
            auto best = max_element(pair_freq.begin(), pair_freq.end(),
                [](const auto& a, const auto& b) {
                    return a.second < b.second;
                });

            auto best_pair = best->first;

            // Assign new token ID
            int new_token = next_token_id++;
            merges[best_pair] = new_token;
            reverse_merges[new_token] = best_pair;

            // Create readable string for merged token
            string merged_string = decode_token(best_pair.first) + decode_token(best_pair.second);
            id_to_token[new_token] = merged_string;
            token_to_id[merged_string] = new_token;

            cout << "Merged (" << best_pair.first << ", " << best_pair.second << ") -> " << new_token << "\n";

            // Replace in corpus
            for (auto& sentence : tokenized_corpus) {
                vector<int> new_sentence;
                size_t i = 0;
                while (i < sentence.size()) {
                    if (i + 1 < sentence.size() &&
                        sentence[i] == best_pair.first &&
                        sentence[i + 1] == best_pair.second) {
                        new_sentence.push_back(new_token);
                        i += 2;
                    } else {
                        new_sentence.push_back(sentence[i]);
                        i += 1;
                    }
                }
                sentence = new_sentence;
            }
        }
    }

    // Encode input string
    Tensor<int,1> encode(const string& input) {
        vector<int> tokens = tokenize_string(input);
        Tensor<int,1> output;
        bool changed = true;
        while (changed) {
            changed = false;

            for (const auto& [pair, id] : merges) {
                vector<int> new_tokens;
                size_t i = 0;
                while (i < tokens.size()) {
                    if (i + 1 < tokens.size() &&
                        tokens[i] == pair.first &&
                        tokens[i + 1] == pair.second) {
                        new_tokens.push_back(id);
                        i += 2;
                        changed = true;
                    } else {
                        new_tokens.push_back(tokens[i]);
                        i++;
                    }
                }
                tokens = new_tokens;
            }
        }
        output.get_data_ref() = tokens;
        return output;
    }

    void print_token_map() const {
        cout << "\nToken ID → Token String:\n";
        for (const auto& [id, str] : id_to_token)
            cout << id << ": \"" << str << "\"\n";
    }

    // Save / Load merges
    void save_merges(const string& filename) {
        ofstream out(filename);
        for (const auto& [pair, id] : merges)
            out << pair.first << " " << pair.second << " " << id << "\n";
        out.close();
    }

    void load_merges(const string& filename) {
        ifstream in(filename);
        int a, b, id;
        while (in >> a >> b >> id) {
            merges[{a, b}] = id;
            reverse_merges[id] = {a, b};
            string s = decode_token(a) + decode_token(b);
            id_to_token[id] = s;
            token_to_id[s] = id;
            next_token_id = max(next_token_id, id + 1);
        }
    }
};
