#pragma once
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include "tensor.hpp"
using namespace std;

class DataFrame {
public:
    using T = float; // default value
    
    DataFrame() = default;

    void read_csv(const string& filename, bool header=true, char delimitter = ',') {
        ifstream file(filename);
        if(!file.is_open()) throw runtime_error("Cannot open file!");

        string line;
        if(header && getline(file,line)) parse_header(line,delimitter);

        vector<vector<T>> temp_columns(column_names_.empty() ? 0 : column_names_.size());

        while(getline(file,line)) {
            stringstream ss(line);
            string cell;
            size_t col_idx =0;
            vector<T> row_values;

            while(getline(ss, cell, delimitter)) {
                row_values.push_back(static_cast<T>(stod(cell)));
            }

            if(temp_columns.empty()) temp_columns.resize(row_values.size());
            for(size_t i =0; i < row_values.size(); i++) 
                temp_columns[i].push_back(row_values[i]);
        }

        n_rows_ = temp_columns.empty() ? 0 : temp_columns[0].size();
        columns_.size();
        for(auto& col_data : temp_columns) {
            columns_.emplace_back(Tensor<T,1>({col_data.size()},col_data));
        }

        if(column_names_.empty()) {
            for(size_t i =0; i < columns_.size(); i++) 
                column_names_.push_back("col" + to_string(i));
        }
    }

    void set_column_names(vector<string>& names) {
        column_names_ = names;
    }

    Tensor<T,1>& operator[](const string& col_name) {
        auto it = find(column_names_.begin(), column_names_.end(), col_name);
        if (it == column_names_.end()) throw runtime_error("Column not found!");
        return columns_[distance(column_names_.begin(), it)];
    }

    Tensor<T,2> toTensor() const {
        Tensor<T,2> result({n_rows_, columns_.size()});
        auto& data = result.get_data_ref();
        for(size_t col = 0; col < columns_.size(); col++) {
            const auto& col_data = columns_[col].get_data();
            for(size_t row = 0; row < n_rows_; row++) 
                data[row * columns_.size() + col]= col_data[row];
        }
        return result;
    }

    void print() const {
        for(size_t i = 0; i < n_rows_; i++) {
            for(size_t j = 0; j < columns_.size(); j++) {
                cout << columns_[j].get_data()[i];
                if(j < columns_.size() - 1) cout << ",";
            }
            cout << "\n";
        }
    }

private:
    void parse_header(const string& line, char delimitter) {
        stringstream ss(line);
        string col_name;
        column_names_.clear();
        while(getline(ss,col_name, delimitter)) column_names_.push_back(col_name);
    }
    
    size_t n_rows_ = 0;
    vector<string> column_names_;
    vector<Tensor<T,1>> columns_;
};