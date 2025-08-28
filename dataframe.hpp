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