#pragma once
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iostream>
#include <variant>
#include <stdexcept>
#include <algorithm>

using namespace std;

class DataFrame {
public:
    using Cell = variant<int, float, double, string, bool>;

    DataFrame() = default;

    void read_csv(const string& filename, bool header=true, char delimitter = ',') {
        ifstream file(filename);
        if (!file.is_open()) throw runtime_error("Cannot open file!");

        string line;
        if(header && getline(file,line)) parse_header(line, delimitter);

        vector<vector<Cell>> temp_columns;

        size_t row_idx = 0;
        while (getline(file, line)) {
            stringstream ss(line);
            string cell;
            vector<Cell> row_values;

            while(getline(ss, cell, delimitter)) {
                row_values.push_back(parse_cell(cell));
            }

            if (temp_columns.empty()) {
                temp_columns.resize(row_values.size());
            } else if(row_values.size() != temp_columns.size()) {
                throw runtime_error("Row " + to_string(row_idx) + " has wrong number of columns");
            }

            for (size_t i = 0; i < row_values.size(); i++)
                temp_columns[i].push_back(row_values[i]);

            row_idx++;
        }

        n_rows_ = row_idx;
        columns_ = move(temp_columns);

        // Default column names
        if (column_names_.empty()) {
            for (size_t i = 0; i < columns_.size(); i++)
                column_names_.push_back("col" + to_string(i));
        }
    }

    void set_column_names(vector<string>& names) {
        if (names.size() != columns_.size())
            throw runtime_error("Column name count mismatch");
        column_names_ = names;
    }

    vector<Cell>& operator[](const string& col_name) {
        auto it = find(column_names_.begin(), column_names_.end(), col_name);
        if (it == column_names_.end()) throw runtime_error("Column not found!");
        return columns_[distance(column_names_.begin(), it)];
    }

    void print() const {
        for(size_t i = 0; i < n_rows_; i++) {
            for(size_t j = 0; j < columns_.size(); j++) {
                visit([](auto&& val){ cout << val; }, columns_[j][i]);
                if(j < columns_.size()-1) cout << ",";
            }
            cout << "\n";
        }
    }

private:
    Cell parse_cell(const string& s) {
        try {
            if (s == "true" || s == "TRUE") return true;
            if (s == "false" || s == "FALSE") return false;

            if (s.find('.') != string::npos) return stod(s);
            return stoi(s);
        } catch(...) {
            return s;
        }
    }

    void parse_header(const string& line, char delimitter) {
        stringstream ss(line);
        string col_name;
        column_names_.clear();
        while(getline(ss, col_name, delimitter)) column_names_.push_back(col_name);
    }

    size_t n_rows_ = 0;
    vector<string> column_names_;
    vector<vector<Cell>> columns_;
};
