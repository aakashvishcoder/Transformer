#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <bits/stdc++.h>
using namespace std;

template <typename T>
class MinMaxScaler {
    public:
        vector<T> transform(const vector<T>& column) {
            vector<T> ans(column.size(),0);
            for(size_t i = 0; i < column.size(); i++) {
                ans[i] = (column[i]-min)/(max-min);
            }
            return ans;
        }
        vector<T> fit_transform(const vector<T>& column) {
            fit(column);
            return transform(column);
        }
        void fit(const vector<T>& column) {
            min = *min_element(column.begin(), column.end());
            max = *max_element(column.begin(), column.end());
        }
    private:
        T min;
        T max;
};

template <typename T>
class StandardScaler {
public:
    void fit(const vector<T>& column) {
        T sum = accumulate(column.begin(), column.end(),0);
        mean = sum/static_cast<T>(column.size());
        T difference = 0;
        for(size_t i = 0; i < column.size(); i++) {
            difference += pow((column[i]-mean),2);
        }
        std = sqrt(difference/static_cast<T>(column.size()));
    }

    vector<T> transform(const vector<T>& column) {
        vector<T> ans(column.size(),0);
        for(size_t i = 0; i < column.size(); i++) {
             ans[i] = (column[i]- mean)/std;
        }
        return ans;
    }

    vector<T> fit_transform(const vector<T>& column) {
        fit(column);
        return transform(column);
    }
private:
    T mean;
    T std;
};