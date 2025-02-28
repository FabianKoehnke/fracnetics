#ifndef DATA_HPP
#define DATA_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

class Data {
public:
    int test;
    std::vector<std::vector<float> > readCSV(const std::string& filename) {
        std::vector<std::vector<float> > data;
        std::ifstream file(filename);
        std::string line;
        
        if (!file.is_open()) {
            std::cerr << "Fehler beim Öffnen der Datei!" << std::endl;
            return data;
        }
        
        std::getline(file, line); // skip header
        while (std::getline(file, line)) {
            std::vector<float> row;
            std::stringstream ss(line);
            std::string cell;
            
            while (std::getline(ss, cell, ',')) {
                row.push_back(std::stof(cell));
            }
            data.push_back(row);
        }
        
        file.close();
        return data;
    }
};

#endif

