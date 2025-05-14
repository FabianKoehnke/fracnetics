#ifndef DATA_HPP
#define DATA_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <span>

// TODO write docstring 
class Data {
    public:
        
        std::vector<std::vector<float>> dt;
        std::vector<std::vector<float*>> X;
        std::vector<float*> y;
        
        void readCSV(const std::string& filename, bool header=true) {
            std::ifstream file(filename);
            std::string line;
            
            if (!file.is_open()) {
                std::cerr << "Error opening the file!" << std::endl;
                return;
            }
            
            if(header == true){
                std::getline(file, line);
            }
            while (std::getline(file, line)) {
                std::vector<float> row;
                std::stringstream ss(line);
                std::string cell;
                
                while (std::getline(ss, cell, ',')) {
                    row.push_back(std::stof(cell));
                }
                dt.push_back(row);
            }
            
            file.close();
        }

        void printRows(int nrows){
            for(int i=0; i<nrows; i++){
                for(float val : dt[i]){
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
        }

        void xySplit(){
            X.resize(dt.size());// set number of rows in X
            for(int i=0; i<dt.size(); i++){
                y.push_back(&dt[i].back());
                X[i].resize(dt[i].size()-1);// set number of columns in X 
                for(int k=0; k<dt[i].size()-1; k++){
                    X[i][k] = &dt[i][k];
                }
            }
        }
};
#endif

