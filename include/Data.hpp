#ifndef DATA_HPP
#define DATA_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

/**
 * @class Data 
 * @brief Read and transform csv data 
*/
class Data {
    public:
        
        std::vector<std::vector<float>> dt;
        std::vector<std::vector<float>> X;
        std::vector<float> y;
        std::vector<int> yIndices;
        std::vector<int> XIndices;
        std::vector<float> minX;
        std::vector<float> maxX;
        
        /**
         * @fn readCSV
         * @brief read csv data and stores them in member dt.
         * 
         * @param filename (string&)
         * @param header (bool): skip first row if true (defaut)
         */
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

        /**
         * @fn printRows
         * @brief print rows of member dt.
         * @param nrows (int)
         */
        void printRows(int nrows){
            for(int i=0; i<nrows; i++){
                for(float val : dt[i]){
                    std::cout << val << " ";
                }
                std::cout << std::endl;
            }
        }

        /**
         * @fn xySplit 
         * @brief splits dt (data) in X (features) and y (target) and stores them as member 
         * @param yIndex (int) : index of the target variable y
         * @parma xIndices (std:vector<int>) : indices of features X
         */
        void xySplit(int yIndex, std::vector<int>& xIndices){
            X.resize(dt.size());// set number of rows in X
            for(int i=0; i<dt.size(); i++){
                y.push_back(dt[i][yIndex]);
                X[i].resize(xIndices.size());// set number of columns in X 
                for(int k = 0; k<xIndices.size(); k++){
                    int xI = xIndices[k];
                    X[i][k] = dt[i][xI];
                }
            }
        }
        
        /**
         * @fn columnSelector
         * @brief specify the X and y columns for selection of dt.
         * @note e.g. iy = (1,3) selects columsn 1 and 2.
         * @param iy (pair) : y column selector 
         * @param iX (pair) : X column selector 
         */
        void columnSelector(std::pair<int,int>iy, std::pair<int,int>iX){
            int N = iy.second-iy.first;
            for(int i=0; i<N; i++){
                yIndices.push_back(iy.first+i);
            }
            N = iX.second-iX.first;
            for(int i=0; i<N; i++){
                XIndices.push_back(iX.first+i);
            }
        }

        /**
         * @fn minMaxFeatures
         * @brief finds min and max of the features (X).
         * @note stored in std::vector<float> minX and maxX
         * @param X (std::vector<std::vector<float>>X) : table of features 
         */
        void minMaxFeatures(std::vector<std::vector<float>> features){
            for(int i=0; i<features[0].size(); i++){ // for each feature
                float min = features[0][i]; 
                float max = features[0][i]; 
                for(int k=1; k<features.size(); k++){ // for each row in dt 
                    if(features[k][i]<min){ // find min from dt with values from featuresIndices
                        min=features[k][i];
                    } else if(features[k][i]>max){
                        max=features[k][i];
                    }
                }
                minX.push_back(min);
                maxX.push_back(max);
            }
        }
};
#endif

