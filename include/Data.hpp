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
        std::vector<std::vector<float*>> X;
        std::vector<float*> y;
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
         * @note X and y are pointers to dt 
         */
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
         */
        void minMaxFeatures(){
            for(int i=0; i<XIndices.size(); i++){ // for each feature
                float min = dt[0][XIndices[i]]; 
                float max = dt[0][XIndices[i]]; 
                for(int k=1; k<dt.size(); k++){ // for each row in dt 
                    if(dt[k][XIndices[i]]<min){ // find min from dt with values from XIndices
                        min=dt[k][XIndices[i]];
                    } else if(dt[k][XIndices[i]]>max){
                        max=dt[k][XIndices[i]];
                    }
                }
                minX.push_back(min);
                maxX.push_back(max);
            }
        }
};
#endif

