#ifndef FRACTAL_HPP
#define FRACTAL_HPP
#include <vector>
#include <iostream>
#include <random>

/*
 * @fn random_k_d_combination 
 * @brief calculates k and d, where k^d <= N holds. 
 * @note N needs to be greater than 1. 
 *  restrictions: d>=2 if N>3. So d=1 only for the exception of N=2 (two outgoing edges). 
 * @param N (int)
 * @param (std::shared_ptr<std::mt19937_64>) generator
 * @return k,d (std::pair<int,int>): random value from all calculated values k^d <= N
 */
inline std::pair<int,int> random_k_d_combination(int N, std::shared_ptr<std::mt19937_64> generator){
    int k=2;
    int start;
    if(N <= 3){start = 1;}else{start = 2;}
    std::vector<std::pair<int, int>> results;

    while(pow(k,1)<=N){
        int d=start;
        while (pow(k,d)<=N) { 
            results.push_back({k,d});
            d++;
        }
    k++;
    }
    std::uniform_int_distribution<int> distributionUniform(0,results.size()-1);
    int randomIndex = distributionUniform(*generator);
    return results[randomIndex];
}

/*
 * @fn randomParameterCuts 
 * @brief draws N random numbers between (0,1). 
 * @param N (int)
 * @param (std::shared_ptr::mt19937_64) generator
 * @return parameter (std::vector<float>): drawn parameters and min = 0 and max = 1, so [0,v_1,v_2,...,1]
 */
inline std::vector<float> randomParameterCuts(int N, std::shared_ptr<std::mt19937_64> generator){
    std::vector<float> productionRuleParameter {0};
    for(int i=0; i<N; i++){
        std::uniform_real_distribution<float> distributionUniform(std::nextafter(0.0f, 1.0f), 1);
        float v = distributionUniform(*generator);
        productionRuleParameter.push_back(v);
    }
    productionRuleParameter.push_back(1);
    return productionRuleParameter;
}

/*
 * @fn sortAndDistance
 * @brief sort values and calculate the distances between them.
 *  Because of sorting the vector, the sum of the values are always 1.
 *  e.g. [0,0.4,0.1,0.5,1] -> [0,0.1,0.4,0.5,1] -> [0.1,0.3,0.1,0.5]
 * @param values (std::vector<float>)
 * @return values (std::vector<float>): sorted distances without min and max from passed values
 */
inline std::vector<float> sortAndDistance(std::vector<float> value){
    std::sort(value.begin(), value.end());
    for(int i=0; i<value.size()-1; i++){
       value[i] = value[i+1] - value[i];
    }
    value.pop_back();
    return value;
}

/*
 * @fn fractalLengths
 * @brief calcules fractal pattern of lengths like in a L-System
 * @param depth (int): depth of the fractal pattern
 * @param parameter (std::vector<float>): ratios of lengths 
 * @return lengths (std::vector<float>): lengths of fractal pattern as ratios between (0,1)
 */
inline std::vector<float> fractalLengths(int depth, std::vector<float> parameter){
    std::vector<float> lengths {1};
    std::vector<float> temp {1};
    for(int i=0; i<depth; i++){
        temp.clear();
        for(auto len : lengths){
            for(auto para : parameter){
                temp.push_back(len * para);
            }
        }
    lengths = temp;
    }
    return lengths;
}

#endif
