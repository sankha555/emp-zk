#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__

#pragma once

#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "emp-zk/ai/ai.h"
#include "emp-zk/ai/secure-utils.h"
#include <iostream>

using namespace emp;
using namespace std;

template <typename T>
class ParametersVerification {
    public:

    // [m x (n + 1)] matrix [+1 for bias]
    int m;
    int n;
    int party = PUBLIC;

    T* param_matrix;

    ParametersVerification(int m, int n, int party = ALICE){
        this->m = m;
        this->n = n;
        this->party = party;
        this->param_matrix = new T[m*(n+1)];
    }

    int num_parameters(){
        return m*(n+1);
    }


    void read_weights_and_biases(const char* filepath, int offset){
        read_weights(filepath, offset);
        read_biases(filepath, offset + m*n);
    }

    void read_weights(const char* filepath, int offset){
        float* raw_weights = new float[m*n];
        if(this->party != BOB){
            read_next_elements(m*n, raw_weights, offset, filepath);
        }

        if constexpr (std::is_same<T, IntFp>::value){
            IntFp* temp_weights = new IntFp[m*n];
            authenticate_over_field(m*n, raw_weights, temp_weights, this->party);
            
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n; j++){
                    param_matrix[i*(n+1) + j] = temp_weights[i*n + j];
                }
            }
            
            delete[] temp_weights;

        } else if constexpr (std::is_same<T, float>::value) {
            for(int i = 0; i < m; i++){
                for(int j = 0; j < n; j++){
                    param_matrix[i*(n+1) + j] = (raw_weights[i*n + j]);
                }
            }
        }

        delete[] raw_weights;
    }

    void read_biases(const char* filepath, int offset){
        float* raw_biases = new float[m];

        if(this->party != BOB){
            read_next_elements(m, raw_biases, offset, filepath);
        }

        if constexpr (std::is_same<T, IntFp>::value){
            IntFp* temp_biases = new IntFp[m];
            authenticate_over_field(m, raw_biases, temp_biases, this->party);

            for(int i = 0; i < m; i++){
                param_matrix[(i+1)*n + i] = temp_biases[i];
            }

            delete[] temp_biases;

        } else if constexpr (std::is_same<T, float>::value) {
            for(int i = 0; i < m; i++){
                param_matrix[(i+1)*n + i] = (raw_biases[i]);
            }
        }

        delete[] raw_biases;
    }

    void print_parameters(){
        for(int i = 0; i < m; i++){
            for(int j = 0; j < n+1; j++){
                if constexpr (std::is_same<T, IntFp>::value){
                    cout << format_EMP_IntFp(param_matrix[i*(n+1)+j], 1) << " ";
                } else if constexpr (std::is_same<T, float>::value) {
                    cout << param_matrix[i*(n+1)+j] << " ";
                }
            }
            cout << "\n";
        }
    }
};


template <typename T>
class Kernel2D {
    public:

    int out_c;
    int in_c;
    int h;
    int w;
    int party = PUBLIC;


    int params_per_out_channel;
    int params_per_in_channel;


    T* filter_matrix;

    Kernel2D(int out_c, int in_c, int h, int w, int party = ALICE){
        this->out_c = out_c;
        this->in_c = in_c;
        this->h = h;
        this->w = w;
        this->party = party;
        this->filter_matrix = new T[out_c * in_c * h * w + out_c];


        this->params_per_out_channel = this->in_c * this->h * this->w;
        this->params_per_in_channel = this->h * this->w;
    }

    int num_parameters(){
        return out_c * in_c * h * w + out_c;
    }

    int num_weights(){
        return out_c * in_c * h * w;
    }

    void read_filters(const char* filepath, int offset){
        float* raw_weights = new float[this->num_parameters()];
        if(this->party != BOB){
            read_next_elements(this->num_parameters(), raw_weights, offset, filepath);
        }

        if constexpr (std::is_same<T, IntFp>::value){
            IntFp* temp_weights = new IntFp[this->num_parameters()];
            authenticate_over_field(this->num_parameters(), raw_weights, temp_weights, this->party);
            
            for(int q = 0; q < this->out_c; q++){
                // for each out channel

                for(int p = 0; p < this->in_c; p++){
                    // for each in channel

                    for(int i = 0; i < this->h; i++){
                        for(int j = 0; j < this->w; j++){
                            this->filter_matrix[
                                q * params_per_out_channel +
                                p * params_per_in_channel +
                                i * this->w +
                                j
                            ] = temp_weights[
                                q * params_per_out_channel +
                                p * params_per_in_channel +
                                i * this->w +
                                j
                            ];

                            // cout << format_EMP_IntFp(this->filter_matrix[
                            //     q * params_per_out_channel +
                            //     p * params_per_in_channel +
                            //     i * this->w +
                            //     j
                            // ], 1) << " ";
                        }
                        // cout << "\n";
                    }
                }
            }

            for(int q = 0; q < this->out_c; q++){
                this->filter_matrix[this->out_c * this->params_per_out_channel + q] = temp_weights[this->out_c * this->params_per_out_channel + q];
                // cout << format_EMP_IntFp(this->filter_matrix[this->out_c * this->params_per_out_channel + q], 1) << " ";
            }
            // cout << "\n";


            delete[] temp_weights;

        } else if constexpr (std::is_same<T, float>::value) {

            for(int q = 0; q < this->out_c; q++){
                // for each out channel

                for(int p = 0; p < this->in_c; p++){
                    // for each in channel

                    for(int i = 0; i < this->h; i++){
                        for(int j = 0; j < this->w; j++){
                            this->filter_matrix[
                                q * params_per_out_channel +
                                p * params_per_in_channel +
                                i * this->w +
                                j
                            ] = raw_weights[
                                q * params_per_out_channel +
                                p * params_per_in_channel +
                                i * this->w +
                                j
                            ];
                        }
                    }
                }
            }

            for(int q = 0; q < this->out_c; q++){
                this->filter_matrix[this->out_c * this->params_per_out_channel + q] = raw_weights[this->out_c * this->params_per_out_channel + q];
            }
        }

        delete[] raw_weights;
    }


    T* get_flattened_weights(int q){
        return this->filter_matrix + q * this->params_per_out_channel;
    }


    void print_parameters(){
        ;
    }
};

#endif