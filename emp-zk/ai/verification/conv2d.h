#ifndef __CONV2D_H__
#define __CONV2D_H__

#pragma once

#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "emp-zk/ai/verification/verification.h"
#include "emp-zk/ai/utils.h"
#include "emp-zk/ai/inference/affine.h"
#include "emp-zk/ai/inference/normalize.h"
#include "emp-zk/emp-zk-math/ZKmath-functions.h"
#include "emp-zk/ai/secure-utils.h"

#include <iostream>

using namespace emp;
using namespace std;




template <typename T>
class Conv2D : public Layer<T> {
    public:

    int in_channels;
    int out_channels;
    int image_h;
    int image_w;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int out_h;
    int out_w;

    Kernel2D<T>* kernel;
    T* pixel_buffer;

    void set_output_image_sizes(){
        this->out_h = (int) (this->image_h + 2*this->pad_h - this->kernel_h)/this->stride_h + 1;

        this->out_w = (int) (this->image_w + 2*this->pad_w - this->kernel_w)/this->stride_w + 1;
    }
    
    Conv2D(
        int in_channels, int out_channels, int image_h, int image_w, 
        int h, int w, int stride_h, int stride_w, int pad_h, int pad_w,
        int max_coeffs = -1, int party = PUBLIC
    ) : Layer<T>(-1, -1, -1, party) {
        this->party = party;
        this->type = LAYER_TYPE::CONV2D;

        this->in_channels = in_channels;
        this->out_channels = out_channels;
        this->image_h = image_h;
        this->image_w = image_w;
        this->kernel_h = h;
        this->kernel_w = w;
        this->stride_h = stride_h;
        this->stride_w = stride_w;
        this->pad_h = pad_h;
        this->pad_w = pad_w;

        // set feature map sizes
        this->input_size = in_channels * image_h * image_w;
        this->input = new T[this->input_size+1];  // +1 for bias

        this->set_output_image_sizes();
        this->output_size = this->out_channels * this->out_h * this->out_w;
        this->output = new T[this->output_size];


        this->kernel = new Kernel2D<T>(this->out_channels, this->in_channels, this->kernel_h, this->kernel_w, this->party);
        this->pixel_buffer = new T[this->in_channels * this->kernel_h * this->kernel_w];

        // ai tools
        this->lower_bounds = new T[this->output_size];
        this->upper_bounds = new T[this->output_size];
        
        if(max_coeffs == -1){
            max_coeffs = this->input_size+1;
        }
        this->max_coeffs = max_coeffs;


        this->lower_constraints = new T[this->output_size*this->max_coeffs];
        this->upper_constraints = new T[this->output_size*this->max_coeffs];

        this->backsubstituted_lower_constraints= new T[this->output_size*this->max_coeffs];
        this->backsubstituted_upper_constraints= new T[this->output_size*this->max_coeffs];

        
    }

    void forward(Layer<T>* input_layer, Layer<T>* prev_layer, bool do_inference = true){
        this->prev_layer = prev_layer;

        // clone the input
        for(int i = 0; i < this->input_size; i++){
            this->input[i] = prev_layer->output[i];
        }

        if(!ONLY_INFERENCE){
            ;
        }

        if(do_inference){
            inference();
        }

        this->describe(false, false);
    }

    void load_flattened_pixels(int y, int x){
        for(int c = 0; c < this->in_channels; c++){
            for(int i = 0; i < this->kernel_h; i++){
                for(int j = 0; j < this->kernel_w; j++){
                    this->pixel_buffer[
                        c * this->kernel_h * this->kernel_w +
                        i * this->kernel_w +
                        j
                    ] = this->input[
                        c * this->image_h * this->image_w +
                        (y + i) * this->image_w +
                        (x + j)
                    ];
                }
            }
        }
    }

    void inference(){
        if constexpr (std::is_same<T, IntFp>::value && SECURE){
            // cerr << "PARTY: " << this->party << "\n";
            for(int y = 0; y < this->out_h; y++){
                for(int x = 0; x < this->out_w; x++){

                    for(int z = 0; z < this->out_channels; z++){
                        T* mask = this->kernel->get_flattened_weights(z);
                        // for(int i = 0; i < this->kernel->params_per_out_channel; i++){
                        //     cout << format_EMP_IntFp(mask[i], 1) << " ";
                        // }
                        // cout << "\n";
                        
                        load_flattened_pixels(stride_h * y, stride_w * x);

                        this->output[
                            z * this->out_h * this->out_w +
                            y * this->out_w +
                            x
                        ] = inner_product_bundle(
                            this->in_channels * this->kernel_h * this->kernel_w,
                            mask,
                            this->pixel_buffer,
                            this->party
                        );
                    }
                }
            }

            ZKgeneralTruncAny(this->party, this->output, this->output, this->output_size, FXPSCALE);

            int channel = 0;
            for(int z = 0; z < this->out_channels; z++){
                T bias = this->kernel->filter_matrix[
                        this->kernel->num_weights() + channel
                    ];

                for(int y = 0; y < this->out_h; y++){
                    for(int x = 0; x < this->out_w; x++){
                        this->output[
                            z * this->out_h * this->out_w +
                            y * this->out_w +
                            x
                        ] = this->output[
                            z * this->out_h * this->out_w +
                            y * this->out_w +
                            x
                        ] + bias;

                    }
                }

                channel++;
            }

        } else {
            cleartext_inference();
        }
    }

    void cleartext_inference(){
        

        for(int y = 0; y < this->out_h; y++){
            for(int x = 0; x < this->out_w; x++){

                for(int z = 0; z < this->out_channels; z++){
                    T* mask = this->kernel->get_flattened_weights(z);

                    load_flattened_pixels(stride_h * y, stride_w * x);
                    this->output[
                        z * this->out_h * this->out_w +
                        y * this->out_w +
                        x
                    ] = inner_product_emp(
                        this->in_channels * this->kernel_h * this->kernel_w,
                        mask,
                        this->pixel_buffer
                    );
                }
            }
        }

        if constexpr (std::is_same<T, IntFp>::value){
            normalize(this->output_size, this->output, this->output);
        }

        int channel = 0;
        for(int z = 0; z < this->out_channels; z++){
            T bias = this->kernel->filter_matrix[
                        this->kernel->num_weights() + channel
                    ];

            for(int y = 0; y < this->out_h; y++){
                for(int x = 0; x < this->out_w; x++){
                    this->output[
                        z * this->out_h * this->out_w +
                        y * this->out_w +
                        x
                    ] += bias;
                }
            }

            channel++;
        }
    }


    void compute_lower_bounds(){

        if constexpr (std::is_same<IntFp, T>::value && SECURE){

        } else {
            
        }

    }

    void compute_upper_bounds(){

        if constexpr (std::is_same<IntFp, T>::value && SECURE){
            
        } else {
        }

    }


    void compute_lower_constraints(){
        // l_i ≤ x_i ≤ u_i for input layer
        for(int i = 0; i <  this->output_size; i++){
        }
    }

    void compute_upper_constraints(){
        // l_i ≤ x_i ≤ u_i for input layer
        for(int i = 0; i <  this->output_size; i++){
        }
    }


    void backsubstitute(Layer<T>* input_layer){
        
        if(DO_DP_BS){

        } else {

        }
    }


    void reset(){
        delete[] this->input;
        delete[] this->output;
        delete[] this->lower_bounds;
        delete[] this->upper_bounds;
        delete[] this->lower_constraints;
        delete[] this->upper_constraints;

        this->type = LAYER_TYPE::CONV2D;

        // set feature map sizes
        this->input_size = in_channels * image_h * image_w;
        this->input = new T[this->input_size+1];  // +1 for bias

        this->set_output_image_sizes();
        this->output_size = this->out_channels * this->out_h * this->out_w;
        this->output = new T[this->output_size];


        // this->kernel = new Kernel2D<T>(this->out_channels, this->in_channels, this->kernel_h, this->kernel_w, this->party);
        this->pixel_buffer = new T[this->in_channels * this->kernel_h * this->kernel_w];

        // ai tools
        this->lower_bounds = new T[this->output_size];
        this->upper_bounds = new T[this->output_size];
        
        if(this->max_coeffs == -1){
            this->max_coeffs = this->input_size+1;
        }
        this->max_coeffs = this->max_coeffs;


        this->lower_constraints = new T[this->output_size*this->max_coeffs];
        this->upper_constraints = new T[this->output_size*this->max_coeffs];

        this->backsubstituted_lower_constraints= new T[this->output_size*this->max_coeffs];
        this->backsubstituted_upper_constraints= new T[this->output_size*this->max_coeffs];
    }

    void describe(bool print_parameters = true, bool print_expressions = false){
        cout << "Type: " << get_layer_type(this->type) << "\n";


        if (1){
            std::cout << std::fixed << std::setprecision(4);

            cout << "Inputs:\n";
            
            int in_c = ((Conv2D<T>*) this)->in_channels;
            int in_h = ((Conv2D<T>*) this)->image_h;
            int in_w = ((Conv2D<T>*) this)->image_w;

            for(int k = 0; k < in_c; k++){
                cout << "\tChannel " << k << ":\n";
                for(int i = 0; i < in_h; i++){
                    cout << "\tRow " << i << ": [";
                    for(int j = 0; j < in_w; j++){
                        T el = this->input[
                            k * in_h * in_w +
                            i * in_w +
                            j
                        ];
                        
                        if constexpr (std::is_same<T, IntFp>::value){
                            cout << format_EMP_IntFp(el, 1);
                        } else if constexpr (std::is_same<T, float>::value) {
                            cout << el;
                        }
                        
                        cout << (j == in_w-1 ? "" : ", ");
                    } 
                    cout << "]\n";
                }
                cout << "\n";
            }

            int out_c = ((Conv2D<T>*) this)->out_channels;
            int out_h = ((Conv2D<T>*) this)->out_h;
            int out_w = ((Conv2D<T>*) this)->out_w;

            cout << "Outputs:\n";
            for(int k = 0; k < out_c; k++){
                cout << "\tChannel " << k << ":\n";
                for(int i = 0; i < out_h; i++){
                    cout << "\tRow " << i << ": [";
                    for(int j = 0; j < out_w; j++){
                        T el = this->output[
                            k * out_h * out_w +
                            i * out_w +
                            j
                        ];
                        
                        if constexpr (std::is_same<T, IntFp>::value){
                            cout << format_EMP_IntFp(el, 1);
                        } else if constexpr (std::is_same<T, float>::value) {
                            cout << el;
                        }
                        
                        cout << (j == out_w-1 ? "" : ", ");

                    } 
                    cout << "]\n";
                }
                cout << "\n";
            }

            std::cout.unsetf(std::ios::fixed);
        } else {
            cout << "Inputs:\n";

            for(int i = 0; i < this->input_size; i++){
                if constexpr (std::is_same<T, IntFp>::value){
                    cout << format_EMP_IntFp(this->input[i], 1) << " ";
                } else if constexpr (std::is_same<T, float>::value) {
                    cout << this->input[i] << " ";
                }
            }
            cout << "\n";
            
            cout << "Outputs:\n";

            int t = 0;

            for(int y = 0; y < this->out_h; y++){
                for(int x = 0; x < this->out_w; x++){
                    for(int z = 0; z < this->out_channels; z++){

                        T el = this->output[
                            z * this->out_h * this->out_w +
                            y * this->out_w +
                            x 
                        ];

                        if constexpr (std::is_same<T, IntFp>::value){
                            cout << t << ": " << format_EMP_IntFp(el, 1) << "\n";
                        } else if constexpr (std::is_same<T, float>::value) {
                            cout << t << ": " << el << "\n";
                        }

                        t++;
                    }
                    
                }
            }

        }

        
        // for(int i = 0; i < this->output_size; i++){
        //     cout << i << ": ";
        //     if constexpr (std::is_same<T, IntFp>::value){
        //         cout << format_EMP_IntFp(this->output[i], 1) << "\n";
        //     } else if constexpr (std::is_same<T, float>::value) {
        //         cout << this->output[i] << "\n";
        //     }
        // }
        cout << "\n";
    }


    void cleartext_compute_lower_constraints(){
    }

    void cleartext_compute_upper_constraints(){
    }

    void cleartext_compute_lower_bounds(){
    }

    void cleartext_compute_upper_bounds(){
    }



    
    void backsubstitute_lower_constraints(int num_inputs){
        
    }


    void backsubstitute_upper_constraints(int num_inputs){
        
    }

    void compute_lower_bounds_after_backsubstitution(Layer<T>* input_layer){
        
    }

    void compute_upper_bounds_after_backsubstitution(Layer<T>* input_layer){
        

    }

};


#endif