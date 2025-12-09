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

    int* pred_neuron_ids;

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
            max_coeffs = this->kernel->params_per_out_channel + 1;  // +1 for bias
        }
        this->max_coeffs = max_coeffs;
        this->pred_neuron_ids = new int[this->output_size * this->max_coeffs];

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
            
            for(int z = 0; z < this->out_channels; z++){
                for(int y = 0; y < this->out_h; y++){
                    for(int x = 0; x < this->out_w; x++){
                        int nid = z * this->out_h * this->out_w + y * this->out_w + x;
                        load_predecessor_neurons(nid, stride_h * y, stride_w * x);
                    }
                }
            }

            compute_lower_constraints();
            compute_upper_constraints();

            compute_lower_bounds();
            compute_upper_bounds();

            // this->print_predecessor_ids();

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

    void load_predecessor_neurons(int nid, int y, int x){
        int preds = 0;
        for(int c = 0; c < this->in_channels; c++){
            for(int i = 0; i < this->kernel_h; i++){
                for(int j = 0; j < this->kernel_w; j++){
                    this->pred_neuron_ids[
                        nid * this->max_coeffs
                      + preds
                    ] = c * this->image_h * this->image_w +
                        (y + i) * this->image_w +
                        (x + j);

                    preds++;
                }
            }
        }

        // bias
        this->pred_neuron_ids[
            nid * this->max_coeffs
            + preds
        ] = -1;
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
        T* prev_lbs = ((Layer<T>*) this->prev_layer)->lower_bounds;
        T* prev_ubs = ((Layer<T>*) this->prev_layer)->upper_bounds;

        if constexpr (std::is_same<IntFp, T>::value && SECURE){
            IntFp* coeff_sign = new IntFp[2 * (this->max_coeffs - 1)]; 
            T* copied_lc = new T[2*(this->max_coeffs - 1)];
            T* prev_bounds = new T[2*(this->max_coeffs - 1)];       // lb_1 lb_2 ... lb_m   ub_1 ub_2 ... ub_m  
            
            for(int i = 0; i < this->output_size; i++){

                for(int j = 0; j < this->max_coeffs - 1; j++){
                    int j_th_predecessor = this->pred_neuron_ids[i * this->max_coeffs + j];

                    prev_bounds[j] = prev_lbs[j_th_predecessor];
                    prev_bounds[j + this->max_coeffs - 1] = prev_ubs[j_th_predecessor];
                }

                auto start = clock_start();


                // cerr << "N" << i << "\n";
                ZKcmpPositive(this->party, this->lower_constraints + i*this->max_coeffs, ZERO_COMP_CONSTANT, coeff_sign, this->max_coeffs - 1);
                
                for(int j = 0; j < this->max_coeffs - 1; j++){
                    coeff_sign[j + this->max_coeffs - 1] = FIELD_ONE + coeff_sign[j].negate();
                }

                // first select which bound to multiply based on sign
                for(int j = 0; j < 2*(this->max_coeffs - 1); j++){
                    coeff_sign[j] = prev_bounds[j] * coeff_sign[j];
                }

                for(int j = 0; j < this->max_coeffs-1; j++){
                    copied_lc[j]                        = this->lower_constraints[i*this->max_coeffs + j];
                    copied_lc[j + this->max_coeffs - 1] = this->lower_constraints[i*this->max_coeffs + j];
                }
                
                // inner product
                this->lower_bounds[i] = inner_product_bundle(2*(this->max_coeffs - 1), copied_lc, coeff_sign, this->party);

            }

            delete[] copied_lc;
            delete[] coeff_sign;


            // restore the fixed-point scale
            ZKgeneralTruncAny(this->party, this->lower_bounds, this->lower_bounds, this->output_size, FXPSCALE);

            for(int i = 0; i < this->output_size; i++){
                this->lower_bounds[i] = this->lower_bounds[i] + this->lower_constraints[(i+1)*this->max_coeffs - 1];    // adding the constant bias term
            }

            delete[] prev_bounds;

        } else {
            cleartext_compute_lower_bounds();
        }

    }

    void compute_upper_bounds(){
        T* prev_lbs = ((Layer<T>*) this->prev_layer)->lower_bounds;
        T* prev_ubs = ((Layer<T>*) this->prev_layer)->upper_bounds;

        if constexpr (std::is_same<IntFp, T>::value && SECURE){
            IntFp* coeff_sign = new IntFp[2 * (this->max_coeffs - 1)]; 
            T* copied_uc = new T[2*(this->max_coeffs - 1)];
            T* prev_bounds = new T[2*(this->max_coeffs - 1)];       // lb_1 lb_2 ... lb_m   ub_1 ub_2 ... ub_m  
            
            for(int i = 0; i < this->output_size; i++){

                for(int j = 0; j < this->max_coeffs - 1; j++){
                    int j_th_predecessor = this->pred_neuron_ids[i * this->max_coeffs + j];

                    prev_bounds[j] = prev_ubs[j_th_predecessor];
                    prev_bounds[j + this->max_coeffs - 1] = prev_lbs[j_th_predecessor];
                }

                auto start = clock_start();


                // cerr << "N" << i << "\n";
                ZKcmpPositive(this->party, this->upper_constraints + i*this->max_coeffs, ZERO_COMP_CONSTANT, coeff_sign, this->max_coeffs - 1);
                
                for(int j = 0; j < this->max_coeffs - 1; j++){
                    coeff_sign[j + this->max_coeffs - 1] = FIELD_ONE + coeff_sign[j].negate();
                }

                // first select which bound to multiply based on sign
                for(int j = 0; j < 2*(this->max_coeffs - 1); j++){
                    coeff_sign[j] = prev_bounds[j] * coeff_sign[j];
                }

                for(int j = 0; j < this->max_coeffs-1; j++){
                    copied_uc[j]                        = this->upper_constraints[i*this->max_coeffs + j];
                    copied_uc[j + this->max_coeffs - 1] = this->upper_constraints[i*this->max_coeffs + j];
                }
                
                // inner product
                this->upper_bounds[i] = inner_product_bundle(2*(this->max_coeffs - 1), copied_uc, coeff_sign, this->party);

            }

            delete[] copied_uc;
            delete[] coeff_sign;


            // restore the fixed-point scale
            ZKgeneralTruncAny(this->party, this->upper_bounds, this->upper_bounds, this->output_size, FXPSCALE);

            for(int i = 0; i < this->output_size; i++){
                this->upper_bounds[i] = this->upper_bounds[i] + this->upper_constraints[(i+1)*this->max_coeffs - 1];    // adding the constant bias term
            }

            delete[] prev_bounds;
            
        } else {
            cleartext_compute_upper_bounds();
        }

    }


    void compute_lower_constraints(){
        
        for(int z = 0; z < this->out_channels; z++){
            T* mask = this->kernel->get_flattened_weights(z);

            for(int y = 0; y < this->out_h; y++){
                for(int x = 0; x < this->out_w; x++){
                    
                    
                    int preds = 0;
                    for(; preds < this->max_coeffs-1; preds++){
                        this->lower_constraints[(
                            z * this->out_h * this->out_w +
                            y * this->out_w +
                            x
                        ) * this->max_coeffs
                        + preds
                        ] = mask[preds];
                    }
                    
                    // set bias
                    this->lower_constraints[(
                        z * this->out_h * this->out_w +
                        y * this->out_w +
                        x
                    ) * this->max_coeffs
                        + preds
                    ] = this->kernel->filter_matrix[
                        this->kernel->num_weights() + z
                    ];
                }
            }
        }
        
    }

    void compute_upper_constraints(){
        for(int z = 0; z < this->out_channels; z++){
            T* mask = this->kernel->get_flattened_weights(z);

            for(int y = 0; y < this->out_h; y++){
                for(int x = 0; x < this->out_w; x++){
                    
                    
                    int preds = 0;
                    for(; preds < this->max_coeffs-1; preds++){
                        this->upper_constraints[(
                            z * this->out_h * this->out_w +
                            y * this->out_w +
                            x
                        ) * this->max_coeffs
                        + preds
                        ] = mask[preds];
                    }
                    
                    // set bias
                    this->upper_constraints[(
                        z * this->out_h * this->out_w +
                        y * this->out_w +
                        x
                    ) * this->max_coeffs
                        + preds
                    ] = this->kernel->filter_matrix[
                        this->kernel->num_weights() + z
                    ];
                }
            }
        }
        
    }


    void backsubstitute(Layer<T>* input_layer){
        
        if(DO_DP_BS){
            // cout << "LAYER " << this->layer_num << "\n";
            for(int i = 0; i < this->output_size * this->max_coeffs; i++){
                this->backsubstituted_lower_constraints[i] = (this->lower_constraints[i]);
                this->backsubstituted_upper_constraints[i] = (this->upper_constraints[i]);
            }
        
            Layer<T>* prev_layer = this->prev_layer;
            while(prev_layer != NULL){
                update_lower_bounds_using_prev_layers(this, prev_layer);     
                prev_layer = prev_layer->prev_layer;
            }
            this->max_coeffs = this->input_size + 1;

            prev_layer = this->prev_layer;
            while(prev_layer != NULL){
                update_upper_bounds_using_prev_layers(this, prev_layer);        
                prev_layer = prev_layer->prev_layer;
            }
            this->max_coeffs = this->input_size + 1;

        } else {
            for(int i = 0; i < this->output_size * this->max_coeffs; i++){
                this->backsubstituted_lower_constraints[i] = T(this->lower_constraints[i]);
                this->backsubstituted_upper_constraints[i] = T(this->upper_constraints[i]);
            }
        
            Layer<T>* prev_layer = this->prev_layer;
            while(prev_layer != NULL){
                update_lower_bounds_using_prev_layers(this, prev_layer);
                if(prev_layer->type == AFFINE){
                    prev_layer = input_layer;
                } else {
                    prev_layer = prev_layer->prev_layer;
                }
            }
            this->max_coeffs = this->input_size + 1;

            prev_layer = this->prev_layer;
            while(prev_layer != NULL){
                update_upper_bounds_using_prev_layers(this, prev_layer);        
                if(prev_layer->type == AFFINE){
                    prev_layer = input_layer;
                } else {
                    prev_layer = prev_layer->prev_layer;
                }
            }
            this->max_coeffs = this->input_size + 1;
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

    void print_predecessor_ids(){
        cout << "Predecessor Neurons:\n";
        int neurons = 0;
        if constexpr (std::is_same<float, T>::value){
            
            for(int y = 0; y < this->out_h; y++){
                for(int x = 0; x < this->out_w; x++){  
                    for(int z = 0; z < this->out_channels; z++){
                        int o = z* this->out_h * this->out_w + y*this->out_w + x;
                        cout << "neuron " << neurons++ << ": ";
                        for(int i = 0; i < this->kernel_h; i++){
                            for(int j = 0; j < this->kernel_w; j++){
                                
                                for(int k = 0; k < this->in_channels; k++){
                                    cout << this->lower_constraints[
                                        o * this->max_coeffs +
                                        k * this->kernel_h * this->kernel_w +
                                        i * this->kernel_w +
                                        j
                                    ] << " ";
                                }
                            }
                        }
                        cout << "\n";
                    }
                }
            }
        }
    }

    void describe(bool print_parameters = true, bool print_expressions = false){
        cout << "Type: " << get_layer_type(this->type) << "\n";


        if (0){
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
        } else if(0) {
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

        if(!ONLY_INFERENCE){
            cout << "Lower Bounds:\n";
            for(int y = 0; y < this->out_h; y++){
                for(int x = 0; x < this->out_w; x++){  
                    for(int z = 0; z < this->out_channels; z++){
                        int o = z* this->out_h * this->out_w + y*this->out_w + x;

                        T el = this->lower_bounds[o];
                        
                        if constexpr (std::is_same<T, IntFp>::value){
                            cout << format_EMP_IntFp(el, 1) << " ";
                        } else if constexpr (std::is_same<T, float>::value) {
                            cout << el << " ";
                        }
                    }
                }
            }
            cout << "\n\n";

            cout << "Upper Bounds:\n";
            for(int y = 0; y < this->out_h; y++){
                for(int x = 0; x < this->out_w; x++){  
                    for(int z = 0; z < this->out_channels; z++){
                        int o = z* this->out_h * this->out_w + y*this->out_w + x;

                        T el = this->upper_bounds[o];
                        
                        if constexpr (std::is_same<T, IntFp>::value){
                            cout << format_EMP_IntFp(el, 1) << " ";
                        } else if constexpr (std::is_same<T, float>::value) {
                            cout << el << " ";
                        }
                    }
                }
            }
            cout << "\n\n";
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
        T* prev_lbs = ((Layer<T>*) this->prev_layer)->lower_bounds;
        T* prev_ubs = ((Layer<T>*) this->prev_layer)->upper_bounds;

        T* prev_bounds = new T[this->max_coeffs];

        for(int i = 0; i < this->output_size; i++){

            for(int j = 0; j < this->max_coeffs-1; j++){

                int j_th_predecessor = this->pred_neuron_ids[i * this->max_coeffs + j];

                if(greater_eq_zero<T>(this->lower_constraints[i*this->max_coeffs + j], false)){
                    prev_bounds[j] = prev_lbs[j_th_predecessor];
                } else {
                    prev_bounds[j] = prev_ubs[j_th_predecessor];
                }
            }
            prev_bounds[this->max_coeffs-1] = constant<T>(1);

            this->lower_bounds[i] = inner_product_emp(this->max_coeffs, this->lower_constraints + i*(this->max_coeffs),  prev_bounds);
        
        }

        if(std::is_same<IntFp, T>::value){
            normalize(this->output_size, (IntFp*)this->lower_bounds, (IntFp*)this->lower_bounds);
        }

        delete[] prev_bounds;
    }

    void cleartext_compute_upper_bounds(){
        T* prev_lbs = ((Layer<T>*) this->prev_layer)->lower_bounds;
        T* prev_ubs = ((Layer<T>*) this->prev_layer)->upper_bounds;

        T* prev_bounds = new T[this->max_coeffs];

        for(int i = 0; i < this->output_size; i++){

            for(int j = 0; j < this->max_coeffs-1; j++){

                int j_th_predecessor = this->pred_neuron_ids[i * this->max_coeffs + j];

                if(greater_eq_zero<T>(this->upper_constraints[i*this->max_coeffs + j], false)){
                    prev_bounds[j] = prev_ubs[j_th_predecessor];
                } else {
                    prev_bounds[j] = prev_lbs[j_th_predecessor];
                }
            }
            prev_bounds[this->max_coeffs-1] = constant<T>(1);

            this->upper_bounds[i] = inner_product_emp(this->max_coeffs, this->upper_constraints + i*(this->max_coeffs),  prev_bounds);
        
        }

        if(std::is_same<IntFp, T>::value){
            normalize(this->output_size, (IntFp*)this->upper_bounds, (IntFp*)this->upper_bounds);
        }

        delete[] prev_bounds;
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