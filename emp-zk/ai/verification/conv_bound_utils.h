#ifndef __CONV_BOUND_UTILS_H__
#define __CONV_BOUND_UTILS_H__

#pragma once

#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "emp-zk/ai/verification/verification.h"
#include "emp-zk/ai/utils.h"

#include <fstream>

using namespace std;
using namespace emp;

template <typename T>
void update_conv_lower_bounds_using_prev_layers(Conv2D<T>* current_layer, Layer<T>* prev_layer){
    assert(current_layer->type == LAYER_TYPE::CONV2D);
    auto start = clock_start();
    double tt;

    if constexpr (std::is_same<IntFp, T>::value && SECURE){
        T* prev_lbs = prev_layer->lower_bounds;
        T* prev_ubs = prev_layer->upper_bounds;

        // update lower bounds
        T* prev_bounds = new T[2*(current_layer->max_coeffs - 1)];       // lb_1 lb_2 ... lb_m   ub_1 ub_2 ... ub_m  
        for(int j = 0; j < current_layer->max_coeffs - 1; j++){
            prev_bounds[j]                                 = prev_lbs[j];
            prev_bounds[j + current_layer->max_coeffs - 1] = prev_ubs[j];
        }

        IntFp* coeff_sign = new IntFp[2 * (current_layer->max_coeffs - 1)];   
        T* copied_lc = new T[2*(current_layer->max_coeffs - 1)];

        for(int i = 0; i < current_layer->output_size; i++){
        
            ZKcmpPositive(current_layer->party, current_layer->backsubstituted_lower_constraints + i*current_layer->max_coeffs, ZERO_COMP_CONSTANT, coeff_sign, current_layer->max_coeffs - 1);
            for(int j = 0; j < current_layer->max_coeffs - 1; j++){
                coeff_sign[j + current_layer->max_coeffs - 1] = FIELD_ONE + coeff_sign[j].negate();
            }

            // first select which bound to multiply based on sign
            for(int j = 0; j < 2*(current_layer->max_coeffs - 1); j++){
                coeff_sign[j] = prev_bounds[j] * coeff_sign[j];
            }

            for(int j = 0; j < current_layer->max_coeffs-1; j++){
                copied_lc[j]                                 = current_layer->backsubstituted_lower_constraints[i*current_layer->max_coeffs + j];
                copied_lc[j + current_layer->max_coeffs - 1] = current_layer->backsubstituted_lower_constraints[i*current_layer->max_coeffs + j];
            }
            
            // inner product
            current_layer->lower_bounds[i] = inner_product_bundle(2*(current_layer->max_coeffs - 1), copied_lc, coeff_sign, current_layer->party);
        }

        delete[] coeff_sign;
        delete[] copied_lc;

        // restore the fixed-point scale
        ZKgeneralTruncAny(current_layer->party, current_layer->lower_bounds, current_layer->lower_bounds, current_layer->output_size, FXPSCALE);

        for(int i = 0; i < current_layer->output_size; i++){
            current_layer->lower_bounds[i] = current_layer->lower_bounds[i] + current_layer->backsubstituted_lower_constraints[(i+1)*current_layer->max_coeffs - 1];    // adding the constant bias term
        }

        delete[] prev_bounds;


        /* Update constraints now */
        start = clock_start();
        // update lower constraints
        if (prev_layer->type == INPUT){
            ;
        } else if(prev_layer->type == RELU){
            update_lower_constraints_with_activation(current_layer, prev_layer);
        } else {
            if(DO_DP_BS){
                update_lower_constraints_with_affine(current_layer, prev_layer);        
            } else {
                update_lower_constraints_with_bsed_affine(current_layer, prev_layer);
            }
        }
        tt = time_from(start);

        if(prev_layer->type == RELU){
            current_layer->time_for_bs_using_act += tt;
        } else {
            current_layer->time_for_bs_using_affine += tt;
        }

    } else {
        cleartext_update_lower_bounds_using_prev_layers(current_layer, prev_layer);
    }

}


template <typename T>
void cleartext_conv_update_lower_bounds_using_prev_layers(Conv2D<T>* current_layer, Layer<T>* prev_layer){
    assert(current_layer->type == LAYER_TYPE::CONV2D);
    
    auto start = clock_start();

    T* prev_bounds = new T[current_layer->max_coeffs];  

    // update lower bounds
    for(int i = 0; i < current_layer->output_size; i++){
        T* current_lower_constraints = current_layer->backsubstituted_lower_constraints + (i*current_layer->max_coeffs);

        for(int j = 0; j < current_layer->max_coeffs-1; j++){

            int j_th_predecessor = current_layer->pred_neuron_ids[i * current_layer->max_coeffs + j];

            if(greater_eq_zero<T>(current_lower_constraints[j], false)){
                prev_bounds[j] = prev_layer->lower_bounds[j_th_predecessor];
            } else {
                prev_bounds[j] = prev_layer->upper_bounds[j_th_predecessor];
            }
        }
        prev_bounds[current_layer->max_coeffs-1] = constant<T>(1);

        current_layer->lower_bounds[i] = inner_product_emp(current_layer->max_coeffs, current_lower_constraints,  prev_bounds);
    
    }

    if(std::is_same<IntFp, T>::value){
        normalize(current_layer->output_size, current_layer->lower_bounds, current_layer->lower_bounds);
    }

    double tt = time_from(start);
    current_layer->time_for_bs_in_bounds += tt;

    start = clock_start();
    // update lower constraints
    if (prev_layer->type == INPUT){
        ;
    } else if(prev_layer->type == RELU){
        update_conv_lower_constraints_with_activation(current_layer, prev_layer);
    } else {
        if(DO_DP_BS){
            update_conv_lower_constraints_with_nonactivation(current_layer, prev_layer);        
        } else {
            // update_lower_constraints_with_bsed_affine(current_layer, prev_layer);
        }
    }
    tt = time_from(start);

    if(prev_layer->type == RELU){
        current_layer->time_for_bs_using_act += tt;
    } else {
        current_layer->time_for_bs_using_affine += tt;
    }
}


template <typename T>
void update_conv_lower_constraints_with_activation(Conv2D<T>* current_layer, Layer<T>* prev_layer){
    T* new_backsubstituted_lower_constraints = new T[current_layer->output_size * current_layer->max_coeffs];
    
    for(int i = 0; i < current_layer->output_size; i++){
        
        T* current_lower_constraints = current_layer->backsubstituted_lower_constraints + (i*current_layer->max_coeffs);
        new_backsubstituted_lower_constraints[(i+1)*(current_layer->max_coeffs) - 1] = current_lower_constraints[current_layer->max_coeffs-1]*constant<T>(1);


        int constant_term_pos = (i+1)*(current_layer->max_coeffs) - 1;
        for(int j = 0; j < current_layer->max_coeffs - 1; j++){

            int j_th_pred_neuron_id = current_layer->pred_neuron_ids[i * current_layer->max_coeffs + j];

            if(greater_eq_zero<T>(current_lower_constraints[j], false)){
                // non-constant term
                new_backsubstituted_lower_constraints[i*current_layer->max_coeffs + j] = current_lower_constraints[j] * 
                                                                                         prev_layer->lower_constraints[j_th_pred_neuron_id * 2 + 0];

                // constant term
                new_backsubstituted_lower_constraints[constant_term_pos] = new_backsubstituted_lower_constraints[constant_term_pos] +
                                                                           current_lower_constraints[j] * prev_layer->lower_constraints[j_th_pred_neuron_id * 2 + 1];
            
            } else {
                // non-constant term
                new_backsubstituted_lower_constraints[i*current_layer->max_coeffs + j] = current_lower_constraints[j] * 
                                                                                         prev_layer->upper_constraints[j_th_pred_neuron_id * 2 + 0];

                // constant term
                new_backsubstituted_lower_constraints[constant_term_pos] = new_backsubstituted_lower_constraints[constant_term_pos] +
                                                                           current_lower_constraints[j] * prev_layer->upper_constraints[j_th_pred_neuron_id * 2 + 1];
            }
        }

        if constexpr (std::is_same<IntFp, T>::value){
            normalize(current_layer->max_coeffs, new_backsubstituted_lower_constraints + i*current_layer->max_coeffs, new_backsubstituted_lower_constraints + i*current_layer->max_coeffs);
        }
    }

    
    delete[] current_layer->backsubstituted_lower_constraints;
    current_layer->backsubstituted_lower_constraints = new_backsubstituted_lower_constraints;

}


template <typename T>
void cleartext_update_conv_lower_constraints_with_nonactivation(Conv2D<T>* current_layer, Layer<T>* prev_layer){

    T* new_backsubstituted_lower_constraints = new T[current_layer->output_size * (prev_layer->input_size + 1)];
    for(int i = 0; i < current_layer->output_size * (prev_layer->input_size + 1); i++){
        new_backsubstituted_lower_constraints[i] = constant<T>(0);
    }

    int* new_pred_neuron_ids = new int[current_layer->output_size * (prev_layer->input_size + 1)];

    for(int i = 0; i < current_layer->output_size; i++){
        T* current_lower_constraints = current_layer->backsubstituted_lower_constraints + (i*current_layer->max_coeffs);
       
        for(int j = 0; j < current_layer->max_coeffs - 1; j++){

            int j_th_pred_neuron_id = current_layer->pred_neuron_ids[i * current_layer->max_coeffs + j];
            int* pred_neurons_preds = prev_layer->
            
            for(int k = 0; k < prev_layer->input_size; k++){
                
            }
        }
        
        
    }

    current_layer->max_coeffs = prev_layer->max_coeffs; // check

    delete[] current_layer->backsubstituted_lower_constraints;
    current_layer->backsubstituted_lower_constraints = new_backsubstituted_lower_constraints;
}
