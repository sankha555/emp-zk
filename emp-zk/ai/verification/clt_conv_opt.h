#ifndef __CLT_CONV_OPT_H__
#define __CLT_CONV_OPT_H__

#pragma once

#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "emp-zk/ai/verification/verification.h"
#include "emp-zk/ai/utils.h"

#include <map>
#include <fstream>

using namespace std;
using namespace emp;


/* ===================== CLEARTEXT SEMANTICS ===================== */

/* ===================== LOWER CONSTRAINTS ===================== */

template <typename T>
void cleartext_update_conv_lower_bounds_using_prev_layers(Conv2D<T>* current_layer, Layer<T>* prev_layer){

    
    for(int i = 0; i < current_layer->output_size; i++){

        assert(
            (*current_layer->backsubstituted_conv_lower_constraints)[i].size() ==
            (*current_layer->predecessors)[i].size() + 1
        );

        T* prev_layer_bounds_to_mult = new T[(*current_layer->predecessors)[i].size() + 1];   
        
        int* pred = (*current_layer->predecessors)[i].data();
        int* end = pred + (*current_layer->predecessors)[i].size();

        int j = 0;
        for(; pred != end; pred++){
            int j_th_predecessor = *pred;
            T j_th_lc = (*current_layer->backsubstituted_conv_lower_constraints)[i][j];

            if(j_th_predecessor == -1){
                prev_layer_bounds_to_mult[j] = constant<T>(0);
            } else {
                if(greater_eq_zero<T>(j_th_lc, false)){
                    prev_layer_bounds_to_mult[j] = prev_layer->lower_bounds[j_th_predecessor];
                } else {
                    prev_layer_bounds_to_mult[j] = prev_layer->upper_bounds[j_th_predecessor];
                }
            }
            
            j++;
        }
        prev_layer_bounds_to_mult[(*current_layer->predecessors)[i].size()] = constant<T>(1);

        current_layer->lower_bounds[i] = inner_product_emp(
            (*current_layer->predecessors)[i].size() + 1,
            (*current_layer->backsubstituted_conv_lower_constraints)[i].data(),
            prev_layer_bounds_to_mult
        );
    }
    
    if constexpr (std::is_same<IntFp, T>::value){
        normalize(current_layer->output_size, current_layer->lower_bounds, current_layer->lower_bounds);
    }


    // update lower constraints
    if (prev_layer->type == INPUT){
        ;
    } else if(prev_layer->type == RELU){
        cleartext_update_conv_lower_constraints_with_activation(current_layer, prev_layer);
    } else if(prev_layer->type == AFFINE) {
        cleartext_update_conv_lower_constraints_with_affine(current_layer, prev_layer);  
    } else if(prev_layer->type == CONV2D) {
        cleartext_update_conv_lower_constraints_with_conv(current_layer, (Conv2D<T>*) prev_layer);
    }


}

template <typename T>
void cleartext_update_conv_lower_constraints_with_affine(Conv2D<T>* current_layer, Layer<T>* prev_layer){

    T* new_backsubstituted_lower_constraints = new T[current_layer->output_size * (prev_layer->input_size + 1)];
    for(int i = 0; i < current_layer->output_size * (prev_layer->input_size + 1); i++){
        new_backsubstituted_lower_constraints[i]= constant<T>(0);
    }


    int num_preds_of_pred = prev_layer->max_coeffs - 1;


    for(int i = 0; i < current_layer->output_size; i++){
        T* current_lower_constraints = current_layer->backsubstituted_lower_constraints + (i*current_layer->max_coeffs);
        T* prev_coeffs_to_mult = new T[prev_layer->output_size];

        for(int k = 0; k < prev_layer->input_size + 1; k++){        // including constant term
    
            for(int j = 0; j < prev_layer->output_size; j++){
                if(greater_eq_zero<T>(current_lower_constraints[j], false)){
                    prev_coeffs_to_mult[j] = prev_layer->lower_constraints[j*prev_layer->max_coeffs + k];
                } else {
                    prev_coeffs_to_mult[j] = prev_layer->upper_constraints[j*prev_layer->max_coeffs + k];
                }    
            }

            new_backsubstituted_lower_constraints[i*prev_layer->max_coeffs + k] = inner_product_emp(prev_layer->output_size, current_lower_constraints, prev_coeffs_to_mult);
        }

        if constexpr (std::is_same<IntFp, T>::value){
            normalize(prev_layer->max_coeffs, new_backsubstituted_lower_constraints + i*prev_layer->max_coeffs, new_backsubstituted_lower_constraints + i*prev_layer->max_coeffs);
        }

        // adding constant term to constant product
        new_backsubstituted_lower_constraints[(i + 1)*prev_layer->max_coeffs - 1] = new_backsubstituted_lower_constraints[(i + 1)*prev_layer->max_coeffs - 1] 
                                                                                    + current_lower_constraints[current_layer->max_coeffs - 1];
    }

    current_layer->max_coeffs = prev_layer->input_size + 1; // check

    delete[] current_layer->backsubstituted_lower_constraints;
    current_layer->backsubstituted_lower_constraints = new_backsubstituted_lower_constraints;
}


template <typename T>
void cleartext_update_conv_lower_constraints_with_conv(Conv2D<T>* current_layer, Conv2D<T>* prev_layer){
    
    vector<vector<T>>* new_backsubstituted_conv_lower_constraints = new vector<vector<T>>(current_layer->output_size);
    vector<vector<int>>* new_predecessors = new vector<vector<int>>(current_layer->output_size);
    
    for(int i = 0; i < current_layer->output_size; i++){

        assert(
            (*current_layer->backsubstituted_conv_lower_constraints)[i].size() ==
            (*current_layer->predecessors)[i].size() + 1
        );


        int* pred = (*current_layer->predecessors)[i].data();
        int* end = pred + (*current_layer->predecessors)[i].size();

        map<int, int> seen_predecessors;
        T new_const_term = (*current_layer->backsubstituted_conv_lower_constraints)[i][(*current_layer->predecessors)[i].size()] * constant<T>(1);

        int j = 0;
        for(; pred != end; pred++){

            T j_th_lc = (*current_layer->backsubstituted_conv_lower_constraints)[i][j];

            int j_th_predecessor = *pred;
            if(j_th_predecessor == -1){

                (*new_backsubstituted_conv_lower_constraints)[i].push_back(constant<T>(0));
                (*new_predecessors)[i].push_back(-1);

            } else {

                int* prev_pred = (*prev_layer->predecessors)[j_th_predecessor].data();
                int* prev_end = prev_pred + (*prev_layer->predecessors)[j_th_predecessor].size();


                T non_const_prev_constraint;
                T const_prev_constraint;


                int k = 0;
                for(; prev_pred != prev_end; prev_pred++){
                    int kth_pred_of_pred = *prev_pred;

                    if(kth_pred_of_pred == -1){

                        (*new_backsubstituted_conv_lower_constraints)[i].push_back(constant<T>(0));
                        (*new_predecessors)[i].push_back(-1);

                    } else {

                        if(greater_eq_zero<T>(j_th_lc, false)){
                            non_const_prev_constraint = prev_layer->lower_constraints[j_th_predecessor * prev_layer->max_coeffs + k];
                            const_prev_constraint = prev_layer->lower_constraints[(j_th_predecessor + 1) * prev_layer->max_coeffs - 1];
                        } else {
                            non_const_prev_constraint = prev_layer->upper_constraints[j_th_predecessor * prev_layer->max_coeffs + k];
                            const_prev_constraint = prev_layer->upper_constraints[(j_th_predecessor + 1) * prev_layer->max_coeffs - 1];
                        }


                        int pred_index;     
                        if(seen_predecessors.count(kth_pred_of_pred)){

                            pred_index = seen_predecessors[kth_pred_of_pred];

                            (*new_backsubstituted_conv_lower_constraints)[i][pred_index] = (*new_backsubstituted_conv_lower_constraints)[i][pred_index] +
                                                                                            j_th_lc * non_const_prev_constraint;

                        } else {

                            seen_predecessors[kth_pred_of_pred] = (*new_predecessors)[i].size();
                            pred_index = (*new_predecessors)[i].size();
                            (*new_backsubstituted_conv_lower_constraints)[i].push_back(j_th_lc * non_const_prev_constraint);

                            (*new_predecessors)[i].push_back(kth_pred_of_pred);
                        }
                    }
                    
                    k++;
                }            

                new_const_term = new_const_term + j_th_lc * const_prev_constraint;
            }

            j++;
        }

        (*new_backsubstituted_conv_lower_constraints)[i].push_back(new_const_term);
        assert((*new_backsubstituted_conv_lower_constraints)[i].size() == (*new_predecessors)[i].size() + 1);

        if constexpr (std::is_same<IntFp, T>::value){
            normalize(
                (*new_backsubstituted_conv_lower_constraints)[i].size(),
                (*new_backsubstituted_conv_lower_constraints)[i].data(),
                (*new_backsubstituted_conv_lower_constraints)[i].data()
            );
        }
    }

    delete current_layer->backsubstituted_conv_lower_constraints;
    current_layer->backsubstituted_conv_lower_constraints = new_backsubstituted_conv_lower_constraints;

    delete current_layer->predecessors;
    current_layer->predecessors = new_predecessors;
}


template <typename T>
void cleartext_update_conv_lower_constraints_with_activation(Conv2D<T>* current_layer, Layer<T>* prev_layer){

    vector<vector<T>>* new_backsubstituted_conv_lower_constraints = new vector<vector<T>>(current_layer->output_size);
    vector<vector<int>>* new_predecessors = new vector<vector<int>>(current_layer->output_size);

    for(int i = 0; i < current_layer->output_size; i++){

        assert(
            (*current_layer->backsubstituted_conv_lower_constraints)[i].size() ==
            (*current_layer->predecessors)[i].size() + 1
        );

        int* pred = (*current_layer->predecessors)[i].data();
        int* end = pred + (*current_layer->predecessors)[i].size();

        T new_const_term = (*current_layer->backsubstituted_conv_lower_constraints)[i][(*current_layer->predecessors)[i].size()] * constant<T>(1);

        int j = 0;
        for(; pred != end; pred++){

            T j_th_lc = (*current_layer->backsubstituted_conv_lower_constraints)[i][j];

            int j_th_predecessor = *pred;
            if(j_th_predecessor == -1){

                (*new_predecessors)[i].push_back(-1);
                (*new_backsubstituted_conv_lower_constraints)[i].push_back(constant<T>(0));

            } else {

                T non_const_prev_constraint;
                T const_prev_constraint;
                if(greater_eq_zero<T>(j_th_lc, false)){
                    non_const_prev_constraint = prev_layer->lower_constraints[j_th_predecessor * 2 + 0];
                    const_prev_constraint = prev_layer->lower_constraints[j_th_predecessor * 2 + 1];
                } else {
                    non_const_prev_constraint = prev_layer->upper_constraints[j_th_predecessor * 2 + 0];
                    const_prev_constraint = prev_layer->upper_constraints[j_th_predecessor * 2 + 1];        
                }

                (*new_backsubstituted_conv_lower_constraints)[i].push_back(j_th_lc * non_const_prev_constraint);
                new_const_term = new_const_term + j_th_lc * const_prev_constraint;

                (*new_predecessors)[i].push_back(j_th_predecessor);

            }            

            j++;
        }

        (*new_backsubstituted_conv_lower_constraints)[i].push_back(new_const_term);
        
        assert(
            (*new_backsubstituted_conv_lower_constraints)[i].size() ==
            (*new_predecessors)[i].size() + 1
        );


        if constexpr (std::is_same<IntFp, T>::value){
            normalize(
                (*new_backsubstituted_conv_lower_constraints)[i].size(),
                (*new_backsubstituted_conv_lower_constraints)[i].data(),
                (*new_backsubstituted_conv_lower_constraints)[i].data()
            );
        }            
    }

    delete current_layer->backsubstituted_conv_lower_constraints;
    current_layer->backsubstituted_conv_lower_constraints = new_backsubstituted_conv_lower_constraints;

    delete current_layer->predecessors;
    current_layer->predecessors = new_predecessors;
}


/* ===================== UPPER CONSTRAINTS ===================== */
template <typename T>
void cleartext_update_conv_upper_bounds_using_prev_layers(Conv2D<T>* current_layer, Layer<T>* prev_layer){

    for(int i = 0; i < current_layer->output_size; i++){

        assert(
            (*current_layer->backsubstituted_conv_upper_constraints)[i].size() ==
            (*current_layer->predecessors)[i].size() + 1
        );

        T* prev_layer_bounds_to_mult = new T[(*current_layer->predecessors)[i].size() + 1];   
        
        int* pred = (*current_layer->predecessors)[i].data();
        int* end = pred + (*current_layer->predecessors)[i].size();

        int j = 0;
        for(; pred != end; pred++){
            int j_th_predecessor = *pred;
            T j_th_lc = (*current_layer->backsubstituted_conv_upper_constraints)[i][j];

            if(j_th_predecessor == -1){
                prev_layer_bounds_to_mult[j] = constant<T>(0);
            } else if(greater_eq_zero<T>(j_th_lc, false)){
                prev_layer_bounds_to_mult[j] = prev_layer->upper_bounds[j_th_predecessor];
            } else {
                prev_layer_bounds_to_mult[j] = prev_layer->lower_bounds[j_th_predecessor];
            }
            
            j++;
        }
        prev_layer_bounds_to_mult[(*current_layer->predecessors)[i].size()] = constant<T>(1);

        current_layer->upper_bounds[i] = inner_product_emp(
            (*current_layer->predecessors)[i].size() + 1,
            (*current_layer->backsubstituted_conv_upper_constraints)[i].data(),
            prev_layer_bounds_to_mult
        );
    }
    
    if constexpr (std::is_same<IntFp, T>::value){
        normalize(current_layer->output_size, current_layer->upper_bounds, current_layer->upper_bounds);
    }
    

    // update upper constraints
    if (prev_layer->type == INPUT){
        ;
    } else if(prev_layer->type == RELU){
        cleartext_update_conv_upper_constraints_with_activation(current_layer, prev_layer);
    } else if(prev_layer->type == AFFINE) {
        cleartext_update_conv_upper_constraints_with_affine(current_layer, prev_layer);        
    } else if(prev_layer->type == CONV2D) {
        cleartext_update_conv_upper_constraints_with_conv(current_layer, (Conv2D<T>*) prev_layer);
    }
}

template <typename T>
void cleartext_update_conv_upper_constraints_with_affine(Conv2D<T>* current_layer, Layer<T>* prev_layer){
    T* new_backsubstituted_upper_constraints = new T[current_layer->output_size * prev_layer->max_coeffs];
    for(int i = 0; i < current_layer->output_size * (prev_layer->input_size + 1); i++){
        new_backsubstituted_upper_constraints[i]= constant<T>(0);
    }

    for(int i = 0; i < current_layer->output_size; i++){
        T* current_upper_constraints = current_layer->backsubstituted_upper_constraints + (i*current_layer->max_coeffs);
        T* prev_coeffs_to_mult = new T[prev_layer->output_size];
        
        for(int k = 0; k < prev_layer->max_coeffs; k++){        // including constant term
    
            for(int j = 0; j < prev_layer->output_size; j++){
                if(greater_eq_zero<T>(current_upper_constraints[j], false)){
                    prev_coeffs_to_mult[j] = prev_layer->upper_constraints[j*prev_layer->max_coeffs + k];
                } else {
                    prev_coeffs_to_mult[j] = prev_layer->lower_constraints[j*prev_layer->max_coeffs + k];
                }    
            }

            new_backsubstituted_upper_constraints[i*prev_layer->max_coeffs + k] = inner_product_emp(prev_layer->output_size, current_upper_constraints, prev_coeffs_to_mult);
        }

        if constexpr (std::is_same<IntFp, T>::value){
            normalize(prev_layer->max_coeffs, new_backsubstituted_upper_constraints + i*prev_layer->max_coeffs, new_backsubstituted_upper_constraints + i*prev_layer->max_coeffs);
        }

        // adding constant term to constant product
        new_backsubstituted_upper_constraints[(i + 1)*prev_layer->max_coeffs - 1] = new_backsubstituted_upper_constraints[(i + 1)*prev_layer->max_coeffs - 1] 
                                                                                    + current_upper_constraints[current_layer->max_coeffs - 1];
    }

    current_layer->max_coeffs = prev_layer->max_coeffs; // check

    delete[] current_layer->backsubstituted_upper_constraints;
    current_layer->backsubstituted_upper_constraints = new_backsubstituted_upper_constraints;
}


template <typename T>
void cleartext_update_conv_upper_constraints_with_conv(Conv2D<T>* current_layer, Conv2D<T>* prev_layer){
    
    vector<vector<T>>* new_backsubstituted_conv_upper_constraints = new vector<vector<T>>(current_layer->output_size);
    vector<vector<int>>* new_predecessors = new vector<vector<int>>(current_layer->output_size);
    
    for(int i = 0; i < current_layer->output_size; i++){

        assert(
            (*current_layer->backsubstituted_conv_upper_constraints)[i].size() ==
            (*current_layer->predecessors)[i].size() + 1
        );


        int* pred = (*current_layer->predecessors)[i].data();
        int* end = pred + (*current_layer->predecessors)[i].size();

        map<int, int> seen_predecessors;
        T new_const_term = (*current_layer->backsubstituted_conv_upper_constraints)[i][(*current_layer->predecessors)[i].size()] * constant<T>(1);

        int j = 0;
        for(; pred != end; pred++){

            T j_th_lc = (*current_layer->backsubstituted_conv_upper_constraints)[i][j];

            int j_th_predecessor = *pred;
            if(j_th_predecessor == -1){

                (*new_backsubstituted_conv_upper_constraints)[i].push_back(constant<T>(0));
                (*new_predecessors)[i].push_back(-1);

            } else {

                int* prev_pred = (*prev_layer->predecessors)[j_th_predecessor].data();
                int* prev_end = prev_pred + (*prev_layer->predecessors)[j_th_predecessor].size();


                T non_const_prev_constraint;
                T const_prev_constraint;


                int k = 0;
                for(; prev_pred != prev_end; prev_pred++){
                    int kth_pred_of_pred = *prev_pred;
                    if(kth_pred_of_pred == -1){

                        (*new_backsubstituted_conv_upper_constraints)[i].push_back(constant<T>(0));
                        (*new_predecessors)[i].push_back(-1);

                    } else {

                        if(greater_eq_zero<T>(j_th_lc, false)){
                            non_const_prev_constraint = prev_layer->upper_constraints[j_th_predecessor * prev_layer->max_coeffs + k];
                            const_prev_constraint = prev_layer->upper_constraints[(j_th_predecessor + 1) * prev_layer->max_coeffs - 1];
                        } else {
                            non_const_prev_constraint = prev_layer->lower_constraints[j_th_predecessor * prev_layer->max_coeffs + k];
                            const_prev_constraint = prev_layer->lower_constraints[(j_th_predecessor + 1) * prev_layer->max_coeffs - 1];
                        }


                        int pred_index;     
                        if(seen_predecessors.count(kth_pred_of_pred)){

                            pred_index = seen_predecessors[kth_pred_of_pred];

                            (*new_backsubstituted_conv_upper_constraints)[i][pred_index] = (*new_backsubstituted_conv_upper_constraints)[i][pred_index] +
                                                                                            j_th_lc * non_const_prev_constraint;

                        } else {

                            seen_predecessors[kth_pred_of_pred] = (*new_predecessors)[i].size();
                            pred_index = (*new_predecessors)[i].size();
                            (*new_backsubstituted_conv_upper_constraints)[i].push_back(j_th_lc * non_const_prev_constraint);

                            (*new_predecessors)[i].push_back(kth_pred_of_pred);
                        }
                    }
                    
                    k++;
                }            

                new_const_term = new_const_term + j_th_lc * const_prev_constraint;
            }

            j++;
        }

        (*new_backsubstituted_conv_upper_constraints)[i].push_back(new_const_term);
        assert((*new_backsubstituted_conv_upper_constraints)[i].size() == (*new_predecessors)[i].size() + 1);

        if constexpr (std::is_same<IntFp, T>::value){
            normalize(
                (*new_backsubstituted_conv_upper_constraints)[i].size(),
                (*new_backsubstituted_conv_upper_constraints)[i].data(),
                (*new_backsubstituted_conv_upper_constraints)[i].data()
            );
        }
    }

    delete current_layer->backsubstituted_conv_upper_constraints;
    current_layer->backsubstituted_conv_upper_constraints = new_backsubstituted_conv_upper_constraints;

    delete current_layer->predecessors;
    current_layer->predecessors = new_predecessors;
}


template <typename T>
void cleartext_update_conv_upper_constraints_with_activation(Conv2D<T>* current_layer, Layer<T>* prev_layer){
    
    vector<vector<T>>* new_backsubstituted_conv_upper_constraints = new vector<vector<T>>(current_layer->output_size);
    vector<vector<int>>* new_predecessors = new vector<vector<int>>(current_layer->output_size);

    for(int i = 0; i < current_layer->output_size; i++){

        assert(
            (*current_layer->backsubstituted_conv_upper_constraints)[i].size() ==
            (*current_layer->predecessors)[i].size() + 1
        );

        int* pred = (*current_layer->predecessors)[i].data();
        int* end = pred + (*current_layer->predecessors)[i].size();

        T new_const_term = (*current_layer->backsubstituted_conv_upper_constraints)[i][(*current_layer->predecessors)[i].size()] * constant<T>(1);

        int j = 0;
        for(; pred != end; pred++){

            T j_th_lc = (*current_layer->backsubstituted_conv_upper_constraints)[i][j];

            int j_th_predecessor = *pred;

            if(j_th_predecessor == -1){

                (*new_predecessors)[i].push_back(-1);
                (*new_backsubstituted_conv_upper_constraints)[i].push_back(constant<T>(0));

            } else {

                T non_const_prev_constraint;
                T const_prev_constraint;
                if(greater_eq_zero<T>(j_th_lc, false)){
                    non_const_prev_constraint = prev_layer->upper_constraints[j_th_predecessor * 2 + 0];
                    const_prev_constraint = prev_layer->upper_constraints[j_th_predecessor * 2 + 1];
                } else {
                    non_const_prev_constraint = prev_layer->lower_constraints[j_th_predecessor * 2 + 0];
                    const_prev_constraint = prev_layer->lower_constraints[j_th_predecessor * 2 + 1];        
                }

                (*new_backsubstituted_conv_upper_constraints)[i].push_back(j_th_lc * non_const_prev_constraint);
                new_const_term = new_const_term + j_th_lc * const_prev_constraint;

                (*new_predecessors)[i].push_back(j_th_predecessor);
            }   

            j++;
        }

        (*new_backsubstituted_conv_upper_constraints)[i].push_back(new_const_term);
        
        assert(
            (*new_backsubstituted_conv_upper_constraints)[i].size() ==
            (*new_predecessors)[i].size() + 1
        );


        if constexpr (std::is_same<IntFp, T>::value){
            normalize(
                (*new_backsubstituted_conv_upper_constraints)[i].size(),
                (*new_backsubstituted_conv_upper_constraints)[i].data(),
                (*new_backsubstituted_conv_upper_constraints)[i].data()
            );
        }            
    }

    delete current_layer->backsubstituted_conv_upper_constraints;
    current_layer->backsubstituted_conv_upper_constraints = new_backsubstituted_conv_upper_constraints;

    delete current_layer->predecessors;
    current_layer->predecessors = new_predecessors;
}

#endif