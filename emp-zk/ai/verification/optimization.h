#ifndef __OPTIMIZATION_H__
#define __OPTIMIZATION_H__

#pragma once

#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "emp-zk/ai/verification/verification.h"
#include "emp-zk/ai/utils.h"

#include <fstream>

using namespace std;
using namespace emp;

template <typename T>
long get_total_substitutions(Layer<T>* current_layer){
    long num_substitutions = 0;

    Layer<T>* prev_layer = current_layer->prev_layer;
    int num_predecessors = (current_layer->type == AFFINE ? current_layer->input_size : ((Conv2D<float>*) current_layer)->in_channels * ((Conv2D<float>*) current_layer)->kernel_h * ((Conv2D<float>*) current_layer)->kernel_w);

    while(prev_layer->type != INPUT){
        switch (prev_layer->type){
            case RELU:
                num_substitutions += num_predecessors * 1;
                break;
            
            case AFFINE:
                num_substitutions += num_predecessors * prev_layer->input_size;
                break;

            case CONV2D:
                num_substitutions += num_predecessors * ((Conv2D<float>*) prev_layer)->in_channels * ((Conv2D<float>*) prev_layer)->kernel_h * ((Conv2D<float>*) prev_layer)->kernel_w;
                break;

            default:
                break;
        }

        prev_layer = prev_layer->prev_layer;
    }

    return num_substitutions;
}

template <typename T>
float* get_neuron_scores(Layer<T>* current_layer, int score_function = 1){
    int n = current_layer->output_size;
    int h = current_layer->layer_num;
    long num_substitutions = get_total_substitutions(current_layer);

    float* scores = new float[n];

    for(int i = 0; i < n; i++){
        float delta = current_layer->diff[i];

        if(score_function == 1){
            scores[i] = sqrt(delta) * current_layer->layer_num;
        } else if(score_function == 2){
            scores[i] = (100 * delta) / num_substitutions;
        }
    }

    return scores;
}

template <typename T>
void collect_bound_statistics(Layer<T>* current_layer){
    if(!current_layer->type == AFFINE || !current_layer->type == CONV2D){
        return;
    }

    if(!BS_WAIVER_THRESHOLDS.count(to_string(current_layer->layer_num)) && !BS_WAIVER_THRESHOLDS2.count(to_string(current_layer->layer_num))){
        return;
    }

    for(int i = 0; i < current_layer->output_size; i++){
        current_layer->diff[i] = current_layer->diff[i] - (current_layer->upper_bounds[i] - current_layer->lower_bounds[i]);
    }

    float* neuron_scores = get_neuron_scores(current_layer);
    for(int i = 0; i < current_layer->output_size; i++){
        current_layer->neuron_info->push_back(
            {abs(neuron_scores[i]), {current_layer->layer_num, i}}
        );
    }
    
    delete[] neuron_scores;
}

template <typename T>
void analyse_and_unset_bs_bounds(Layer<T>* current_layer, Layer<T>* input_layer){
    if(!current_layer->type == AFFINE || !current_layer->type == CONV2D){
        return;
    }

    if(!BS_WAIVER_THRESHOLDS.count(to_string(current_layer->layer_num)) && !BS_WAIVER_THRESHOLDS2.count(to_string(current_layer->layer_num))){
        return;
    }


    current_layer->skippable_neurons = new set<int>();
    current_layer->skippable_neurons2 = new set<int>();

    float tau = BS_WAIVER_THRESHOLDS[to_string(current_layer->layer_num)];
    float tau2 = BS_WAIVER_THRESHOLDS2[to_string(current_layer->layer_num)];

    for(int i = 0; i < current_layer->output_size; i++){
        current_layer->lower_diff[i] = (current_layer->lower_bounds[i] - current_layer->lower_diff[i]);
        current_layer->upper_diff[i] = (current_layer->upper_bounds[i] - current_layer->upper_diff[i]);
        current_layer->diff[i] = current_layer->diff[i] - (current_layer->upper_bounds[i] - current_layer->lower_bounds[i]);
    }

    vector<pair<float, int>> neuron_info;
    float* neuron_scores = get_neuron_scores(current_layer);

    for(int i = 0; i < current_layer->output_size; i++){
        neuron_info.push_back(
            {abs(current_layer->diff[i]), i}
        );
    }

    sort(neuron_info.begin(), neuron_info.end());

    int num_nonbs_neurons = ((current_layer->output_size * 1.0) * BS_WAIVER_FRACTION);

    for(int k = 0; k < current_layer->output_size; k++){
        int nid = neuron_info[k].second;
        if(neuron_info[k].first < tau){
            current_layer->lower_bounds[nid] = -current_layer->lower_diff[nid] + current_layer->lower_bounds[nid]; 
            current_layer->upper_bounds[nid] = -current_layer->upper_diff[nid] + current_layer->upper_bounds[nid];

            current_layer->skippable_neurons->insert(nid);
        } else if(neuron_info[k].first < tau2){
            current_layer->skippable_neurons2->insert(nid);
        }
    }

    if(current_layer->layer_num > 2 && current_layer->skippable_neurons2->size() > 0){
        if(current_layer->type == AFFINE){
            backsubstitute_lc_using_prev_layers(current_layer, current_layer->prev_layer, current_layer->prev_layer->prev_layer, input_layer, *current_layer->skippable_neurons2);
            backsubstitute_uc_using_prev_layers(current_layer, current_layer->prev_layer, current_layer->prev_layer->prev_layer, input_layer, *current_layer->skippable_neurons2);           
        } else {
            backsubstitute_conv_lc_using_prev_layers((Conv2D<T>*) current_layer, current_layer->prev_layer, current_layer->prev_layer->prev_layer, input_layer, *current_layer->skippable_neurons2);
            backsubstitute_conv_uc_using_prev_layers((Conv2D<T>*) current_layer, current_layer->prev_layer, current_layer->prev_layer->prev_layer, input_layer, *current_layer->skippable_neurons2); 
        }

    }
}

template <typename T>
void apply_bs_heuristics(Layer<T>* current_layer, Layer<T>* input_layer){
    // reset to bounds before backsubstitution
    for(int nid : *current_layer->skippable_neurons){
        current_layer->lower_bounds[nid] = current_layer->lower_diff[nid]; 
        current_layer->upper_bounds[nid] = current_layer->upper_diff[nid];
    }

    // recompute bounds for the neurons selected for the 2nd heuristic
    if(current_layer->skippable_neurons2->size() > 0){
        backsubstitute_lc_using_prev_layers(current_layer, current_layer->prev_layer, current_layer->prev_layer->prev_layer, input_layer, *current_layer->skippable_neurons2);
        backsubstitute_uc_using_prev_layers(current_layer, current_layer->prev_layer, current_layer->prev_layer->prev_layer, input_layer, *current_layer->skippable_neurons2);
    }
}

template <typename T>
vector<long> get_raw_savings(Layer<T>* current_layer){
    if(!current_layer->type == AFFINE || !current_layer->type == CONV2D){
        return {0, 0};
    }

    long num_substitutions = get_total_substitutions(current_layer);
    int num_predecessors = (current_layer->type == AFFINE ? current_layer->input_size : ((Conv2D<float>*) current_layer)->in_channels * ((Conv2D<float>*) current_layer)->kernel_h * ((Conv2D<float>*) current_layer)->kernel_w);

    // no backsubstitution at all
    long savings1 = current_layer->skippable_neurons->size() * num_substitutions;

    // backsubstitution only till previous layer
    long savings2 = current_layer->skippable_neurons2->size() * (num_substitutions - num_predecessors);

    long total_computations = num_substitutions * current_layer->output_size;

    return {savings1, savings2, total_computations};
}



// the next two functions substitute bounds of the previous AFFINE layer in the (optimized) backsubstituted constraints
template <typename T>
void backsubstitute_lc_using_prev_layers(Layer<T>* current_layer, Layer<T>* prev_activation, Layer<T>* prev_affine, Layer<T>* input_layer, set<int> neuron_list = {}){
    if(neuron_list.size() == 0){
        neuron_list = set<int>();
        for(int i = 0; i < current_layer->output_size; i++){
            neuron_list.insert(i);
        }
    }

    T* new_backsubstituted_lower_constraints = new T[neuron_list.size() * (prev_affine->output_size + 1) ];

    if constexpr (std::is_same<IntFp, T>::value && SECURE) {

        // T* constant_terms = new T[current_layer->output_size];
        vector<T> constant_terms;
        T* prev_layer_coeffs = new T[2 * prev_activation->output_size];
        IntFp* coeff_sign = new IntFp[2 * (current_layer->max_coeffs - 1)];   
        T* copied_lc = new T[2*(current_layer->max_coeffs - 1)];
        T* non_constant_coeffs = new T[prev_activation->output_size];

        int n = 0;
        /* handle constant term */
        for(int i : neuron_list){
            T* current_lower_constraints = current_layer->lower_constraints + (i*current_layer->max_coeffs);
            
            ZKcmpPositive(current_layer->party, current_lower_constraints, ZERO_COMP_CONSTANT, coeff_sign, current_layer->max_coeffs - 1);
            for(int j = 0; j < current_layer->max_coeffs - 1; j++){
                coeff_sign[j + current_layer->max_coeffs - 1] = FIELD_ONE + coeff_sign[j].negate();
            }


            for(int j = 0; j < prev_activation->output_size; j++){
                prev_layer_coeffs[j]                           = prev_activation->lower_constraints[j*2 + 1];
                prev_layer_coeffs[j + prev_activation->output_size] = prev_activation->upper_constraints[j*2 + 1];
            }

            // select which prev LC or UC to multiply with
            for(int j = 0; j < 2*prev_activation->output_size; j++){
                prev_layer_coeffs[j] = prev_layer_coeffs[j] * coeff_sign[j];
            }

            for(int j = 0; j < current_layer->max_coeffs-1; j++){
                copied_lc[j]                                 = current_lower_constraints[j];
                copied_lc[j + current_layer->max_coeffs - 1] = current_lower_constraints[j];
            }

            // constant_terms[i] = inner_product_bundle(2*prev_activation->output_size, copied_lc, prev_layer_coeffs, current_layer->party);
            constant_terms.push_back(inner_product_bundle(2*prev_activation->output_size, copied_lc, prev_layer_coeffs, current_layer->party));

            new_backsubstituted_lower_constraints[(n + 1) * (prev_activation->input_size + 1) - 1] = current_lower_constraints[current_layer->max_coeffs - 1]; 



            /* lookahead: for non-constant term */

            for(int j = 0; j < prev_activation->output_size; j++){
                non_constant_coeffs[j] =  prev_activation->lower_constraints[j*2 + 0] * coeff_sign[j]
                                        + prev_activation->upper_constraints[j*2 + 0] * coeff_sign[j + prev_activation->output_size]; 

                non_constant_coeffs[j] = non_constant_coeffs[j] * copied_lc[j];
            }

            ZKgeneralTruncAny(current_layer->party, non_constant_coeffs, non_constant_coeffs, prev_activation->output_size, FXPSCALE);

            for(int j = 0; j < prev_activation->output_size; j++){
                new_backsubstituted_lower_constraints[n* (prev_activation->input_size + 1) + j] = non_constant_coeffs[j];
            }

            n++;
        }    

        ZKgeneralTruncAny(current_layer->party, constant_terms.data(), constant_terms.data(), constant_terms.size(), FXPSCALE);

        n = 0;
        for(int i : neuron_list){
            new_backsubstituted_lower_constraints[(n + 1) * (prev_activation->input_size + 1) - 1] = current_layer->lower_constraints[(i + 1) * (prev_activation->input_size + 1) - 1] +
                                                                                                constant_terms[n];

            n++;
        }


        T* accumulator = new T[neuron_list.size()];
        n = 0;
        for(int i : neuron_list){
            T* current_lower_constraints = new_backsubstituted_lower_constraints + (n* (prev_affine->output_size + 1));

            ZKcmpPositive(
                current_layer->party,
                current_lower_constraints,
                ZERO_COMP_CONSTANT,
                coeff_sign,
                prev_affine->output_size
            );
            for(int j = 0; j < prev_affine->output_size; j++){
                coeff_sign[j + prev_affine->output_size] = FIELD_ONE + coeff_sign[j].negate();
            
                prev_layer_coeffs[j]                            = coeff_sign[j]                            * prev_affine->lower_bounds[j];
                prev_layer_coeffs[j + prev_affine->output_size] = coeff_sign[j + prev_affine->output_size] * prev_affine->upper_bounds[j];
            }
            
            accumulator[n] = inner_product_bundle(
                prev_affine->output_size,
                current_lower_constraints,
                prev_layer_coeffs,
                current_layer->party
            ) + inner_product_bundle(
                prev_affine->output_size,
                current_lower_constraints,
                prev_layer_coeffs + prev_affine->output_size,
                current_layer->party
            );

            n++;
        }

        ZKgeneralTruncAny(current_layer->party, accumulator, accumulator, neuron_list.size(), FXPSCALE);

        n = 0;
        for(int i : neuron_list){
            current_layer->lower_bounds[i] = accumulator[n] + new_backsubstituted_lower_constraints[(n + 1) * (prev_affine->output_size + 1) - 1];

            n++;
        }



    } else {

        int n = 0;
        for(int i : neuron_list){
            T constant_term = constant<T>(0);
            for(int j = 0; j < prev_affine->output_size; j++){
                if(greater_eq_zero<T>(current_layer->lower_constraints[i * current_layer->max_coeffs + j], true)){

                    // constant due to ReLU
                    constant_term = constant_term + current_layer->lower_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->lower_constraints[j * 2 + 1];
                    
                } else {

                    // constant due to ReLU
                    constant_term = constant_term + current_layer->lower_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->upper_constraints[j * 2 + 1];

                }
            }
            new_backsubstituted_lower_constraints[(n + 1)* (prev_affine->output_size + 1) - 1] = constant_term + current_layer->lower_constraints[(i + 1) * current_layer->max_coeffs - 1];



            for(int j = 0; j < (prev_affine->output_size); j++){

                T accumulated = constant<T>(0);

                if(greater_eq_zero<T>(current_layer->lower_constraints[i * current_layer->max_coeffs + j], true)){
                    accumulated = accumulated + current_layer->lower_constraints[i * current_layer->max_coeffs + j] *
                                                prev_activation->lower_constraints[j * 2 + 0];
                    
                } else {
                    accumulated = accumulated + current_layer->lower_constraints[i * current_layer->max_coeffs + j] *
                                                prev_activation->upper_constraints[j * 2 + 0];

                }

                new_backsubstituted_lower_constraints[n * (prev_affine->output_size + 1) + j] = accumulated;
            }

            n++;
        }

        n = 0;
        for(int i : neuron_list){
            T* current_lower_constraints = new_backsubstituted_lower_constraints + (n* (prev_affine->output_size + 1));

            T accumulated = current_lower_constraints[prev_affine->output_size];

            for(int j = 0; j < prev_affine->output_size; j++){
                if(greater_eq_zero<T>((T) current_lower_constraints[j], true)){
                    accumulated = accumulated + current_lower_constraints[j] * prev_affine->lower_bounds[j];
                } else {
                    accumulated = accumulated + current_lower_constraints[j] * prev_affine->upper_bounds[j];
                }
            }

            // current_layer->lower_bounds[i] = inner_product_emp(current_layer->max_coeffs, current_lower_constraints, prev_layer_bounds_to_mult);
            current_layer->lower_bounds[i] = accumulated;

            n++;
        }

        if constexpr (std::is_same<IntFp, T>::value){
            normalize(current_layer->output_size, current_layer->lower_bounds, current_layer->lower_bounds);
        }
    }
}

template <typename T>
void backsubstitute_uc_using_prev_layers(Layer<T>* current_layer, Layer<T>* prev_activation, Layer<T>* prev_affine, Layer<T>* input_layer, set<int> neuron_list = {}){
    if(neuron_list.size() == 0){
        neuron_list = set<int>();
        for(int i = 0; i < current_layer->output_size; i++){
            neuron_list.insert(i);
        }
    }

    T* new_backsubstituted_upper_constraints = new T[neuron_list.size() * (prev_affine->output_size + 1) ];

    if constexpr (std::is_same<IntFp, T>::value && SECURE){

        // T* constant_terms = new T[current_layer->output_size];
        vector<T> constant_terms;
        T* prev_layer_coeffs = new T[2 * prev_activation->output_size];
        IntFp* coeff_sign = new IntFp[2 * (current_layer->max_coeffs - 1)];   
        T* copied_lc = new T[2*(current_layer->max_coeffs - 1)];
        T* non_constant_coeffs = new T[prev_activation->output_size];

        int n = 0;
        /* handle constant term */
        for(int i : neuron_list){
            T* current_upper_constraints = current_layer->upper_constraints + (i*current_layer->max_coeffs);
            
            ZKcmpPositive(current_layer->party, current_upper_constraints, ZERO_COMP_CONSTANT, coeff_sign, current_layer->max_coeffs - 1);
            for(int j = 0; j < current_layer->max_coeffs - 1; j++){
                coeff_sign[j + current_layer->max_coeffs - 1] = FIELD_ONE + coeff_sign[j].negate();
            }


            for(int j = 0; j < prev_activation->output_size; j++){
                prev_layer_coeffs[j]                                = prev_activation->upper_constraints[j*2 + 1];
                prev_layer_coeffs[j + prev_activation->output_size] = prev_activation->lower_constraints[j*2 + 1];
            }

            // select which prev LC or UC to multiply with
            for(int j = 0; j < 2*prev_activation->output_size; j++){
                prev_layer_coeffs[j] = prev_layer_coeffs[j] * coeff_sign[j];
            }

            for(int j = 0; j < current_layer->max_coeffs-1; j++){
                copied_lc[j]                                 = current_upper_constraints[j];
                copied_lc[j + current_layer->max_coeffs - 1] = current_upper_constraints[j];
            }

            // constant_terms[i] = inner_product_bundle(2*prev_activation->output_size, copied_lc, prev_layer_coeffs, current_layer->party);
            constant_terms.push_back(inner_product_bundle(2*prev_activation->output_size, copied_lc, prev_layer_coeffs, current_layer->party));

            new_backsubstituted_upper_constraints[(n + 1) * (prev_activation->input_size + 1) - 1] = current_upper_constraints[current_layer->max_coeffs - 1]; 



            /* lookahead: for non-constant term */

            for(int j = 0; j < prev_activation->output_size; j++){
                non_constant_coeffs[j] =  prev_activation->upper_constraints[j*2 + 0] * coeff_sign[j]
                                        + prev_activation->lower_constraints[j*2 + 0] * coeff_sign[j + prev_activation->output_size]; 

                non_constant_coeffs[j] = non_constant_coeffs[j] * copied_lc[j];
            }

            ZKgeneralTruncAny(current_layer->party, non_constant_coeffs, non_constant_coeffs, prev_activation->output_size, FXPSCALE);

            for(int j = 0; j < prev_activation->output_size; j++){
                new_backsubstituted_upper_constraints[n* (prev_activation->input_size + 1) + j] = non_constant_coeffs[j];
            }

            n++;
        }    

        ZKgeneralTruncAny(current_layer->party, constant_terms.data(), constant_terms.data(), constant_terms.size(), FXPSCALE);

        n = 0;
        for(int i : neuron_list){
            new_backsubstituted_upper_constraints[(n + 1) * (prev_activation->input_size + 1) - 1] = current_layer->upper_constraints[(i + 1) * (prev_activation->input_size + 1) - 1] +
                                                                                                constant_terms[n];

            n++;
        }


        T* accumulator = new T[neuron_list.size()];
        n = 0;
        for(int i : neuron_list){
            T* current_upper_constraints = new_backsubstituted_upper_constraints + (n* (prev_affine->output_size + 1));

            ZKcmpPositive(
                current_layer->party,
                current_upper_constraints,
                ZERO_COMP_CONSTANT,
                coeff_sign,
                prev_affine->output_size
            );
            for(int j = 0; j < prev_affine->output_size; j++){
                coeff_sign[j + prev_affine->output_size] = FIELD_ONE + coeff_sign[j].negate();
            
                prev_layer_coeffs[j]                            = coeff_sign[j]                            * prev_affine->upper_bounds[j];
                prev_layer_coeffs[j + prev_affine->output_size] = coeff_sign[j + prev_affine->output_size] * prev_affine->lower_bounds[j];
            }
            
            accumulator[n] = inner_product_bundle(
                prev_affine->output_size,
                current_upper_constraints,
                prev_layer_coeffs,
                current_layer->party
            ) + inner_product_bundle(
                prev_affine->output_size,
                current_upper_constraints,
                prev_layer_coeffs + prev_affine->output_size,
                current_layer->party
            );

            n++;
        }

        ZKgeneralTruncAny(current_layer->party, accumulator, accumulator, neuron_list.size(), FXPSCALE);

        n = 0;
        for(int i : neuron_list){
            current_layer->upper_bounds[i] = accumulator[n] + new_backsubstituted_upper_constraints[(n + 1) * (prev_affine->output_size + 1) - 1];

            n++;
        }



    } else {

        int n = 0;
        for(int i : neuron_list){
            T constant_term = constant<T>(0);
            for(int j = 0; j < prev_affine->output_size; j++){
                if(greater_eq_zero<T>(current_layer->upper_constraints[i * current_layer->max_coeffs + j], true)){

                    // constant due to ReLU
                    constant_term = constant_term + current_layer->upper_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->upper_constraints[j * 2 + 1];
                    
                } else {

                    // constant due to ReLU
                    constant_term = constant_term + current_layer->upper_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->lower_constraints[j * 2 + 1];
                }
            }
            new_backsubstituted_upper_constraints[(n + 1)* (prev_affine->output_size + 1) - 1] = constant_term + current_layer->upper_constraints[(i + 1) * current_layer->max_coeffs - 1];



            for(int j = 0; j < (prev_affine->output_size); j++){

                T accumulated = constant<T>(0);

                if(greater_eq_zero<T>(current_layer->upper_constraints[i * current_layer->max_coeffs + j], true)){
                    accumulated = accumulated + current_layer->upper_constraints[i * current_layer->max_coeffs + j] *
                                                prev_activation->upper_constraints[j * 2 + 0];
                    
                } else {
                    accumulated = accumulated + current_layer->upper_constraints[i * current_layer->max_coeffs + j] *
                                                prev_activation->lower_constraints[j * 2 + 0];

                }

                new_backsubstituted_upper_constraints[n * (prev_affine->output_size + 1) + j] = accumulated;
            }

            n++;
        }

        n = 0;
        for(int i : neuron_list){
            T* current_upper_constraints = new_backsubstituted_upper_constraints + (n* (prev_affine->output_size + 1));

            T accumulated = current_upper_constraints[prev_affine->output_size];

            for(int j = 0; j < prev_affine->output_size; j++){
                if(greater_eq_zero<T>((T) current_upper_constraints[j], true)){
                    accumulated = accumulated + current_upper_constraints[j] * prev_affine->upper_bounds[j];
                } else {
                    accumulated = accumulated + current_upper_constraints[j] * prev_affine->lower_bounds[j];
                }
            }

            // current_layer->upper_bounds[i] = inner_product_emp(current_layer->max_coeffs, current_upper_constraints, prev_layer_bounds_to_mult);
            current_layer->upper_bounds[i] = accumulated;

            n++;
        }

        if constexpr (std::is_same<IntFp, T>::value){
            normalize(current_layer->output_size, current_layer->upper_bounds, current_layer->upper_bounds);
        }
    }
}



// the next two functions substitute bounds of the INPUT layer in the (optimized) backsubstituted constraints
template <typename T>
void backsubstitute_lc_using_prev_affine(Layer<T>* current_layer, Layer<T>* prev_activation, Layer<T>* prev_affine, Layer<T>* input_layer, set<int> neuron_list = {}){

    if(neuron_list.size() == 0){
        neuron_list = set<int>();
        for(int i = 0; i < current_layer->output_size; i++){
            neuron_list.insert(i);
        }
    }

    T* new_backsubstituted_lower_constraints = new T[neuron_list.size() * (NUM_FEATURES[CURR_DATASET] + 1) ];

    if constexpr (std::is_same<IntFp, T>::value && SECURE){

        T* coeff_sign = new T[2 * prev_affine->output_size];
        T* prev_layer_term = new T[2 * prev_affine->output_size];
        T* constant_terms = new T[2 * neuron_list.size()];  // activation - affine

        int n = 0;
        for(int i : neuron_list){

            ZKcmpPositive(
                current_layer->party,
                current_layer->lower_constraints + i * current_layer->max_coeffs,
                ZERO_COMP_CONSTANT,
                coeff_sign,
                prev_affine->output_size
            );
            for(int j = 0; j < prev_affine->output_size; j++){
                coeff_sign[j + prev_affine->output_size] = FIELD_ONE + coeff_sign[j].negate();
            }
            cerr << "time cmp = " << time_from(start_time)*1.0/1e6 << " s\n";
            
            // activation contribution
            for(int j = 0; j < prev_affine->output_size; j++){
                prev_layer_term[j]                            = coeff_sign[j]                            * prev_activation->lower_constraints[j * 2 + 1];
                prev_layer_term[j + prev_affine->output_size] = coeff_sign[j + prev_affine->output_size] * prev_activation->upper_constraints[j * 2 + 1];
            }
            constant_terms[n * 2 + 0] = inner_product_bundle(
                prev_activation->output_size,
                current_layer->lower_constraints + i * current_layer->max_coeffs,
                prev_layer_term,
                current_layer->party
            ) + inner_product_bundle(
                prev_activation->output_size,
                current_layer->lower_constraints + i * current_layer->max_coeffs,
                prev_layer_term + prev_activation->output_size,
                current_layer->party
            );
            cerr << "time act const. = " << time_from(start_time)*1.0/1e6 << " s\n";


            // affine contribution
            for(int j = 0; j < prev_affine->output_size; j++){
                prev_layer_term[j]                            = coeff_sign[j]                            * inner_product_bundle(1, prev_activation->lower_constraints + j * 2, prev_affine->backsubstituted_lower_constraints + (j + 1) * (NUM_FEATURES[CURR_DATASET] + 1) - 1, current_layer->party);
                prev_layer_term[j + prev_affine->output_size] = coeff_sign[j + prev_affine->output_size] * inner_product_bundle(1, prev_activation->upper_constraints + j * 2, prev_affine->backsubstituted_upper_constraints + (j + 1) * (NUM_FEATURES[CURR_DATASET] + 1) - 1, current_layer->party);
            }
            ZKgeneralTruncAny(current_layer->party, prev_layer_term, prev_layer_term, 2 * prev_affine->output_size, FXPSCALE);

            constant_terms[n * 2 + 1] = inner_product_bundle(
                prev_activation->output_size,
                current_layer->lower_constraints + i * current_layer->max_coeffs,
                prev_layer_term,
                current_layer->party
            ) + inner_product_bundle(
                prev_activation->output_size,
                current_layer->lower_constraints + i * current_layer->max_coeffs,
                prev_layer_term + prev_activation->output_size,
                current_layer->party
            );
            cerr << "time aff const. = " << time_from(start_time)*1.0/1e6 << " s\n";

            /* =========== NON-CONSTANT TERMS =========== */
            for(int k = 0; k < (NUM_FEATURES[CURR_DATASET]); k++){

                for(int j = 0; j < prev_affine->output_size; j++){
                    // prev_layer_term[j]                            = coeff_sign[j]                            * prev_activation->lower_constraints[j * 2 + 0] * prev_affine->backsubstituted_lower_constraints[j * (NUM_FEATURES[CURR_DATASET] + 1) + k];
                    // prev_layer_term[j + prev_affine->output_size] = coeff_sign[j + prev_affine->output_size] * prev_activation->upper_constraints[j * 2 + 0] * prev_affine->backsubstituted_upper_constraints[j * (NUM_FEATURES[CURR_DATASET] + 1) + k];

                    prev_layer_term[j]                            = coeff_sign[j]                            * inner_product_bundle(1, prev_activation->lower_constraints + j * 2, prev_affine->backsubstituted_lower_constraints + j * (NUM_FEATURES[CURR_DATASET] + 1) + k, current_layer->party);
                    prev_layer_term[j + prev_affine->output_size] = coeff_sign[j + prev_affine->output_size] * inner_product_bundle(1, prev_activation->upper_constraints + j * 2, prev_affine->backsubstituted_upper_constraints + j * (NUM_FEATURES[CURR_DATASET] + 1) + k, current_layer->party);
                }
                ZKgeneralTruncAny(current_layer->party, prev_layer_term, prev_layer_term, 2 * prev_affine->output_size, FXPSCALE);

                new_backsubstituted_lower_constraints[n * (NUM_FEATURES[CURR_DATASET] + 1) + k] = inner_product_bundle(
                    prev_activation->output_size,
                    current_layer->lower_constraints + i * current_layer->max_coeffs,
                    prev_layer_term,
                    current_layer->party
                ) + inner_product_bundle(
                    prev_activation->output_size,
                    current_layer->lower_constraints + i * current_layer->max_coeffs,
                    prev_layer_term + prev_activation->output_size,
                    current_layer->party
                );
            }
            ZKgeneralTruncAny(
                current_layer->party, 
                new_backsubstituted_lower_constraints + i * (NUM_FEATURES[CURR_DATASET] + 1), 
                new_backsubstituted_lower_constraints + i * (NUM_FEATURES[CURR_DATASET] + 1), 
                (NUM_FEATURES[CURR_DATASET]), 
                FXPSCALE
            );

            n++;
        }

        ZKgeneralTruncAny(current_layer->party, constant_terms, constant_terms, 2 * neuron_list.size(), FXPSCALE);
        cerr << "time non-const. = " << time_from(start_time)*1.0/1e6 << " s\n";

        n = 0;
        for(int i : neuron_list){
            new_backsubstituted_lower_constraints[(n + 1)* (NUM_FEATURES[CURR_DATASET] + 1) - 1] = constant_terms[n * 2 + 0] + constant_terms[n * 2 + 1] +
                                                                                                   current_layer->lower_constraints[(i + 1) * current_layer->max_coeffs - 1];

            n++;
        }

        n = 0;
        for(int i : neuron_list){
            for(int j = 0; j < (NUM_FEATURES[CURR_DATASET] + 1); j++){
                current_layer->backsubstituted_lower_constraints[i * (NUM_FEATURES[CURR_DATASET] + 1) + j] = new_backsubstituted_lower_constraints[n * (NUM_FEATURES[CURR_DATASET] + 1) + j];
            }

            n++;
        }
        delete[] new_backsubstituted_lower_constraints;
        cerr << "time final rearr = " << time_from(start_time)*1.0/1e6 << " s\n";


        delete[] coeff_sign;
        coeff_sign = new T[2 * NUM_FEATURES[CURR_DATASET]];
        delete[] prev_layer_term;
        prev_layer_term = new T[2 * NUM_FEATURES[CURR_DATASET]];

        n = 0;
        for(int i : neuron_list){
            T* current_lower_constraints = current_layer->backsubstituted_lower_constraints + (i* (NUM_FEATURES[CURR_DATASET] + 1));

            ZKcmpPositive(
                current_layer->party,
                current_layer->backsubstituted_lower_constraints + i * (NUM_FEATURES[CURR_DATASET] + 1),
                ZERO_COMP_CONSTANT,
                coeff_sign,
                NUM_FEATURES[CURR_DATASET]
            );
            for(int j = 0; j < NUM_FEATURES[CURR_DATASET]; j++){
                coeff_sign[j + NUM_FEATURES[CURR_DATASET]] = FIELD_ONE + coeff_sign[j].negate();
            
                prev_layer_term[j]                              = coeff_sign[j]                              * input_layer->lower_bounds[j];
                prev_layer_term[j + NUM_FEATURES[CURR_DATASET]] = coeff_sign[j + NUM_FEATURES[CURR_DATASET]] * input_layer->upper_bounds[j];
            }
            
            constant_terms[n] = inner_product_bundle(
                NUM_FEATURES[CURR_DATASET],
                current_layer->backsubstituted_lower_constraints + i * (NUM_FEATURES[CURR_DATASET] + 1),
                prev_layer_term,
                current_layer->party
            ) + inner_product_bundle(
                NUM_FEATURES[CURR_DATASET],
                current_layer->backsubstituted_lower_constraints + i * (NUM_FEATURES[CURR_DATASET] + 1),
                prev_layer_term + NUM_FEATURES[CURR_DATASET],
                current_layer->party
            );

            n++;
        }

        ZKgeneralTruncAny(current_layer->party, constant_terms, constant_terms, neuron_list.size(), FXPSCALE);

        n = 0;
        for(int i : neuron_list){
            current_layer->lower_bounds[i] = constant_terms[n] + current_layer->backsubstituted_lower_constraints[(i + 1) * (NUM_FEATURES[CURR_DATASET] + 1) - 1];

            n++;
        }

    } else {

        for(int i : neuron_list){
            T constant_term = constant<T>(0);
            for(int j = 0; j < prev_affine->output_size; j++){
                if(greater_eq_zero<T>(current_layer->lower_constraints[i * current_layer->max_coeffs + j], true)){

                    // constant due to ReLU
                    constant_term = constant_term + current_layer->lower_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->lower_constraints[j * 2 + 1];

                    // constant due to Affine
                    constant_term = constant_term + current_layer->lower_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->lower_constraints[j * 2 + 0] *
                                                    prev_affine->backsubstituted_lower_constraints[(j + 1) * (NUM_FEATURES[CURR_DATASET] + 1) - 1];
                    
                } else {

                    // constant due to ReLU
                    constant_term = constant_term + current_layer->lower_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->upper_constraints[j * 2 + 1];

                    // constant due to Affine
                    constant_term = constant_term + current_layer->lower_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->upper_constraints[j * 2 + 0] *
                                                    prev_affine->backsubstituted_upper_constraints[(j + 1) * (NUM_FEATURES[CURR_DATASET] + 1) - 1];
                }
            }
            new_backsubstituted_lower_constraints[(i + 1)* (NUM_FEATURES[CURR_DATASET] + 1) - 1] = constant_term + current_layer->lower_constraints[(i + 1) * current_layer->max_coeffs - 1];



            for(int k = 0; k < (NUM_FEATURES[CURR_DATASET]); k++){

                T accumulated = constant<T>(0);

                for(int j = 0; j < prev_affine->output_size; j++){
                    if(greater_eq_zero<T>(current_layer->lower_constraints[i * current_layer->max_coeffs + j], true)){
                        accumulated = accumulated + current_layer->lower_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->lower_constraints[j * 2 + 0] *
                                                    prev_affine->backsubstituted_lower_constraints[j * (NUM_FEATURES[CURR_DATASET] + 1) + k];
                        
                    } else {
                        accumulated = accumulated + current_layer->lower_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->upper_constraints[j * 2 + 0] *
                                                    prev_affine->backsubstituted_upper_constraints[j * (NUM_FEATURES[CURR_DATASET] + 1) + k];

                    }
                }

                new_backsubstituted_lower_constraints[i * (NUM_FEATURES[CURR_DATASET] + 1) + k] = accumulated;
            }

        }

        delete[] current_layer->backsubstituted_lower_constraints;
        current_layer->backsubstituted_lower_constraints = new_backsubstituted_lower_constraints;


        for(int i : neuron_list){
            T* current_lower_constraints = current_layer->backsubstituted_lower_constraints + (i* (NUM_FEATURES[CURR_DATASET] + 1));

            T accumulated = current_lower_constraints[NUM_FEATURES[CURR_DATASET]];

            for(int j = 0; j < input_layer->output_size; j++){
                if(greater_eq_zero<T>((T) current_lower_constraints[j], true)){
                    accumulated = accumulated + current_lower_constraints[j] * input_layer->lower_bounds[j];
                } else {
                    accumulated = accumulated + current_lower_constraints[j] * input_layer->upper_bounds[j];
                }
            }

            // current_layer->lower_bounds[i] = inner_product_emp(current_layer->max_coeffs, current_lower_constraints, prev_layer_bounds_to_mult);
            current_layer->lower_bounds[i] = accumulated;
        }

        if constexpr (std::is_same<IntFp, T>::value){
            normalize(current_layer->output_size, current_layer->lower_bounds, current_layer->lower_bounds);
        }
    } 

}

template <typename T>
void backsubstitute_uc_using_prev_affine(Layer<T>* current_layer, Layer<T>* prev_activation, Layer<T>* prev_affine, Layer<T>* input_layer, set<int> neuron_list = {}){

    if(neuron_list.size() == 0){
        neuron_list = set<int>();
        for(int i = 0; i < current_layer->output_size; i++){
            neuron_list.insert(i);
        }
    }

    T* new_backsubstituted_upper_constraints = new T[neuron_list.size() * (NUM_FEATURES[CURR_DATASET] + 1) ];

    if constexpr (std::is_same<IntFp, T>::value && SECURE){

        T* coeff_sign = new T[2 * prev_affine->output_size];
        T* prev_layer_term = new T[2 * prev_affine->output_size];
        T* constant_terms = new T[2 * neuron_list.size()];  // activation - affine

        int n = 0;
        for(int i : neuron_list){

            ZKcmpPositive(
                current_layer->party,
                current_layer->upper_constraints + i * current_layer->max_coeffs,
                ZERO_COMP_CONSTANT,
                coeff_sign,
                prev_affine->output_size
            );
            for(int j = 0; j < prev_affine->output_size; j++){
                coeff_sign[j + prev_affine->output_size] = FIELD_ONE + coeff_sign[j].negate();
            }

            
            // activation contribution
            for(int j = 0; j < prev_affine->output_size; j++){
                prev_layer_term[j]                            = coeff_sign[j]                            * prev_activation->upper_constraints[j * 2 + 1];
                prev_layer_term[j + prev_affine->output_size] = coeff_sign[j + prev_affine->output_size] * prev_activation->lower_constraints[j * 2 + 1];
            }
            constant_terms[n * 2 + 0] = inner_product_bundle(
                prev_activation->output_size,
                current_layer->upper_constraints + i * current_layer->max_coeffs,
                prev_layer_term,
                current_layer->party
            ) + inner_product_bundle(
                prev_activation->output_size,
                current_layer->upper_constraints + i * current_layer->max_coeffs,
                prev_layer_term + prev_activation->output_size,
                current_layer->party
            );


            // affine contribution
            for(int j = 0; j < prev_affine->output_size; j++){
                prev_layer_term[j]                            = coeff_sign[j]                            * inner_product_bundle(1, prev_activation->upper_constraints + j * 2, prev_affine->backsubstituted_upper_constraints + (j + 1) * (NUM_FEATURES[CURR_DATASET] + 1) - 1, current_layer->party);
                prev_layer_term[j + prev_affine->output_size] = coeff_sign[j + prev_affine->output_size] * inner_product_bundle(1, prev_activation->lower_constraints + j * 2, prev_affine->backsubstituted_lower_constraints + (j + 1) * (NUM_FEATURES[CURR_DATASET] + 1) - 1, current_layer->party);
            }
            ZKgeneralTruncAny(current_layer->party, prev_layer_term, prev_layer_term, 2 * prev_affine->output_size, FXPSCALE);

            constant_terms[n * 2 + 1] = inner_product_bundle(
                prev_activation->output_size,
                current_layer->upper_constraints + i * current_layer->max_coeffs,
                prev_layer_term,
                current_layer->party
            ) + inner_product_bundle(
                prev_activation->output_size,
                current_layer->upper_constraints + i * current_layer->max_coeffs,
                prev_layer_term + prev_activation->output_size,
                current_layer->party
            );


            /* =========== NON-CONSTANT TERMS =========== */
            for(int k = 0; k < (NUM_FEATURES[CURR_DATASET]); k++){

                for(int j = 0; j < prev_affine->output_size; j++){
                    // prev_layer_term[j]                            = coeff_sign[j]                            * prev_activation->lower_constraints[j * 2 + 0] * prev_affine->backsubstituted_lower_constraints[j * (NUM_FEATURES[CURR_DATASET] + 1) + k];
                    // prev_layer_term[j + prev_affine->output_size] = coeff_sign[j + prev_affine->output_size] * prev_activation->upper_constraints[j * 2 + 0] * prev_affine->backsubstituted_upper_constraints[j * (NUM_FEATURES[CURR_DATASET] + 1) + k];

                    prev_layer_term[j]                            = coeff_sign[j]                            * inner_product_bundle(1, prev_activation->upper_constraints + j * 2, prev_affine->backsubstituted_upper_constraints + j * (NUM_FEATURES[CURR_DATASET] + 1) + k, current_layer->party);
                    prev_layer_term[j + prev_affine->output_size] = coeff_sign[j + prev_affine->output_size] * inner_product_bundle(1, prev_activation->lower_constraints + j * 2, prev_affine->backsubstituted_lower_constraints + j * (NUM_FEATURES[CURR_DATASET] + 1) + k, current_layer->party);
                }
                ZKgeneralTruncAny(current_layer->party, prev_layer_term, prev_layer_term, 2 * prev_affine->output_size, FXPSCALE);

                new_backsubstituted_upper_constraints[n * (NUM_FEATURES[CURR_DATASET] + 1) + k] = inner_product_bundle(
                    prev_activation->output_size,
                    current_layer->upper_constraints + i * current_layer->max_coeffs,
                    prev_layer_term,
                    current_layer->party
                ) + inner_product_bundle(
                    prev_activation->output_size,
                    current_layer->upper_constraints + i * current_layer->max_coeffs,
                    prev_layer_term + prev_activation->output_size,
                    current_layer->party
                );
            }
            ZKgeneralTruncAny(
                current_layer->party, 
                new_backsubstituted_upper_constraints + i * (NUM_FEATURES[CURR_DATASET] + 1), 
                new_backsubstituted_upper_constraints + i * (NUM_FEATURES[CURR_DATASET] + 1), 
                (NUM_FEATURES[CURR_DATASET]), 
                FXPSCALE
            );

            n++;
        }

        ZKgeneralTruncAny(current_layer->party, constant_terms, constant_terms, 2 * neuron_list.size(), FXPSCALE);

        n = 0;
        for(int i : neuron_list){
            new_backsubstituted_upper_constraints[(n + 1)* (NUM_FEATURES[CURR_DATASET] + 1) - 1] = constant_terms[n * 2 + 0] + constant_terms[n * 2 + 1] +
                                                                                                   current_layer->upper_constraints[(i + 1) * current_layer->max_coeffs - 1];

            n++;
        }

        n = 0;
        for(int i : neuron_list){
            for(int j = 0; j < (NUM_FEATURES[CURR_DATASET] + 1); j++){
                current_layer->backsubstituted_upper_constraints[i * (NUM_FEATURES[CURR_DATASET] + 1) + j] = new_backsubstituted_upper_constraints[n * (NUM_FEATURES[CURR_DATASET] + 1) + j];
            }

            n++;
        }
        delete[] new_backsubstituted_upper_constraints;


        delete[] coeff_sign;
        coeff_sign = new T[2 * NUM_FEATURES[CURR_DATASET]];
        delete[] prev_layer_term;
        prev_layer_term = new T[2 * NUM_FEATURES[CURR_DATASET]];

        n = 0;
        for(int i : neuron_list){
            T* current_upper_constraints = current_layer->backsubstituted_upper_constraints + (i* (NUM_FEATURES[CURR_DATASET] + 1));

            ZKcmpPositive(
                current_layer->party,
                current_layer->backsubstituted_upper_constraints + i * (NUM_FEATURES[CURR_DATASET] + 1),
                ZERO_COMP_CONSTANT,
                coeff_sign,
                NUM_FEATURES[CURR_DATASET]
            );
            for(int j = 0; j < NUM_FEATURES[CURR_DATASET]; j++){
                coeff_sign[j + NUM_FEATURES[CURR_DATASET]] = FIELD_ONE + coeff_sign[j].negate();
            
                prev_layer_term[j]                              = coeff_sign[j]                              * input_layer->upper_bounds[j];
                prev_layer_term[j + NUM_FEATURES[CURR_DATASET]] = coeff_sign[j + NUM_FEATURES[CURR_DATASET]] * input_layer->lower_bounds[j];
            }
            
            constant_terms[n] = inner_product_bundle(
                NUM_FEATURES[CURR_DATASET],
                current_layer->backsubstituted_upper_constraints + i * (NUM_FEATURES[CURR_DATASET] + 1),
                prev_layer_term,
                current_layer->party
            ) + inner_product_bundle(
                NUM_FEATURES[CURR_DATASET],
                current_layer->backsubstituted_upper_constraints + i * (NUM_FEATURES[CURR_DATASET] + 1),
                prev_layer_term + NUM_FEATURES[CURR_DATASET],
                current_layer->party
            );

            n++;
        }

        ZKgeneralTruncAny(current_layer->party, constant_terms, constant_terms, neuron_list.size(), FXPSCALE);

        n = 0;
        for(int i : neuron_list){
            current_layer->upper_bounds[i] = constant_terms[n] + current_layer->backsubstituted_upper_constraints[(i + 1) * (NUM_FEATURES[CURR_DATASET] + 1) - 1];

            n++;
        }

    } else {

        for(int i : neuron_list){
            T constant_term = constant<T>(0);
            for(int j = 0; j < prev_affine->output_size; j++){
                if(greater_eq_zero<T>(current_layer->upper_constraints[i * current_layer->max_coeffs + j], false)){

                    // constant due to ReLU
                    constant_term = constant_term + current_layer->upper_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->upper_constraints[j * 2 + 1];

                    // constant due to Affine
                    constant_term = constant_term + current_layer->upper_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->upper_constraints[j * 2 + 0] *
                                                    prev_affine->backsubstituted_upper_constraints[(j + 1) * (NUM_FEATURES[CURR_DATASET] + 1) - 1];
                    
                } else {

                    // constant due to ReLU
                    constant_term = constant_term + current_layer->upper_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->lower_constraints[j * 2 + 1];

                    // constant due to Affine
                    constant_term = constant_term + current_layer->upper_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->lower_constraints[j * 2 + 0] *
                                                    prev_affine->backsubstituted_lower_constraints[(j + 1) * (NUM_FEATURES[CURR_DATASET] + 1) - 1];
                }
            }
            new_backsubstituted_upper_constraints[(i + 1)* (NUM_FEATURES[CURR_DATASET] + 1) - 1] = constant_term + current_layer->upper_constraints[(i + 1) * current_layer->max_coeffs - 1];



            for(int k = 0; k < (NUM_FEATURES[CURR_DATASET]); k++){

                T accumulated = constant<T>(0);

                for(int j = 0; j < prev_affine->output_size; j++){
                    if(greater_eq_zero<T>(current_layer->upper_constraints[i * current_layer->max_coeffs + j], false)){
                        accumulated = accumulated + current_layer->upper_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->upper_constraints[j * 2 + 0] *
                                                    prev_affine->backsubstituted_upper_constraints[j * (NUM_FEATURES[CURR_DATASET] + 1) + k];
                        
                    } else {
                        accumulated = accumulated + current_layer->upper_constraints[i * current_layer->max_coeffs + j] *
                                                    prev_activation->lower_constraints[j * 2 + 0] *
                                                    prev_affine->backsubstituted_lower_constraints[j * (NUM_FEATURES[CURR_DATASET] + 1) + k];

                    }
                }

                new_backsubstituted_upper_constraints[i * (NUM_FEATURES[CURR_DATASET] + 1) + k] = accumulated;
            }

        }

        delete[] current_layer->backsubstituted_upper_constraints;
        current_layer->backsubstituted_upper_constraints = new_backsubstituted_upper_constraints;


        for(int i : neuron_list){
            T* current_upper_constraints = current_layer->backsubstituted_upper_constraints + (i* (NUM_FEATURES[CURR_DATASET] + 1));

            T accumulated = current_upper_constraints[NUM_FEATURES[CURR_DATASET]];

            for(int j = 0; j < input_layer->output_size; j++){
                if(greater_eq_zero<T>((T) current_upper_constraints[j], false)){
                    accumulated = accumulated + current_upper_constraints[j] * input_layer->upper_bounds[j];
                } else {
                    accumulated = accumulated + current_upper_constraints[j] * input_layer->lower_bounds[j];
                }
            }

            // current_layer->upper_bounds[i] = inner_product_emp(current_layer->max_coeffs, current_upper_constraints, prev_layer_bounds_to_mult);
            current_layer->upper_bounds[i] = accumulated;
        }

        if constexpr (std::is_same<IntFp, T>::value){
            normalize(current_layer->output_size, current_layer->upper_bounds, current_layer->upper_bounds);
        }
    }

}





// conv 

// the next two functions substitute bounds of the previous AFFINE layer in the (optimized) backsubstituted constraints
template <typename T>
void backsubstitute_conv_lc_using_prev_layers(Conv2D<T>* current_layer, Layer<T>* prev_activation, Layer<T>* prev_conv, Layer<T>* input_layer, set<int> neuron_list = {}){
    if(neuron_list.size() == 0){
        neuron_list = set<int>();
        for(int i = 0; i < current_layer->output_size; i++){
            neuron_list.insert(i);
        }
    }

    vector<vector<T>>* new_backsubstituted_conv_lower_constraints = new vector<vector<T>>((*current_layer->predecessors)[0].size() + 1);
    vector<vector<int>>* new_predecessors = new vector<vector<int>>(neuron_list.size());

    if constexpr (std::is_same<IntFp, T>::value && SECURE) {

        // T* constant_terms = new T[current_layer->output_size];
        vector<T> constant_terms;
        T* prev_layer_coeffs = new T[2 * prev_activation->output_size];
        IntFp* coeff_sign = new IntFp[2 * (current_layer->max_coeffs - 1)];   
        T* copied_lc = new T[2*(current_layer->max_coeffs - 1)];
        T* non_constant_coeffs = new T[prev_activation->output_size];

        int n = 0;
        /* handle constant term */
        for(int i : neuron_list){
            T* current_lower_constraints = current_layer->lower_constraints + (i*current_layer->max_coeffs);
            
            ZKcmpPositive(current_layer->party, current_lower_constraints, ZERO_COMP_CONSTANT, coeff_sign, current_layer->max_coeffs - 1);
            for(int j = 0; j < current_layer->max_coeffs - 1; j++){
                coeff_sign[j + current_layer->max_coeffs - 1] = FIELD_ONE + coeff_sign[j].negate();
            }


            for(int j = 0; j < prev_activation->output_size; j++){
                prev_layer_coeffs[j]                           = prev_activation->lower_constraints[j*2 + 1];
                prev_layer_coeffs[j + prev_activation->output_size] = prev_activation->upper_constraints[j*2 + 1];
            }

            // select which prev LC or UC to multiply with
            for(int j = 0; j < 2*prev_activation->output_size; j++){
                prev_layer_coeffs[j] = prev_layer_coeffs[j] * coeff_sign[j];
            }

            for(int j = 0; j < current_layer->max_coeffs-1; j++){
                copied_lc[j]                                 = current_lower_constraints[j];
                copied_lc[j + current_layer->max_coeffs - 1] = current_lower_constraints[j];
            }

            // constant_terms[i] = inner_product_bundle(2*prev_activation->output_size, copied_lc, prev_layer_coeffs, current_layer->party);
            constant_terms.push_back(inner_product_bundle(2*prev_activation->output_size, copied_lc, prev_layer_coeffs, current_layer->party));

            new_backsubstituted_conv_lower_constraints[(n + 1) * (prev_activation->input_size + 1) - 1] = current_lower_constraints[current_layer->max_coeffs - 1]; 



            /* lookahead: for non-constant term */

            for(int j = 0; j < prev_activation->output_size; j++){
                non_constant_coeffs[j] =  prev_activation->lower_constraints[j*2 + 0] * coeff_sign[j]
                                        + prev_activation->upper_constraints[j*2 + 0] * coeff_sign[j + prev_activation->output_size]; 

                non_constant_coeffs[j] = non_constant_coeffs[j] * copied_lc[j];
            }

            ZKgeneralTruncAny(current_layer->party, non_constant_coeffs, non_constant_coeffs, prev_activation->output_size, FXPSCALE);

            for(int j = 0; j < prev_activation->output_size; j++){
                new_backsubstituted_conv_lower_constraints[n* (prev_activation->input_size + 1) + j] = non_constant_coeffs[j];
            }

            n++;
        }    

        ZKgeneralTruncAny(current_layer->party, constant_terms.data(), constant_terms.data(), constant_terms.size(), FXPSCALE);

        n = 0;
        for(int i : neuron_list){
            new_backsubstituted_conv_lower_constraints[(n + 1) * (prev_activation->input_size + 1) - 1] = current_layer->lower_constraints[(i + 1) * (prev_activation->input_size + 1) - 1] +
                                                                                                constant_terms[n];

            n++;
        }


        T* accumulator = new T[neuron_list.size()];
        n = 0;
        for(int i : neuron_list){
            T* current_lower_constraints = new_backsubstituted_conv_lower_constraints + (n* (prev_conv->output_size + 1));

            ZKcmpPositive(
                current_layer->party,
                current_lower_constraints,
                ZERO_COMP_CONSTANT,
                coeff_sign,
                prev_conv->output_size
            );
            for(int j = 0; j < prev_conv->output_size; j++){
                coeff_sign[j + prev_conv->output_size] = FIELD_ONE + coeff_sign[j].negate();
            
                prev_layer_coeffs[j]                            = coeff_sign[j]                            * prev_conv->lower_bounds[j];
                prev_layer_coeffs[j + prev_conv->output_size] = coeff_sign[j + prev_conv->output_size] * prev_conv->upper_bounds[j];
            }
            
            accumulator[n] = inner_product_bundle(
                prev_conv->output_size,
                current_lower_constraints,
                prev_layer_coeffs,
                current_layer->party
            ) + inner_product_bundle(
                prev_conv->output_size,
                current_lower_constraints,
                prev_layer_coeffs + prev_conv->output_size,
                current_layer->party
            );

            n++;
        }

        ZKgeneralTruncAny(current_layer->party, accumulator, accumulator, neuron_list.size(), FXPSCALE);

        n = 0;
        for(int i : neuron_list){
            current_layer->lower_bounds[i] = accumulator[n] + new_backsubstituted_conv_lower_constraints[(n + 1) * (prev_conv->output_size + 1) - 1];

            n++;
        }



    } else {

        int n = 0;
        for(int i : neuron_list){
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

                T non_const_prev_constraint;
                T const_prev_constraint;
                if(greater_eq_zero<T>(j_th_lc, false)){
                    non_const_prev_constraint = prev_activation->lower_constraints[j_th_predecessor * 2 + 0];
                    const_prev_constraint = prev_activation->lower_constraints[j_th_predecessor * 2 + 1];
                } else {
                    non_const_prev_constraint = prev_activation->upper_constraints[j_th_predecessor * 2 + 0];
                    const_prev_constraint = prev_activation->upper_constraints[j_th_predecessor * 2 + 1];        
                }

                (*new_backsubstituted_conv_lower_constraints)[n].push_back(j_th_lc * non_const_prev_constraint);
                new_const_term = new_const_term + j_th_lc * const_prev_constraint;

                (*new_predecessors)[n].push_back(j_th_predecessor);

                j++;
            }

            (*new_backsubstituted_conv_lower_constraints)[n].push_back(new_const_term);
            
            assert(
                (*new_backsubstituted_conv_lower_constraints)[n].size() ==
                (*new_predecessors)[n].size() + 1
            );

            n++;
        }

        n = 0;
        for(int i : neuron_list){
            vector<T> current_lower_constraints = (*new_backsubstituted_conv_lower_constraints)[n];

            T accumulated = current_lower_constraints[prev_conv->output_size];

            int* pred = (*current_layer->predecessors)[i].data();
            int* end = pred + (*current_layer->predecessors)[i].size();

            int j = 0;
            for(; pred != end; pred++){

                T j_th_lc = current_lower_constraints[j];

                int j_th_predecessor = *pred;
                if(greater_eq_zero<T>((T) j_th_lc, true)){
                    accumulated = accumulated + j_th_lc * prev_conv->lower_bounds[j_th_predecessor];
                } else {
                    accumulated = accumulated + j_th_lc * prev_conv->upper_bounds[j_th_predecessor];
                }

                j++;
            }

            // current_layer->lower_bounds[i] = inner_product_emp(current_layer->max_coeffs, current_lower_constraints, prev_layer_bounds_to_mult);
            current_layer->lower_bounds[i] = accumulated;

            n++;
        }
    }
}

template <typename T>
void backsubstitute_conv_uc_using_prev_layers(Conv2D<T>* current_layer, Layer<T>* prev_activation, Layer<T>* prev_conv, Layer<T>* input_layer, set<int> neuron_list = {}){
    if(neuron_list.size() == 0){
        neuron_list = set<int>();
        for(int i = 0; i < current_layer->output_size; i++){
            neuron_list.insert(i);
        }
    }

    vector<vector<T>>* new_backsubstituted_conv_upper_constraints = new vector<vector<T>>((*current_layer->predecessors)[0].size() + 1);
    vector<vector<int>>* new_predecessors = new vector<vector<int>>(neuron_list.size());

    if constexpr (std::is_same<IntFp, T>::value && SECURE) {

        // T* constant_terms = new T[current_layer->output_size];
        vector<T> constant_terms;
        T* prev_layer_coeffs = new T[2 * prev_activation->output_size];
        IntFp* coeff_sign = new IntFp[2 * (current_layer->max_coeffs - 1)];   
        T* copied_lc = new T[2*(current_layer->max_coeffs - 1)];
        T* non_constant_coeffs = new T[prev_activation->output_size];

        int n = 0;
        /* handle constant term */
        for(int i : neuron_list){
            T* current_lower_constraints = current_layer->lower_constraints + (i*current_layer->max_coeffs);
            
            ZKcmpPositive(current_layer->party, current_lower_constraints, ZERO_COMP_CONSTANT, coeff_sign, current_layer->max_coeffs - 1);
            for(int j = 0; j < current_layer->max_coeffs - 1; j++){
                coeff_sign[j + current_layer->max_coeffs - 1] = FIELD_ONE + coeff_sign[j].negate();
            }


            for(int j = 0; j < prev_activation->output_size; j++){
                prev_layer_coeffs[j]                           = prev_activation->lower_constraints[j*2 + 1];
                prev_layer_coeffs[j + prev_activation->output_size] = prev_activation->upper_constraints[j*2 + 1];
            }

            // select which prev LC or UC to multiply with
            for(int j = 0; j < 2*prev_activation->output_size; j++){
                prev_layer_coeffs[j] = prev_layer_coeffs[j] * coeff_sign[j];
            }

            for(int j = 0; j < current_layer->max_coeffs-1; j++){
                copied_lc[j]                                 = current_lower_constraints[j];
                copied_lc[j + current_layer->max_coeffs - 1] = current_lower_constraints[j];
            }

            // constant_terms[i] = inner_product_bundle(2*prev_activation->output_size, copied_lc, prev_layer_coeffs, current_layer->party);
            constant_terms.push_back(inner_product_bundle(2*prev_activation->output_size, copied_lc, prev_layer_coeffs, current_layer->party));

            new_backsubstituted_conv_upper_constraints[(n + 1) * (prev_activation->input_size + 1) - 1] = current_lower_constraints[current_layer->max_coeffs - 1]; 



            /* lookahead: for non-constant term */

            for(int j = 0; j < prev_activation->output_size; j++){
                non_constant_coeffs[j] =  prev_activation->lower_constraints[j*2 + 0] * coeff_sign[j]
                                        + prev_activation->upper_constraints[j*2 + 0] * coeff_sign[j + prev_activation->output_size]; 

                non_constant_coeffs[j] = non_constant_coeffs[j] * copied_lc[j];
            }

            ZKgeneralTruncAny(current_layer->party, non_constant_coeffs, non_constant_coeffs, prev_activation->output_size, FXPSCALE);

            for(int j = 0; j < prev_activation->output_size; j++){
                new_backsubstituted_conv_upper_constraints[n* (prev_activation->input_size + 1) + j] = non_constant_coeffs[j];
            }

            n++;
        }    

        ZKgeneralTruncAny(current_layer->party, constant_terms.data(), constant_terms.data(), constant_terms.size(), FXPSCALE);

        n = 0;
        for(int i : neuron_list){
            new_backsubstituted_conv_upper_constraints[(n + 1) * (prev_activation->input_size + 1) - 1] = current_layer->lower_constraints[(i + 1) * (prev_activation->input_size + 1) - 1] +
                                                                                                constant_terms[n];

            n++;
        }


        T* accumulator = new T[neuron_list.size()];
        n = 0;
        for(int i : neuron_list){
            T* current_lower_constraints = new_backsubstituted_conv_upper_constraints + (n* (prev_conv->output_size + 1));

            ZKcmpPositive(
                current_layer->party,
                current_lower_constraints,
                ZERO_COMP_CONSTANT,
                coeff_sign,
                prev_conv->output_size
            );
            for(int j = 0; j < prev_conv->output_size; j++){
                coeff_sign[j + prev_conv->output_size] = FIELD_ONE + coeff_sign[j].negate();
            
                prev_layer_coeffs[j]                            = coeff_sign[j]                            * prev_conv->lower_bounds[j];
                prev_layer_coeffs[j + prev_conv->output_size] = coeff_sign[j + prev_conv->output_size] * prev_conv->upper_bounds[j];
            }
            
            accumulator[n] = inner_product_bundle(
                prev_conv->output_size,
                current_lower_constraints,
                prev_layer_coeffs,
                current_layer->party
            ) + inner_product_bundle(
                prev_conv->output_size,
                current_lower_constraints,
                prev_layer_coeffs + prev_conv->output_size,
                current_layer->party
            );

            n++;
        }

        ZKgeneralTruncAny(current_layer->party, accumulator, accumulator, neuron_list.size(), FXPSCALE);

        n = 0;
        for(int i : neuron_list){
            current_layer->lower_bounds[i] = accumulator[n] + new_backsubstituted_conv_upper_constraints[(n + 1) * (prev_conv->output_size + 1) - 1];

            n++;
        }



    } else {

        int n = 0;
        for(int i : neuron_list){
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

                T non_const_prev_constraint;
                T const_prev_constraint;
                if(greater_eq_zero<T>(j_th_lc, false)){
                    non_const_prev_constraint = prev_activation->upper_constraints[j_th_predecessor * 2 + 0];
                    const_prev_constraint = prev_activation->upper_constraints[j_th_predecessor * 2 + 1];
                } else {
                    non_const_prev_constraint = prev_activation->lower_constraints[j_th_predecessor * 2 + 0];
                    const_prev_constraint = prev_activation->lower_constraints[j_th_predecessor * 2 + 1];        
                }

                (*new_backsubstituted_conv_upper_constraints)[n].push_back(j_th_lc * non_const_prev_constraint);
                new_const_term = new_const_term + j_th_lc * const_prev_constraint;

                (*new_predecessors)[n].push_back(j_th_predecessor);

                j++;
            }

            (*new_backsubstituted_conv_upper_constraints)[n].push_back(new_const_term);
            
            assert(
                (*new_backsubstituted_conv_upper_constraints)[n].size() ==
                (*new_predecessors)[n].size() + 1
            );

            n++;
        }

        n = 0;
        for(int i : neuron_list){
            vector<T> current_upper_constraints = (*new_backsubstituted_conv_upper_constraints)[n];


            T accumulated = current_upper_constraints[prev_conv->output_size];

            int* pred = (*current_layer->predecessors)[i].data();
            int* end = pred + (*current_layer->predecessors)[i].size();

            int j = 0;
            for(; pred != end; pred++){

                T j_th_lc = current_upper_constraints[j];

                int j_th_predecessor = *pred;
                if(greater_eq_zero<T>((T) j_th_lc, true)){
                    accumulated = accumulated + j_th_lc * prev_conv->upper_bounds[j_th_predecessor];
                } else {
                    accumulated = accumulated + j_th_lc * prev_conv->lower_bounds[j_th_predecessor];
                }

                j++;
            }

            // current_layer->upper_bounds[i] = inner_product_emp(current_layer->max_coeffs, current_upper_constraints, prev_layer_bounds_to_mult);
            current_layer->upper_bounds[i] = accumulated;

            n++;
        }
    }
}



#endif