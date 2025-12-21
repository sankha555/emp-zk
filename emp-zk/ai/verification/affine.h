#ifndef __AFFINE_H__
#define __AFFINE_H__

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
class Affine : public Layer<T> {
    public:

    ParametersVerification<T>* param;
    
    int* pred_neuron_ids;

    set<int>* skippable_neurons;
    int neurons_saved = 0;
    
    Affine(int input_size, int output_size, int max_coeffs = -1, int party = PUBLIC) : Layer<T>(input_size, output_size, max_coeffs, party){
        if(max_coeffs == -1){
            max_coeffs = this->input_size+1;
        }
        this->max_coeffs = max_coeffs;

        this->input = new T[input_size+1];  // +1 for bias
        this->output = new T[output_size];
        this->param = new ParametersVerification<T>(output_size, input_size, party);
        this->type = LAYER_TYPE::AFFINE;

        this->lower_bounds = new T[output_size];
        this->upper_bounds = new T[output_size];
        
        this->lower_constraints = new T[output_size*this->max_coeffs];
        this->upper_constraints = new T[output_size*this->max_coeffs];

        this->backsubstituted_lower_constraints= new T[output_size*this->max_coeffs];
        this->backsubstituted_upper_constraints= new T[output_size*this->max_coeffs];
    
        this->pred_neuron_ids = new int[this->output_size * this->max_coeffs];
        for(int i = 0; i < this->output_size; i++){
            for(int j = 0; j < this->max_coeffs - 1; j++){
                this->pred_neuron_ids[i*this->max_coeffs + j] = j;
            }
            this->pred_neuron_ids[(i+1)*this->max_coeffs - 1] = -1;
        }

        this->lower_diff = new float[this->output_size]{0};
        this->upper_diff = new float[this->output_size]{0};
    }

    void forward(Layer<T>* input_layer, Layer<T>* prev_layer, bool do_inference = true){
        this->prev_layer = prev_layer;

        // clone the input
        for(int i = 0; i < this->input_size; i++){
            this->input[i] = prev_layer->output[i];
        }

        // for bias
        if constexpr (std::is_same<T, IntFp>::value) {
            this->input[this->input_size] = IntFp(1ULL << FXPSCALE);
        } else if constexpr (std::is_same<T, float>::value) {
            this->input[this->input_size] = float(1);
        }

        assert((this->param != NULL && this->param->param_matrix != NULL) && "Parameters not initialized for AFFINE layer");
        
        if(!ONLY_INFERENCE){
            compute_lower_constraints();
            compute_lower_bounds();

            compute_upper_constraints();
            compute_upper_bounds();

            if constexpr (std::is_same<float, T>::value){
                for(int i = 0; i < this->output_size; i++){
                    this->lower_diff[i] = this->lower_bounds[i];
                    this->upper_diff[i] = this->upper_bounds[i];
                }
            }

            backsubstitute(input_layer);
            
            if constexpr (std::is_same<float, T>::value){
                this->analyse_and_unset_bs_bounds();
            }       
        }

        if(do_inference){
            if constexpr (std::is_same<IntFp, T>::value && SECURE){
                for(int i = 0; i < this->output_size; i++){
                    this->output[i] = inner_product_bundle(this->input_size + 1, this->param->param_matrix + i*(this->input_size+1), this->input, this->party);
                }

                ZKgeneralTruncAny(this->party, this->output, this->output, this->output_size, FXPSCALE);

            } else {
            
                affine_layer(this->output_size, this->input_size, this->param->param_matrix, this->input, this->output);
                if constexpr (std::is_same<T, IntFp>::value){
                    normalize(this->output_size, this->output, this->output);
                }
            }
        }
    }


    void analyse_and_unset_bs_bounds(){
        this->skippable_neurons = new set<int>();
        if(!BS_WAIVER_THRESHOLDS.count(to_string(this->layer_num))){
            return;
        }

        float threshold = BS_WAIVER_THRESHOLDS[to_string(this->layer_num)];
        // cerr << threshold << "\n";

        for(int i = 0; i < this->output_size; i++){
            this->lower_diff[i] = (this->lower_bounds[i] - this->lower_diff[i]);
            this->upper_diff[i] = (this->upper_bounds[i] - this->upper_diff[i]);
        }

        vector<pair<pair<float, float>, int>> neuron_info;

        for(int i = 0; i < this->output_size; i++){
            neuron_info.push_back(
                {{abs(this->lower_diff[i]), abs(this->upper_diff[i])}, i}
            );
        }

        sort(neuron_info.begin(), neuron_info.end());

        int num_nonbs_neurons = ((this->output_size * 1.0) * BS_WAIVER_FRACTION);
        // cerr << num_nonbs_neurons << "\n";
        // for(int k = 0; k < num_nonbs_neurons; k++){
        //     int nid = neuron_info[k].second;
        //     this->lower_bounds[nid] = this->lower_diff[nid] - this->lower_bounds[nid]; 
        //     this->upper_bounds[nid] = this->upper_diff[nid] - this->upper_bounds[nid]; 
        // }

        for(int k = 0; k < this->output_size; k++){
            int nid = neuron_info[k].second;
            if(neuron_info[k].first.first < threshold){
                this->lower_bounds[nid] = this->lower_diff[nid] - this->lower_bounds[nid]; 
                this->upper_bounds[nid] = this->upper_diff[nid] - this->upper_bounds[nid];

                this->skippable_neurons->insert(nid);
                neurons_saved++;
            } 
        }
    }


    void compute_lower_bounds(){
        T* prev_lbs = ((Layer<T>*) this->prev_layer)->lower_bounds;
        T* prev_ubs = ((Layer<T>*) this->prev_layer)->upper_bounds;

        if constexpr (std::is_same<IntFp, T>::value && SECURE){

            T* prev_bounds = new T[2*(this->max_coeffs - 1)];       // lb_1 lb_2 ... lb_m   ub_1 ub_2 ... ub_m  
            for(int j = 0; j < this->max_coeffs - 1; j++){
                prev_bounds[j] = prev_lbs[j];
                prev_bounds[j + this->max_coeffs - 1] = prev_ubs[j];
            }

            auto start = clock_start();

            IntFp* coeff_sign = new IntFp[2 * (this->max_coeffs - 1)]; 
            T* copied_lc = new T[2*(this->max_coeffs - 1)];

            for(int i = 0; i < this->output_size; i++){
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

            double tt = time_from(start);


            // restore the fixed-point scale
            ZKgeneralTruncAny(this->party, this->lower_bounds, this->lower_bounds, this->output_size, FXPSCALE);

            for(int i = 0; i < this->output_size; i++){
                this->lower_bounds[i] = this->lower_bounds[i] + this->lower_constraints[(i+1)*this->max_coeffs - 1];    // adding the constant bias term
            }

            delete[] prev_bounds;

        } else {
            T* prev_bounds = new T[this->max_coeffs];

            for(int i = 0; i <  this->output_size; i++){
                for(int j = 0; j < this->max_coeffs-1; j++){
                    if(greater_eq_zero<T>(this->lower_constraints[i*this->max_coeffs + j], false)){
                        prev_bounds[j] = prev_lbs[j];
                    } else {
                        prev_bounds[j] = prev_ubs[j];
                    }
                }
                prev_bounds[this->max_coeffs-1] = constant<T>(1);
                
                this->lower_bounds[i] = inner_product_emp(this->max_coeffs, this->lower_constraints + i*(this->max_coeffs),  prev_bounds);
            }

            delete[] prev_bounds;
        }

    }

    void compute_upper_bounds(){
        T* prev_lbs = ((Layer<T>*) this->prev_layer)->lower_bounds;
        T* prev_ubs = ((Layer<T>*) this->prev_layer)->upper_bounds;

        if constexpr (std::is_same<IntFp, T>::value && SECURE){

            T* prev_bounds = new T[2*(this->max_coeffs - 1)];       // ub_1 ub_2 ... ub_m   lb_1 lb_2 ... lb_m  
            for(int j = 0; j < this->max_coeffs - 1; j++){
                prev_bounds[j] = prev_ubs[j];
                prev_bounds[j + this->max_coeffs - 1] = prev_lbs[j];
            }

            IntFp* coeff_sign = new IntFp[2 * (this->max_coeffs - 1)];   
            T* copied_uc = new T[2*(this->max_coeffs - 1)];

            for(int i = 0; i < this->output_size; i++){
            
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
            T* prev_bounds = new T[this->max_coeffs];

            for(int i = 0; i < this->output_size; i++){
                for(int j = 0; j < this->max_coeffs-1; j++){
                    if(greater_eq_zero<T>(this->upper_constraints[i*this->max_coeffs + j], false)){
                        prev_bounds[j] = prev_ubs[j];
                    } else {
                        prev_bounds[j] = prev_lbs[j];
                    }
                }
                prev_bounds[this->max_coeffs-1] = constant<T>(1);

                this->upper_bounds[i] = inner_product_emp(this->max_coeffs, this->upper_constraints + i*(this->max_coeffs),  prev_bounds);
            
            }

            delete[] prev_bounds;
        }

    }


    void compute_lower_constraints(){
        // l_i ≤ x_i ≤ u_i for input layer
        for(int i = 0; i <  this->output_size; i++){
            for(int j = 0; j < this->max_coeffs; j++){
                this->lower_constraints[i*this->max_coeffs + j] = (this->param->param_matrix[i*this->max_coeffs + j]);
            }
        }
    }

    void compute_upper_constraints(){
        // l_i ≤ x_i ≤ u_i for input layer
        for(int i = 0; i <  this->output_size; i++){
            for(int j = 0; j < this->max_coeffs; j++){
                this->upper_constraints[i*this->max_coeffs + j] = (this->param->param_matrix[i*this->max_coeffs + j]);
            }
        }
    }


    void backsubstitute(Layer<T>* input_layer){
        for(int i = 0; i < this->output_size * this->max_coeffs; i++){
            this->backsubstituted_lower_constraints[i] = (this->lower_constraints[i]);
            this->backsubstituted_upper_constraints[i] = (this->upper_constraints[i]);
        }

        if(DO_DP_BS){
            
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

            if(this->layer_num > 2){
                backsubstitute_lc_using_prev_affine(this, this->prev_layer, this->prev_layer->prev_layer, input_layer);
                backsubstitute_uc_using_prev_affine(this, this->prev_layer, this->prev_layer->prev_layer, input_layer);
            }

        }
    }


    void reset(){
        delete[] this->input;
        delete[] this->output;
        delete[] this->lower_bounds;
        delete[] this->upper_bounds;
        delete[] this->lower_constraints;
        delete[] this->upper_constraints;

        this->max_coeffs = this->input_size + 1;

        this->input = new T[this->input_size+1];  // +1 for bias
        this->output = new T[this->output_size];

        this->lower_bounds = new T[this->output_size];
        this->upper_bounds = new T[this->output_size];
        
        this->lower_constraints = new T[this->output_size*this->max_coeffs];
        this->upper_constraints = new T[this->output_size*this->max_coeffs];

        this->is_backsubstituted = false;

        this->backsubstituted_lower_constraints= new T[this->output_size*this->max_coeffs];
        this->backsubstituted_upper_constraints= new T[this->output_size*this->max_coeffs];
        this->neurons_saved = 0;
    }

    void describe(bool print_parameters = true, bool print_expressions = false){
        if(this->party == BOB){
            return;
        }
        cout << "Type: " << get_layer_type(this->type) << "[" << this->layer_num << "]" << "\n";
        if(print_parameters){
            cout << "Parameters:\n";
            param->print_parameters();
        }

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
        for(int i = 0; i < this->output_size; i++){
            if constexpr (std::is_same<T, IntFp>::value){
                cout << format_EMP_IntFp(this->output[i], 1) << " ";
            } else if constexpr (std::is_same<T, float>::value) {
                cout << this->output[i] << " ";
            }
        }
        cout << "\n";

        if(!ONLY_INFERENCE){
            cout << "Lower Bounds:\n";
            for(int i = 0; i < this->output_size; i++){
                if constexpr (std::is_same<T, IntFp>::value){
                    cout << format_EMP_IntFp(this->lower_bounds[i], 1) << " ";
                } else if constexpr (std::is_same<T, float>::value) {
                    cout << this->lower_bounds[i] << " ";
                }
            }
            cout << "\n";
            
            cout << "Upper Bounds:\n";
            for(int i = 0; i < this->output_size; i++){
                if constexpr (std::is_same<T, IntFp>::value){
                    cout << format_EMP_IntFp(this->upper_bounds[i], 1) << " ";
                } else if constexpr (std::is_same<T, float>::value) {
                    cout << this->upper_bounds[i] << " ";
                }
            }

            if constexpr (std::is_same<float, T>::value){
                cout << "\n\n";
                cout << "Lower Diff:\n";
                for(int i = 0; i < this->output_size; i++){
                    cout << this->lower_diff[i] << " ";
                }

                cout << "\nUpper Diff:\n";
                for(int i = 0; i < this->output_size; i++){
                    cout << this->upper_diff[i] << " ";
                }
                cout << "\n";
            }
            
        }

        if (print_expressions && !ONLY_INFERENCE){
            cout << "\n";

            cout << "Lower Expression:\n";
            for(int i = 0; i < this->output_size; i++){
                cout << "N" << i+1 << ": ";
                for(int k = 0; k < NUM_FEATURES[CURR_DATASET]+1; k++){
                    if constexpr (std::is_same<T, IntFp>::value){
                        cout << format_EMP_IntFp(this->backsubstituted_lower_constraints[i*(NUM_FEATURES[CURR_DATASET]+1) + k], 1) << " ";
                    } else if constexpr (std::is_same<T, float>::value) {
                        cout << this->backsubstituted_lower_constraints[i*(NUM_FEATURES[CURR_DATASET]+1) + k] << " ";
                    }
                }
                
                cout << "\n";
            }
            cout << "\n";

            cout << "Upper Expression:\n";
            for(int i = 0; i < this->output_size; i++){
                cout << "N" << i+1 << ": ";
                for(int k = 0; k < NUM_FEATURES[CURR_DATASET]+1; k++){
                    if constexpr (std::is_same<T, IntFp>::value){
                        cout << format_EMP_IntFp(this->backsubstituted_upper_constraints[i*(NUM_FEATURES[CURR_DATASET]+1) + k], 1) << " ";
                    } else if constexpr (std::is_same<T, float>::value) {
                        cout << this->backsubstituted_upper_constraints[i*(NUM_FEATURES[CURR_DATASET]+1) + k] << " ";
                    }
                }
                
                cout << "\n";
            }
        }

        cout << "\n\n";
    }


    void cleartext_compute_lower_constraints(){
        // l_i ≤ x_i ≤ u_i for input layer
        for(int i = 0; i <  this->output_size; i++){
            for(int j = 0; j < this->max_coeffs; j++){
                this->lower_constraints[i*this->max_coeffs + j] = (this->param->param_matrix[i*this->max_coeffs + j]);
            }
        }
    }

    void cleartext_compute_upper_constraints(){
        // l_i ≤ x_i ≤ u_i for input layer
        for(int i = 0; i <  this->output_size; i++){
            for(int j = 0; j < this->max_coeffs; j++){
                this->upper_constraints[i*this->max_coeffs + j] = (this->param->param_matrix[i*this->max_coeffs + j]);
            }
        }
    }

    void cleartext_compute_lower_bounds(){
        T* prev_lbs = ((Layer<T>*) this->prev_layer)->lower_bounds;
        T* prev_ubs = ((Layer<T>*) this->prev_layer)->upper_bounds;

        for(int i = 0; i <  this->output_size; i++){
            T* prev_bounds = new T[this->max_coeffs];
            for(int j = 0; j < this->max_coeffs-1; j++){
                if(greater_eq_zero<T>(this->lower_constraints[i*this->max_coeffs + j], false)){
                    prev_bounds[j] = prev_lbs[j];
                } else {
                    prev_bounds[j] = prev_ubs[j];
                }
            }
            prev_bounds[this->max_coeffs-1] = constant<T>(1);
            
            this->lower_bounds[i] = inner_product_emp(this->max_coeffs, this->lower_constraints + i*(this->max_coeffs),  prev_bounds);
        }

        if(std::is_same<IntFp, T>::value){
            normalize(this->output_size, (IntFp*)this->lower_bounds, (IntFp*)this->lower_bounds);
        }
    }

    void cleartext_compute_upper_bounds(){
        T* prev_lbs = ((Layer<T>*) this->prev_layer)->lower_bounds;
        T* prev_ubs = ((Layer<T>*) this->prev_layer)->upper_bounds;

        for(int i = 0; i <  this->output_size; i++){
            T* prev_bounds = new T[this->max_coeffs];
            for(int j = 0; j < this->max_coeffs-1; j++){
                if(greater_eq_zero<T>(this->upper_constraints[i*this->max_coeffs + j], false)){
                    prev_bounds[j] = prev_ubs[j];
                } else {
                    prev_bounds[j] = prev_lbs[j];
                }
            }
            prev_bounds[this->max_coeffs-1] = constant<T>(1);
            
            this->upper_bounds[i] = inner_product_emp(this->max_coeffs, this->upper_constraints + i*(this->max_coeffs),  prev_bounds);
        }

        if(std::is_same<IntFp, T>::value){
            normalize(this->output_size, (IntFp*)this->upper_bounds, (IntFp*)this->upper_bounds);
        }
    }

};


#endif