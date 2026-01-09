#ifndef __VERIFIABLE_FEEDFORWARD_H__
#define __VERIFIABLE_FEEDFORWARD_H__

#pragma once

#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "emp-zk/ai/verification/verification.h"
#include "emp-zk/ai/utils.h"
#include <iostream>

using namespace emp;
using namespace std;

typedef struct stats {
    long min_savings;
    long min_example;
    long max_savings;
    long max_example;
    long avg_savings;
    long total_examples;
    long total_comps;

    void new_example(long i, long savings, long total_computations){
        if(savings < min_savings){
            min_savings = savings;
            min_example = i;
        }

        if(savings > max_savings){
            max_savings = savings;
            max_example = i;
        }

        avg_savings = avg_savings * total_examples + savings;
        total_examples++;
        avg_savings = avg_savings / total_examples;

        total_comps = total_computations;
    }

    void print_stats(){
        cerr << "Avg. savings = " << avg_savings << " [" << (avg_savings * 100)/total_comps << " %] " << " (Total " << total_examples << " examples)\n"; 
        cerr << "Min. savings = " << min_savings << " [" << (min_savings * 100)/total_comps << " %] " << " (Example " << min_example << ")\n";
        cerr << "Max. savings = " << max_savings << " [" << (max_savings * 100)/total_comps << " %] " << " (Example " << max_example << ")\n"; 
    }
} stats;

template <typename T>
class VerifiableFeedForwardNeuralNetwork {
    public:
    int party = PUBLIC;
    int num_layers;
    Layer<T>** layers;
    
    int num_inputs;

    stats* savings1_stats;
    stats* savings2_stats;
    stats* savings_stats;

    int mode = 0;  // 0 = analyse, 1 = save
    vector<pair<float, pair<int, int>>>* global_neuron_info;
    map<int, set<int>*> skip_map;
    map<int, set<int>*> skip_map2;

    VerifiableFeedForwardNeuralNetwork(int num_layers, Layer<T>** layers, int party = PUBLIC){
        this->num_layers = num_layers;
        this->layers = layers;
        this->party = party;

        this->savings1_stats = new stats();
        this->savings2_stats = new stats();
        this->savings_stats = new stats();
        // validate_layers();

        this->global_neuron_info = new vector<pair<float, pair<int, int>>>();
    }

    void validate_layers(){
        assert(num_layers > 0 && "Network must have at least 1 layer\n");

        assert(layers[0]->type == INPUT && "The first layer of the network must be an INPUT layer\n");

        for(int i = 1; i < num_layers-1; i += 2){
            assert(
                (layers[i]->type == AFFINE && layers[i+1]->type == RELU) && 
                "Intermediate layers must alternate with AFFINE and RELU\n"
            );
        }

        assert(layers[num_layers-1]->type == OUTPUT &&  "The last layer of the network must be an OUTPUT layer\n");


        // validate layer sizes
        for(int i = 1; i < num_layers; i++){
            Layer<T>* prev_layer = layers[i-1];
            assert(
                prev_layer->output_size == layers[i]->input_size && 
                "Output size of previous layer must match input size of current layer"
            );

            prev_layer = layers[i];
        }
    }

    void load_weights_and_biases(const char* param_file_path){
        int layer_offset = 0;
        for(int k = 0; k < num_layers; k++){
            Layer<T>* curr_layer = layers[k];
            if(curr_layer->type == AFFINE){
                ((Affine<T>*) curr_layer)->param->read_weights_and_biases(param_file_path, layer_offset);
                layer_offset += ((Affine<T>*) curr_layer)->param->num_parameters();
            } else if(curr_layer->type == CONV2D){
                ((Conv2D<T>*) curr_layer)->kernel->read_filters(param_file_path, layer_offset);
                layer_offset += ((Conv2D<T>*) curr_layer)->kernel->num_parameters();
            }
        }
    }

    void load_input(const char* input_file_path, int input_offset = 0, float epsilon = 0.1){
        assert(layers != NULL && "Layers not yet initialized, cannot load inputs!");
        assert(layers[0]->type == INPUT && "The first layer should be an INPUT layer, cannot load inputs!");

        Layer<T>* input_layer = layers[0];

        // read ground truth
        float gt;
        read_next_elements(1, &gt, input_offset, input_file_path);
        ((Output<T>*) layers[num_layers-1])->ground_truth = int(gt);
        input_offset++;
        
        
        // read inputs
        float* raw_inputs = new float[input_layer->input_size];
        read_next_elements(input_layer->input_size, raw_inputs, input_offset, input_file_path);

        float* raw_lb = new float[input_layer->input_size];
        for(int i = 0; i < input_layer->input_size; i++){
            raw_lb[i] = raw_inputs[i] - epsilon;
            if(raw_lb[i] < INPUT_MIN){
                raw_lb[i] = INPUT_MIN;
            }

            if(raw_lb[i] > INPUT_MAX){
                raw_lb[i] = INPUT_MAX;
            }
        }

        float* raw_ub = new float[input_layer->input_size];
        for(int i = 0; i < input_layer->input_size; i++){
            raw_ub[i] = raw_inputs[i] + epsilon;
            if(raw_ub[i] < INPUT_MIN){
                raw_ub[i] = INPUT_MIN;
            }

            if(raw_ub[i] > INPUT_MAX){
                raw_ub[i] = INPUT_MAX;
            }
        }
            
        if constexpr (std::is_same<T, IntFp>::value){
            
            input_layer->input = convert_reals_to_field_rep(input_layer->input_size, raw_inputs, PUBLIC);
            input_layer->output = convert_reals_to_field_rep(input_layer->output_size, raw_inputs, PUBLIC);
            input_layer->lower_bounds = convert_reals_to_field_rep(input_layer->input_size, raw_lb, PUBLIC);
            input_layer->upper_bounds = convert_reals_to_field_rep(input_layer->input_size, raw_ub, PUBLIC);

        } else if constexpr (std::is_same<T, float>::value){
            for(int i = 0; i < input_layer->input_size; i++){
                input_layer->input[i] = raw_inputs[i];
                input_layer->output[i] = raw_inputs[i];
                input_layer->lower_bounds[i] = raw_lb[i];
                input_layer->upper_bounds[i] = raw_ub[i];
            }
        }

        this->num_inputs = input_layer->input_size;


        delete[] raw_inputs;
        delete[] raw_lb;
        delete[] raw_ub;
    }
    
    void reset(){
        for(int i = 0; i < this->num_layers; i++){
            ((Layer<T>*) this->layers[i])->reset();
        }
        if(std::is_same<T, float>::value){
            this->skip_map = {};
            this->skip_map2 = {};
        }
        this->global_neuron_info = new vector<pair<float, pair<int, int>>>();
    }


    std::pair<bool, bool> forward(int example_num = 1, bool do_backsubstitution = false, bool do_inference = true){
        Layer<T>* prev_layer = nullptr;
        Layer<T>* input_layer = layers[0];

        long savings1, savings2, savings, total_computations;
        savings1 = savings2 = savings = total_computations = 0;
        for(int i = 0; i < num_layers; i++){
            layers[i]->layer_num = i+1;

            bool use_bs_heuristic = false;
            if(this->mode == 1){
                use_bs_heuristic = true;

                if(layers[i]->type == AFFINE || layers[i]->type == CONV2D){
                    if (this->skip_map.count(layers[i]->layer_num)){
                        ((Layer<T>*) layers[i])->skippable_neurons = this->skip_map[layers[i]->layer_num];
                    }

                    if (this->skip_map2.count(layers[i]->layer_num)){
                        ((Layer<T>*) layers[i])->skippable_neurons2 = this->skip_map2[layers[i]->layer_num];
                    }
                }
            }

            layers[i]->forward(input_layer, prev_layer, do_inference, use_bs_heuristic);
            layers[i]->sanity_check();

            if (this->mode == 0){
                this->aggregate_neuron_info(layers[i]);
            } else {
                this->aggregate_savings(layers[i], &savings1, &savings2, &savings, &total_computations);
            }

            prev_layer = layers[i];
        }


        bool verification_result, classification_result;
        verification_result = ((Output<T>*) layers[this->num_layers-1])->verified;
        classification_result = ((Output<T>*) layers[this->num_layers-1])->correctly_classified;

        if(this->mode == 0 && classification_result == true){
            this->select_neurons_for_heuristics(example_num);
        } else if(this->mode == 1 && classification_result == true){
            this->savings1_stats->new_example(example_num, savings1, total_computations);
            this->savings2_stats->new_example(example_num, savings2, total_computations);
            this->savings_stats->new_example(example_num, savings, total_computations);
        }

        return {classification_result, verification_result};
    }

    void aggregate_neuron_info(Layer<T>* layer){
        if constexpr (std::is_same<T, float>::value){
            if (layer->neuron_info->size() > 0){
                for(int i = 0; i < layer->neuron_info->size(); i++){
                    this->global_neuron_info->push_back((*layer->neuron_info)[i]);
                }
            }
        }
    }

    void select_neurons_for_heuristics(int example_num){
        if constexpr (std::is_same<T, float>::value){
            this->skip_map = {};
            this->skip_map2 = {};

            sort(this->global_neuron_info->begin(), this->global_neuron_info->end());
        
            int n = this->global_neuron_info->size();
            float threshold_fraction1 = THRESHOLD_FRACTION1, threshold_fraction2 = THRESHOLD_FRACTION2;
            
            int i = 0;
            for(auto info : *this->global_neuron_info){
                int layer_num = info.second.first;
                int nid = info.second.second;

                if(i < threshold_fraction1 * n){
                    if(!this->skip_map.count(layer_num)){
                        this->skip_map[layer_num] = new set<int>();
                    }
                    this->skip_map[layer_num]->insert(nid);
                } else if (i < threshold_fraction2 * n){
                    if(!this->skip_map2.count(layer_num)){
                        this->skip_map2[layer_num] = new set<int>();
                    }
                    this->skip_map2[layer_num]->insert(nid);
                } else {
                    break;
                }

                i++;
            }


            string filepath = HEURISTICS_FILE_PATH + "/skip_map1.json";
            write_to_json_file(example_num, filepath.c_str(), this->skip_map);

            filepath = HEURISTICS_FILE_PATH + "/skip_map2.json";
            write_to_json_file(example_num, filepath.c_str(), this->skip_map2);
        }
    }

    void aggregate_savings(Layer<T>* layer, long* savings1, long* savings2, long* savings, long* total_computations){
        if constexpr (std::is_same<T, float>::value){
            if (layer->type == AFFINE || layer->type == CONV2D){
                vector<long> sv = get_raw_savings((Layer<T>*) layer);
                *savings1 += sv[0];
                *savings2 += sv[1];
                *savings += sv[0] + sv[1];
                *total_computations += sv[2];
            }
        }
    }

    void describe(bool print_parameters = true, bool print_expressions = false){
        if(this->party == BOB){
            return;
        }
        
        for(int i = 0; i < num_layers; i++){
            cout << "LAYER " << (i+1) << "\n";
            layers[i]->describe(print_parameters, print_expressions);
        }
    }
};

template <typename T>
void profiling(Layer<T>* layer){
    // if(layer->type != AFFINE){
    //     return;
    // }
    cout << "==============================================================================\n";
    cout << "Layer " << layer->layer_num << ":: " << get_layer_type(layer->type) << "\n";
    cout << "Time for FP: " << layer->time_for_fp/1e6 << " seconds\n";
    cout << "Time for BS: " << layer->time_for_bs/1e6 << " seconds\n";
    cout << "|\n";
    cout << "-----> Bound Computation: " << layer->time_for_bs_in_bounds/1e6 << " seconds\n|\n";
    cout << "-----> Update using ReLU: " << layer->time_for_bs_using_act/1e6 << " seconds\n|\n";
    cout << "-----> Update using Affn: " << layer->time_for_bs_using_affine/1e6 << " seconds\n";

    cout << "==============================================================================\n\n";
}

#endif