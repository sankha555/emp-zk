#ifndef __VERIFICATION_UTILS_H__
#define __VERIFICATION_UTILS_H__

#pragma once

#include "emp-tool/emp-tool.h"
#include "emp-zk/emp-zk.h"
#include "emp-zk/ai/verification/verification.h"
#include "emp-zk/ai/json.hpp"

#include <fstream>
#include <filesystem>
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using namespace emp;
using json = nlohmann::json;


enum SPEC_LABELS{
    LAYER_TYPE_INDEX = 0,
    INPUT_SIZE_INDEX = 1,
    OUTPUT_SIZE_INDEX = 2,
    MAX_COEFFS_INDEX = 3,

    INPUT_CHANNELS = 1,
    OUTPUT_CHANNELS = 2,
    IMAGE_H = 3,
    IMAGE_W = 4,
    KERNEL_H = 5,
    KERNEL_W = 6,
    STRIDE_H = 7,
    STRIDE_W = 8,
    PAD_H = 9,
    PAD_W = 10,
    CONV_MAX_COEFFS_INDEX = 11
};


template <typename T>
VerifiableFeedForwardNeuralNetwork<T>* create_model(int num_layers, int* layer_specs, int party){
    Layer<T>** layers = new Layer<T>*[num_layers];
    int specs = 0;

    for(int i = 0; i < num_layers; i++){
        switch (layer_specs[specs + LAYER_TYPE_INDEX]){
            case LAYER_TYPE::INPUT:
                layers[i] = new Input<T>(
                    layer_specs[specs + INPUT_SIZE_INDEX],
                    layer_specs[specs + OUTPUT_SIZE_INDEX],
                    layer_specs[specs + MAX_COEFFS_INDEX],
                    party
                );
                specs += 4;
                break;

            case LAYER_TYPE::AFFINE:
                layers[i] = new Affine<T>(
                    layer_specs[specs + INPUT_SIZE_INDEX],
                    layer_specs[specs + OUTPUT_SIZE_INDEX],
                    layer_specs[specs + MAX_COEFFS_INDEX],
                    party
                );
                specs += 4;

                break;

            case LAYER_TYPE::CONV2D:
                layers[i] = new Conv2D<T>(
                    layer_specs[specs + INPUT_CHANNELS],
                    layer_specs[specs + OUTPUT_CHANNELS],
                    layer_specs[specs + IMAGE_H],
                    layer_specs[specs + IMAGE_W],
                    layer_specs[specs + KERNEL_H],
                    layer_specs[specs + KERNEL_W],
                    layer_specs[specs + STRIDE_H],
                    layer_specs[specs + STRIDE_W],
                    layer_specs[specs + PAD_H],
                    layer_specs[specs + PAD_W],
                    layer_specs[specs + CONV_MAX_COEFFS_INDEX],
                    party
                );
                specs += 12;

                break;

            case LAYER_TYPE::RELU:
                layers[i] = new ReLU<T>(
                    layer_specs[specs + INPUT_SIZE_INDEX],
                    layer_specs[specs + OUTPUT_SIZE_INDEX],
                    layer_specs[specs + MAX_COEFFS_INDEX],
                    party
                );
                specs += 4;

                break;

            case LAYER_TYPE::OUTPUT:
                layers[i] = new Output<T>(
                    layer_specs[specs + INPUT_SIZE_INDEX],
                    layer_specs[specs + OUTPUT_SIZE_INDEX],
                    layer_specs[specs + MAX_COEFFS_INDEX],
                    party
                );
                specs += 4;

                break;

            default:
                error("Invalid Layer type!\n");
        }
    }
    VerifiableFeedForwardNeuralNetwork<T>* model = new VerifiableFeedForwardNeuralNetwork<T>(num_layers, layers, party);

    return model;
}

LAYER_TYPE stringToLayerType(const std::string& type) {
    if (type == "INPUT") return INPUT;
    if (type == "AFFINE") return AFFINE;
    if (type == "RELU") return RELU;
    if (type == "OUTPUT") return OUTPUT;
    if (type == "CONV2D") return CONV2D;
    throw std::runtime_error("Unknown layer type: " + type);
}

vector<int> read_exp_specs(
    const char* config_file_path,
    float* epsilon,
    string &INPUT_FILE_PATH,
    string &PARAMS_FILE_PATH,
    string &LOG_FILE_PATH,
    int* test_mode,
    vector<int>* test_examples,
    int worker_id
){

    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        std::cerr << "Failed to open" << config_file_path << std::endl;
        return {};
    }
    
    json config;
    file >> config;
    file.close();

    string model_name = config["model_name"];
    if(model_name.compare(0, 5, "mnist") == 0){
        CURR_DATASET = DATASETS::MNIST;
        if(model_name.find("conv") != std::string::npos){
            INPUT_FILE_PATH = "test/ai/data/inputs/mnist_conv_" + to_string(worker_id) + ".txt";
            cerr << INPUT_FILE_PATH << "\n";
        } else {
            INPUT_FILE_PATH = "test/ai/data/inputs/mnist_test_" + to_string(worker_id) + ".txt";
        }

    } else if(model_name.compare(0, 5, "cifar") == 0) {
        CURR_DATASET = DATASETS::CIFAR10;
        INPUT_FILE_PATH = "test/ai/data/inputs/cifar10_test_nonconv_" + to_string(worker_id) + ".txt";

    } else {
        CURR_DATASET = DATASETS::TOY;
        INPUT_FILE_PATH = "test/ai/data/inputs/toy" + to_string(worker_id) + ".txt";
    }

    int num_neurons = config["num_neurons"];
    std::vector<int> layer_specs;

    int num_layers = 0;
    for (const auto& layer : config["layers"]) {
        std::string type = layer[0];
        layer_specs.push_back(stringToLayerType(type));
        
        if(!strcmp(type.c_str(), "CONV2D")){
            int num_specs_in_conv = 12;
            for(int i = 1; i < num_specs_in_conv; i++){
                layer_specs.push_back(layer[i]);
            }

        } else {
            // Handle input_size
            if (layer[1].is_string() && layer[1] == "num_neurons") {
                layer_specs.push_back(num_neurons);
            } else {
                layer_specs.push_back(layer[1]);
            }
            
            // Handle output_size
            if (layer[2].is_string() && layer[2] == "num_neurons") {
                layer_specs.push_back(num_neurons);
            } else {
                layer_specs.push_back(layer[2]);
            }
            
            layer_specs.push_back(layer[3]);
        }
        
        num_layers++;
    }

    *epsilon = (float) config["epsilon"];


    PARAMS_FILE_PATH = "test/ai/data/parameters/" + model_name + "_" + to_string(worker_id) + ".txt";


    string folder = "test/ai/data/logs/" + model_name;
    struct stat st;
    if (stat(folder.c_str(), &st)) {
        mkdir(folder.c_str(), 0755) == 0;
    }
    LOG_FILE_PATH = folder + "/" + model_name;
    
    *test_mode = (int) config["do_float"];

    FXPSCALE = (int) config["FXPSCALE"];
    INPUT_MIN = (float) config["input_min"];
    INPUT_MAX = (float) config["input_max"];

    layer_specs.push_back(num_layers);

    vector<int> examples = config["examples"];
    if(examples.size() > 0){
        for(int e : examples){
            test_examples->push_back(e);
        }
    }

    if(config.contains("bs_mode")){
        BS_MODE = config["bs_mode"];
        if(BS_MODE == 1){
            DO_DP_BS = true;
        } else {
            DO_DP_BS = false;
        }
    }

    if(config.contains("bs_waiver_thresholds")){
        BS_WAIVER_THRESHOLDS = config["bs_waiver_thresholds"];
    }

    if(config.contains("bs_waiver_thresholds2")){
        BS_WAIVER_THRESHOLDS2 = config["bs_waiver_thresholds2"];
    }

    for(auto k : BS_WAIVER_THRESHOLDS){
        if(BS_WAIVER_THRESHOLDS[k.first] != -1){
            FORCE_STOP_PRINTING = true;
        }
    }
    for(auto k : BS_WAIVER_THRESHOLDS2){
        if(BS_WAIVER_THRESHOLDS2[k.first] != -1){
            FORCE_STOP_PRINTING = true;
        }
    }


    HEURISTICS_FILE_PATH += model_name;
    if (stat(HEURISTICS_FILE_PATH.c_str(), &st)) {
        mkdir(HEURISTICS_FILE_PATH.c_str(), 0755) == 0;
    }
    std::ofstream outfile(HEURISTICS_FILE_PATH + "/skip_map1.json");
    outfile.close();

    std::ofstream outfile2(HEURISTICS_FILE_PATH + "/skip_map2.json");
    outfile2.close();


    return layer_specs;
}


template <typename T>
std::pair<bool, bool> verify_example(VerifiableFeedForwardNeuralNetwork<T>* model, const char* input_file, int input_offset, float epsilon, int example_num = 1){
    model->reset();
    model->load_input(input_file, input_offset, epsilon);
    
    pair<bool, bool> result;
    if constexpr (std::is_same<IntFp, T>::value){
        model->mode = 1;
        result = model->forward(example_num, true, true); 
    } else {
        // model->mode = 0;
        // model->forward(example_num, true, true);
        model->mode = 1;
        result = model->forward(example_num, true, true);
    }

    return result;
}   


void preprocess_input(int n, float* inputs, float mean, float sdev){
    for(int i = 0; i < n; i++){
        inputs[i] = (inputs[i] - mean)/sdev;
    }
}


void preprocess_input(int dims, int n_per_dim){
    ;
}





#endif