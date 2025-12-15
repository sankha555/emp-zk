#include "emp-tool/emp-tool.h"
#include <emp-zk/emp-zk.h>
#include <emp-zk/ai/ai.h>

#include <iostream>
#include <fstream>

using namespace emp;
using namespace std;

int port, party;
const int threads = 1;
int base_example = 0;
int num_examples = 0;
float epsilon;
string model_name;
bool print_to_stdout = false;

vector<int> test_examples;

string CONFIG_PATH_ROOT = "test/ai/data/configs/";
string INPUTS_PATH = "";
string PARAMETERS_PATH = "";
string LOGS_PATH = "";


int layer_specs[] = {};


void float_inference(BoolIO<NetIO> *ios[threads], int* layer_specs, int num_layers){
  auto start = clock_start();

  int num_examples_classified = 0;

  // float 
  std::ofstream file(std::string(LOGS_PATH) + "_float_worker_" + std::to_string(port - 10000 + 1) + ".txt");
  std::streambuf* original_buf = std::cout.rdbuf(file.rdbuf());

  VerifiableFeedForwardNeuralNetwork<float>* model_float = create_model<float>(num_layers, layer_specs, party);
  model_float->load_weights_and_biases(PARAMETERS_PATH.c_str());

  for(int i = base_example; i < base_example + num_examples; i++){
    auto result = verify_example<float>(model_float, INPUTS_PATH.c_str(), i*(NUM_FEATURES[CURR_DATASET]+1), epsilon);
    bool classified = result.first;

    num_examples_classified += (int) classified;

    if(party == ALICE) model_float->describe(false, false);
  }
  cout << "Correctly classified = " << num_examples_classified << "\n";

  double tt = time_from(start);
  cout << "\nAvg. time to classify: " << (tt/1000000)/num_examples << " s\n";

  std::cout.rdbuf(original_buf);  

  cout << "Float inference completed\n";
  cout << "Correctly classified = " << num_examples_classified << "\n";
  cout << "\n";
}


void field_inference(BoolIO<NetIO> *ios[threads], int* layer_specs, int num_layers){
  // field

  setup_plain_prot(false, "");
  setup_zk_arith<BoolIO<NetIO>>(ios, threads, party);

  init_verification();
  startComputation(party);

  double tt;

  std::ofstream field_file(std::string(LOGS_PATH) + "_worker_" + std::to_string(port - 10000 + 1) + ".txt");
  std::streambuf* original_buf2;
  if(!print_to_stdout) original_buf2 = std::cout.rdbuf(field_file.rdbuf());

  VerifiableFeedForwardNeuralNetwork<IntFp>* model_field = create_model<IntFp>(num_layers, layer_specs, party);
  model_field->load_weights_and_biases(PARAMETERS_PATH.c_str());

  int num_examples_classified = 0;

  auto start = clock_start();
  for(int i = base_example; i < base_example + num_examples; i++){
    auto result = verify_example<IntFp>(model_field, INPUTS_PATH.c_str(), i*(NUM_FEATURES[CURR_DATASET]+1), epsilon);
    bool classified = result.first;

    num_examples_classified += (int) classified;

    tt = time_from(start);

    if(print_to_stdout){
      cout << "Correctly classified = " << num_examples_classified << "\n";
    } else {
      cout << (i+1) << ": " << (classified ? "YES" : "NO") << "\n";
    }
  }
  cout << "\n";

  cout << "Correctly classified = " << num_examples_classified << "\n";

  bool cheated = finalize_zk_arith<BoolIO<NetIO>>();
  if(party == BOB){
    cout << "\n" << (cheated ? "\033[31mVerfication failed!" : "\033[32mVerfication successful!") << "\033[0m\n";
  }

  tt = time_from(start);
  cout << "\nAvg. time to verify: " << (tt/1000000)/num_examples << " s\n";
  cout << "Communication: " << ios[0]->counter/(1024.0 * 1024.0) << " MB\n";

  if(!print_to_stdout) std::cout.rdbuf(original_buf2);
}


void test_inference(BoolIO<NetIO> *ios[threads], int party) {
  // omp_set_num_threads(NUM_THREADS);

  int test_mode;
  int worker_id = port - 10000 + 1;
  string config_file_path = CONFIG_PATH_ROOT + model_name + "_" + to_string(worker_id) + ".json";
  vector<int> layer_specs_vec = read_exp_specs(
    config_file_path.c_str(),
    &epsilon,
    INPUTS_PATH,
    PARAMETERS_PATH,
    LOGS_PATH,
    &test_mode,
    &test_examples,
    worker_id
  );

  cout << "Dataset: " << CURR_DATASET << "\n";

  int num_layers = layer_specs_vec.back();
  int* layer_specs = layer_specs_vec.data();

  if(test_mode == 0){
    float_inference(ios, layer_specs, num_layers);
  } else if(test_mode == 1){
    field_inference(ios, layer_specs, num_layers);
  } else if(test_mode == 2){
    float_inference(ios, layer_specs, num_layers);
    field_inference(ios, layer_specs, num_layers);
  }
}


int main(int argc, char **argv) {
  parse_party_and_port(argv, &party, &port);
  BoolIO<NetIO> *ios[threads];
  for (int i = 0; i < threads; ++i)
    ios[i] = new BoolIO<NetIO>(
        new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i),
        party == ALICE);

  if(argc > 3){
    model_name = (argv[3]);
    cout << model_name << "\n";
  }

  if(argc > 4){
    base_example = atoi(argv[4]);
  }

  if(argc > 5){
    num_examples = atoi(argv[5]);
  }

  if(argc > 6){
    print_to_stdout = (bool) atoi(argv[6]);
  }

  cout << "PARTY = " << party << "\n";

  ONLY_INFERENCE = true;
  test_inference(ios, party);

  for (int i = 0; i < threads; ++i) {
    delete ios[i]->io;
    delete ios[i];
  }

  return 0;
}
