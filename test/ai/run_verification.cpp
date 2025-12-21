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


void float_verification(BoolIO<NetIO> *ios[threads], int* layer_specs, int num_layers){
  auto start = clock_start();

  int num_examples_verified = 0;
  int num_examples_classified = 0;

  // float 
  std::ofstream file(std::string(LOGS_PATH) + "_float_worker_" + std::to_string(port - 10000 + 1) + (DO_DP_BS ? ".txt" : "_opt.txt"));
  std::streambuf* original_buf = std::cout.rdbuf(file.rdbuf());

  VerifiableFeedForwardNeuralNetwork<float>* model_float = create_model<float>(num_layers, layer_specs, party);
  model_float->load_weights_and_biases(PARAMETERS_PATH.c_str());

  set<int> correctly_classified_examples;
  set<int> verified_examples;

  for(int i = base_example; i < base_example + num_examples; i++){
    auto result = verify_example<float>(model_float, INPUTS_PATH.c_str(), i*(NUM_FEATURES[CURR_DATASET]+1), epsilon, i+1);
    bool classified = result.first;
    bool verified = result.second;

    if(classified){
      correctly_classified_examples.insert(i+1);
    }

    if(verified){
      verified_examples.insert(i+1);
    }

    num_examples_verified += (int) verified;
    num_examples_classified += (int) classified;

    if(party == ALICE)    cout << "EXAMPLE " << i+1 << " : " << (verified ? "YES" : "NO") << "\n";

    model_float->describe(false, false);
  }

  if(party == ALICE) model_float->savings_stats->print_stats();

  if(party == ALICE)   cout << "Verified " << num_examples_verified << "/" << num_examples << " examples [ correctly classified = " << num_examples_classified << " ]\n";

  double tt = time_from(start);
  if(party == ALICE)   cout << "\nAvg. time to verify: " << (tt/1000000)/num_examples << " s\n";

  if(party == ALICE){
    cout << "Correctly Classified Examples\n";
    for(auto e : correctly_classified_examples){
      cout << e << ", ";
    }
    cout << "\n\n";

    cout << "Verified Examples\n";
    for(auto e : verified_examples){
      cout << e << ", ";
    }
    cout << "\n\n";
  }

  std::cout.rdbuf(original_buf);  

  if(party == ALICE)   cout << "Float verification completed\n";
  if(party == ALICE)   cout << "Verified " << num_examples_verified << "/" << num_examples << " examples [ correctly classified = " << num_examples_classified << " ]\n";
  if(party == ALICE)   cout << "\n";
}


void field_verification(BoolIO<NetIO> *ios[threads], int* layer_specs, int num_layers){
  // field

  setup_plain_prot(false, "");
  setup_zk_arith<BoolIO<NetIO>>(ios, threads, party);

  init_verification();
  startComputation(party);

  double tt;

  std::ofstream field_file(std::string(LOGS_PATH) + "_worker_" + std::to_string(port - 10000 + 1) + ".txt");
  std::streambuf* original_buf2;
  if(!print_to_stdout) original_buf2 = std::cout.rdbuf(field_file.rdbuf());

  VerifiableFeedForwardNeuralNetwork<float>* model_float = create_model<float>(num_layers, layer_specs, party);
  model_float->load_weights_and_biases(PARAMETERS_PATH.c_str());

  VerifiableFeedForwardNeuralNetwork<IntFp>* model_field = create_model<IntFp>(num_layers, layer_specs, party);
  model_field->load_weights_and_biases(PARAMETERS_PATH.c_str());

  int num_examples_verified = 0;
  int num_examples_classified = 0;
  
  set<int> correctly_classified_examples;
  set<int> verified_examples;

  auto start = clock_start();
  if(base_example != -1){
    test_examples = vector<int>(100);
    std::iota(test_examples.begin(), test_examples.end(), 1);
    num_examples = test_examples.size();
  }
  num_examples = test_examples.size();
  
  for(int j : test_examples){
    int i = j-1;


    verify_example<float>(model_float, INPUTS_PATH.c_str(), i*(NUM_FEATURES[CURR_DATASET]+1), epsilon, i+1);
    model_field->skip_map = model_float->skip_map;
    
    auto result = verify_example<IntFp>(model_field, INPUTS_PATH.c_str(), i*(NUM_FEATURES[CURR_DATASET]+1), epsilon, i+1);
    bool classified = result.first;
    bool verified = result.second;

    if(classified){
      correctly_classified_examples.insert(i+1);
    }

    if(verified){
      verified_examples.insert(i+1);
    }

    num_examples_verified += (int) verified;
    num_examples_classified += (int) classified;

    tt = time_from(start);

    if(print_to_stdout){
      if(party == ALICE)   cout << "\rVerified: " << num_examples_verified << "/" << (i+1) << " images [Avg. time = " << (tt/(i+1))/1e6 << " sec]" << std::flush;
    } else {
      if(party == ALICE)   cout << (verified ? "YES" : "NO") << "\n";
    }

    model_field->describe(false, true);
  }


  if(party == ALICE){
    cout << "Correctly Classified Examples\n";
    for(auto e : correctly_classified_examples){
      cout << e << ", ";
    }
    cout << "\n\n";

    cout << "Verified Examples\n";
    for(auto e : verified_examples){
      cout << e << ", ";
    }
    cout << "\n\n";
  }

  if(party == ALICE)   cout << "\nVerified " << num_examples_verified << "/" << num_examples << " examples\n";

  bool cheated = finalize_zk_arith<BoolIO<NetIO>>();
  if(party == BOB){
    cerr << "\n" << (cheated ? "\033[31mVerfication failed!" : "\033[32mVerfication successful!") << "\033[0m\n";
  }

  tt = time_from(start); 
  if(party == ALICE)   cout << "\nAvg. time to verify: " << (tt/1000000)/num_examples << " s\n";
  if(party == ALICE)   cout << "Communication: " << ios[0]->counter/(1024.0 * 1024.0) << " MB\n";

  if(!print_to_stdout) std::cout.rdbuf(original_buf2);
}


void test_verification(BoolIO<NetIO> *ios[threads], int party) {
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
    float_verification(ios, layer_specs, num_layers);
  } else if(test_mode == 1){
    field_verification(ios, layer_specs, num_layers);
  } else if(test_mode == 2){
    float_verification(ios, layer_specs, num_layers);
    field_verification(ios, layer_specs, num_layers);
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
    BS_WAIVER_FRACTION = (float) atof(argv[6]);
  }

  if(argc > 7){
    print_to_stdout = (bool) atoi(argv[7]);
  }

  cout << "PARTY = " << party << "\n";


  test_verification(ios, party);

  for (int i = 0; i < threads; ++i) {
    delete ios[i]->io;
    delete ios[i];
  }

  return 0;
}
