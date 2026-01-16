#include "emp-tool/emp-tool.h"
#include <emp-zk/emp-zk.h>
#include <emp-zk/ai/ai.h>
#include <emp-zk/ai/utils.h>
#include <emp-zk/ai/secure-utils.h>

#include <iostream>
using namespace emp;
using namespace std;

int port, party;
const int threads = 1;
int sz = 0;
float v;

const char* PARAMS_PATH = "test/ai/data/test.txt";

void test_sigmoid_sankha(BoolIO<NetIO> *ios[threads], int party){
    setup_plain_prot(false, "");
    setup_zk_arith<BoolIO<NetIO>>(ios, threads, party);

    SCALE = FXPSCALE;

    init_verification();
    startComputation(party);

    IntFp* x_IntFp = new IntFp[sz];
    IntFp* y_IntFp = new IntFp[sz];
    IntFp* z_IntFp = new IntFp[sz];
    IntFp* lambda_IntFp = new IntFp[sz];

    uint64_t* clt_lambda = new uint64_t[sz];

    // cleartext float
    float* x = new float[sz];
    if(party == ALICE){        
        read_next_elements(sz, x, 0, PARAMS_PATH);
    }
    authenticate_over_field(sz, x, x_IntFp, party);

    auto start = clock_start();
    ZKSigmoidEW(party, x_IntFp, y_IntFp, sz, FXPSCALE);
    double tt = time_from(start);
    cerr << "\n\nSGM Sankha = " << tt/1e3 << " ms\n";

    cout << "\n\nSigmoid(x):\n";
    for(int i = 0; i < sz; i++){
        cout << format_EMP_IntFp(y_IntFp[i].reveal(), 1) << " ";
    }

    endComputation(party);

    bool cheated = finalize_zk_arith<BoolIO<NetIO>>();
    if(party == BOB && cheated){
        error("Cheat!\n");
    }
}

void test_sigmoid_chenkai(BoolIO<NetIO> *ios[threads], int party){
    setup_plain_prot(false, "");
    setup_zk_arith<BoolIO<NetIO>>(ios, threads, party);

    SCALE = FXPSCALE;

    init_verification();
    startComputation(party);

    IntFp* x_IntFp = new IntFp[sz];
    IntFp* y_IntFp = new IntFp[sz];
    IntFp* z_IntFp = new IntFp[sz];
    IntFp* lambda_IntFp = new IntFp[sz];

    uint64_t* clt_lambda = new uint64_t[sz];

    // int sz = 1;

    // cleartext float
    float* x = new float[sz];
    if(party == ALICE){        
        read_next_elements(sz, x, 0, PARAMS_PATH);
    }
    authenticate_over_field(sz, x, x_IntFp, party);


    auto start = clock_start();
    ZKSigmoid(party, x_IntFp, y_IntFp, sz, FXPSCALE);
    double tt = time_from(start);
    cerr << "\n\nSGM Chen = " << tt/1e3 << " ms\n";

    cout << "\n\nSigmoid(x):\n";
    for(int i = 0; i < sz; i++){
        cout << format_EMP_IntFp(y_IntFp[i].reveal(), 1) << " ";
    }


    // cout << "\n\nSigmoid'(x):\n";
    // for(int i = 0; i < sz; i++){
    //     cout << format_EMP_IntFp(z_IntFp[i].reveal(), 1) << " ";
    // }

    // IntFp del_y = y_IntFp[1] + y_IntFp[0].negate();
    // IntFp del_x = x_IntFp[1] + x_IntFp[0].negate();


    // if(party == ALICE){
    //     clt_lambda[0] = divide<uint64_t>(HIGH64(del_y.value), HIGH64(del_x.value));
    // }
    
    // lambda_IntFp[0] = IntFp(clt_lambda[0], ALICE);
    // cout << "\n" << format_EMP_IntFp(lambda_IntFp[0].reveal(), 1) << " ";


    // lambda_IntFp[0] = y_IntFp[0]*FIELD_SCALED_ONE + lambda_IntFp[0] * x_IntFp[0].negate();
    // ZKgeneralTruncAny(party, lambda_IntFp, lambda_IntFp, 1, FXPSCALE);

    // // lambda_IntFp[0] = lambda_IntFp[0] + y_IntFp[0];

    // cout << format_EMP_IntFp(lambda_IntFp[0].reveal(), 1) << " ";
    

    // ZKTanhAndDerivative(party, x_IntFp, y_IntFp, z_IntFp, sz);

    // cout << "\n\nTanh(x):\n";
    // for(int i = 0; i < sz; i++){
    //     cout << format_EMP_IntFp(y_IntFp[i].reveal(), 1) << " ";
    // }

    // cout << "\n\nTanh'(x):\n";
    // for(int i = 0; i < sz; i++){
    //     cout << format_EMP_IntFp(z_IntFp[i].reveal(), 1) << " ";
    // }
    // cout << "\n";

    endComputation(party);

    bool cheated = finalize_zk_arith<BoolIO<NetIO>>();
    if(party == BOB && cheated){
        error("Cheat!\n");
    }
}

void test_activations(BoolIO<NetIO> *ios[threads], int party){
    setup_plain_prot(false, "");
    setup_zk_arith<BoolIO<NetIO>>(ios, threads, party);

    SCALE = FXPSCALE;

    init_verification();
    startComputation(party);

    IntFp* x_IntFp = new IntFp[sz];
    IntFp* y_IntFp = new IntFp[sz];
    IntFp* z_IntFp = new IntFp[sz];

    // cleartext float
    float* x = new float[sz];

    if(party == ALICE){        
        read_next_elements(sz, x, 0, PARAMS_PATH);
    }
    authenticate_over_field(sz, x, x_IntFp, party);


    ZKSigmoidAndDerivative(party, x_IntFp, y_IntFp, z_IntFp, sz);

    cout << "\n\nSigmoid(x):\n";
    for(int i = 0; i < sz; i++){
        cout << format_EMP_IntFp(y_IntFp[i].reveal(), 1) << " ";
    }

    cout << "\n\nSigmoid'(x):\n";
    for(int i = 0; i < sz; i++){
        cout << format_EMP_IntFp(z_IntFp[i].reveal(), 1) << " ";
    }


    ZKTanhAndDerivative(party, x_IntFp, y_IntFp, z_IntFp, sz);

    cout << "\n\nTanh(x):\n";
    for(int i = 0; i < sz; i++){
        cout << format_EMP_IntFp(y_IntFp[i].reveal(), 1) << " ";
    }

    cout << "\n\nTanh'(x):\n";
    for(int i = 0; i < sz; i++){
        cout << format_EMP_IntFp(z_IntFp[i].reveal(), 1) << " ";
    }
    cout << "\n";

    endComputation(party);

    bool cheated = finalize_zk_arith<BoolIO<NetIO>>();
    if(party == BOB && cheated){
        error("Cheat!\n");
    }
}
    


int main(int argc, char** argv){
    parse_party_and_port(argv, &party, &port);
    BoolIO<NetIO> *ios[threads];
    for (int i = 0; i < threads; ++i)
        ios[i] = new BoolIO<NetIO>(
            new NetIO(party == ALICE ? nullptr : "127.0.0.1", port + i),
            party == ALICE);
            
    sz = atoi(argv[3]);
    FXPSCALE = atoi(argv[4]);
    // test_affine(ios, party);

    v = atof(argv[5]);

    if ((int) v == 1){
        test_sigmoid_sankha(ios, party);
    } else {
        test_sigmoid_chenkai(ios, party);
    }

    
}