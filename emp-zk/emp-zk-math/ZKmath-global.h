#include "emp-zk/emp-zk-math/LUT-range.h"
#include "emp-zk/emp-zk-math/LUT.h"
#include "emp-zk/emp-zk-math/LUT-twoValue.h"

#pragma once

#define BIT_LENGTH 61
extern int SCALE;

// LUTRange
extern int NUM_RANGE;
extern std::vector<std::unique_ptr<LUTRangeIntFp>> LUTRange; //[NUM_RANGE];  // 0~12

// LUTdiv
extern int DIV_M;     //(SCALE - 2)/2
extern int DIV_N;
// #define DIV_N 13
extern LUTTwoValueIntFp *LUTdiv;

// LUTextend
extern LUTIntFp *LUTextend;

// LUTexp
extern int EXP_N;
extern int EXP_DIGIT_LEN;
extern int EXP_LUT_NUM;       // EXP_LUT_NUM = ceil(EXP_N/EXP_DIGIT_LEN)
extern std::vector<std::unique_ptr<LUTIntFp>> LUTexp; 

// sqrt
extern int SQRT_M;    // SCALE/2
extern int SQRT_N;
extern LUTIntFp *LUTsqrt;
extern LUTIntFp *LUTsqrtExtend;

// LUTmsnzb
extern LUTTwoValueIntFp *LUTmsnzb2value;

// LUTcmp - verify
extern int CMP_DIGIT_LEN;
extern int CMP_LUT_NUM;        // CMP_LUT_NUM = ceil(BIT_LENGTH/CMP_DIGIT_LEN)
extern uint64_t FINIAL_CMP_LUT_NUM;
extern uint64_t CMP_LAST_DIGIT_BITLEN;
extern std::vector<std::unique_ptr<LUTIntFp>> LUTvrfyCmpLx;
extern LUTIntFp *LUTvrfyCmpLy;
extern std::vector<std::unique_ptr<LUTIntFp>> LUTvrfyCmpLx_InP;
// LUTcmp - computation
extern std::vector<std::unique_ptr<LUTTwoValueIntFp>> LUTCmpLx;

void startComputation(int party);
void endComputation(int party);

uint64_t Real2Field(double x, int scale);
double Field2Real(uint64_t x, int scale);
uint64_t computeULPErr(int64_t calc_fixed, int64_t actual_fixed);