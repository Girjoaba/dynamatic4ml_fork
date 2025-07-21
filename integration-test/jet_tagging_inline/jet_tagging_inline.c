#include "jet_tagging_inline.h"
#include "dynamatic/Integration.h"
#include <stdlib.h>
#include <stdint.h>

#define N_INPUT_1_1 16
#define N_LAYER_2 64
#define N_LAYER_5 32
#define N_LAYER_8 32
#define N_LAYER_11 5

void jet_tagging_inline(
    default_t input[N_INPUT_1_1],
    default_t z0[N_LAYER_2],
    default_t w1[N_LAYER_5][N_LAYER_2],
    default_t b1[N_LAYER_5],
    default_t z1[N_LAYER_5],
    default_t w2[N_LAYER_8][N_LAYER_5],
    default_t b2[N_LAYER_8],
    default_t z2[N_LAYER_8],
    default_t z3[N_LAYER_11]
    ) {

    // ===========================================================================
    // Layer 0: Dense ReLU
    dense_accum_t acc0_0;
    default_t tmp_relu0_0 = 0;
    dense_accum_t acc0_1;
    default_t tmp_relu0_1 = 0;
    dense_accum_t acc0_2;
    default_t tmp_relu0_2 = 0;
    dense_accum_t acc0_3;
    default_t tmp_relu0_3 = 0;
    dense_accum_t acc0_4;
    default_t tmp_relu0_4 = 0;
    dense_accum_t acc0_5;
    default_t tmp_relu0_5 = 0;
    dense_accum_t acc0_6;
    default_t tmp_relu0_6 = 0;
    dense_accum_t acc0_7;
    default_t tmp_relu0_7 = 0;
    dense_accum_t acc0_8;
    default_t tmp_relu0_8 = 0;
    dense_accum_t acc0_9;
    default_t tmp_relu0_9 = 0;
    dense_accum_t acc0_10;
    default_t tmp_relu0_10 = 0;
    dense_accum_t acc0_11;
    default_t tmp_relu0_11 = 0;
    dense_accum_t acc0_12;
    default_t tmp_relu0_12 = 0;
    dense_accum_t acc0_13;
    default_t tmp_relu0_13 = 0;
    dense_accum_t acc0_14;
    default_t tmp_relu0_14 = 0;
    dense_accum_t acc0_15;
    default_t tmp_relu0_15 = 0;
    dense_accum_t acc0_16;
    default_t tmp_relu0_16 = 0;
    dense_accum_t acc0_17;
    default_t tmp_relu0_17 = 0;
    dense_accum_t acc0_18;
    default_t tmp_relu0_18 = 0;
    dense_accum_t acc0_19;
    default_t tmp_relu0_19 = 0;
    dense_accum_t acc0_20;
    default_t tmp_relu0_20 = 0;
    dense_accum_t acc0_21;
    default_t tmp_relu0_21 = 0;
    dense_accum_t acc0_22;
    default_t tmp_relu0_22 = 0;
    dense_accum_t acc0_23;
    default_t tmp_relu0_23 = 0;
    dense_accum_t acc0_24;
    default_t tmp_relu0_24 = 0;
    dense_accum_t acc0_25;
    default_t tmp_relu0_25 = 0;
    dense_accum_t acc0_26;
    default_t tmp_relu0_26 = 0;
    dense_accum_t acc0_27;
    default_t tmp_relu0_27 = 0;
    dense_accum_t acc0_28;
    default_t tmp_relu0_28 = 0;
    dense_accum_t acc0_29;
    default_t tmp_relu0_29 = 0;
    dense_accum_t acc0_30;
    default_t tmp_relu0_30 = 0;
    dense_accum_t acc0_31;
    default_t tmp_relu0_31 = 0;
    dense_accum_t acc0_32;
    default_t tmp_relu0_32 = 0;
    dense_accum_t acc0_33;
    default_t tmp_relu0_33 = 0;
    dense_accum_t acc0_34;
    default_t tmp_relu0_34 = 0;
    dense_accum_t acc0_35;
    default_t tmp_relu0_35 = 0;
    dense_accum_t acc0_36;
    default_t tmp_relu0_36 = 0;
    dense_accum_t acc0_37;
    default_t tmp_relu0_37 = 0;
    dense_accum_t acc0_38;
    default_t tmp_relu0_38 = 0;
    dense_accum_t acc0_39;
    default_t tmp_relu0_39 = 0;
    dense_accum_t acc0_40;
    default_t tmp_relu0_40 = 0;
    dense_accum_t acc0_41;
    default_t tmp_relu0_41 = 0;
    dense_accum_t acc0_42;
    default_t tmp_relu0_42 = 0;
    dense_accum_t acc0_43;
    default_t tmp_relu0_43 = 0;
    dense_accum_t acc0_44;
    default_t tmp_relu0_44 = 0;
    dense_accum_t acc0_45;
    default_t tmp_relu0_45 = 0;
    dense_accum_t acc0_46;
    default_t tmp_relu0_46 = 0;
    dense_accum_t acc0_47;
    default_t tmp_relu0_47 = 0;
    dense_accum_t acc0_48;
    default_t tmp_relu0_48 = 0;
    dense_accum_t acc0_49;
    default_t tmp_relu0_49 = 0;
    dense_accum_t acc0_50;
    default_t tmp_relu0_50 = 0;
    dense_accum_t acc0_51;
    default_t tmp_relu0_51 = 0;
    dense_accum_t acc0_52;
    default_t tmp_relu0_52 = 0;
    dense_accum_t acc0_53;
    default_t tmp_relu0_53 = 0;
    dense_accum_t acc0_54;
    default_t tmp_relu0_54 = 0;
    dense_accum_t acc0_55;
    default_t tmp_relu0_55 = 0;
    dense_accum_t acc0_56;
    default_t tmp_relu0_56 = 0;
    dense_accum_t acc0_57;
    default_t tmp_relu0_57 = 0;
    dense_accum_t acc0_58;
    default_t tmp_relu0_58 = 0;
    dense_accum_t acc0_59;
    default_t tmp_relu0_59 = 0;
    dense_accum_t acc0_60;
    default_t tmp_relu0_60 = 0;
    dense_accum_t acc0_61;
    default_t tmp_relu0_61 = 0;
    dense_accum_t acc0_62;
    default_t tmp_relu0_62 = 0;
    dense_accum_t acc0_63;
    default_t tmp_relu0_63 = 0;
    default_t input_0 = input[0];
    default_t input_1 = input[1];
    default_t input_2 = input[2];
    default_t input_3 = input[3];
    default_t input_4 = input[4];
    default_t input_5 = input[5];
    default_t input_6 = input[6];
    default_t input_7 = input[7];
    default_t input_8 = input[8];
    default_t input_9 = input[9];
    default_t input_10 = input[10];
    default_t input_11 = input[11];
    default_t input_12 = input[12];
    default_t input_13 = input[13];
    default_t input_14 = input[14];
    default_t input_15 = input[15];
    /* ReLU Layer Iteration: 0 */
    acc0_0 = 0;
    /* Unrolled Dot Product */
    acc0_0 += input_0 * -142;
    acc0_0 += input_1 * -259;
    acc0_0 += input_2 * 167;
    acc0_0 += input_3 * -479;
    acc0_0 += input_4 * 112;
    acc0_0 += input_5 * -2;
    acc0_0 += input_6 * 413;
    acc0_0 += input_7 * -181;
    acc0_0 += input_8 * 262;
    acc0_0 += input_9 * -38;
    acc0_0 += input_10 * -2;
    acc0_0 += input_11 * -107;
    acc0_0 += input_12 * 200;
    acc0_0 += input_13 * -89;
    acc0_0 += input_14 * 349;
    acc0_0 += input_15 * -184;
    acc0_0 += (dense_accum_t)(177 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_0 = acc0_0 >> (FRAC_DEFAULT);
    tmp_relu0_0 = (default_t)acc0_0;

    /* RELU ACTIVATION */
    z0[0] = tmp_relu0_0 > 0 ? tmp_relu0_0 : 0;

    /* ReLU Layer Iteration: 1 */
    acc0_1 = 0;
    /* Unrolled Dot Product */
    acc0_1 += input_0 * 484;
    acc0_1 += input_1 * 77;
    acc0_1 += input_2 * 0;
    acc0_1 += input_3 * -2;
    acc0_1 += input_4 * 50;
    acc0_1 += input_5 * -14;
    acc0_1 += input_6 * 280;
    acc0_1 += input_7 * -308;
    acc0_1 += input_8 * -6;
    acc0_1 += input_9 * -316;
    acc0_1 += input_10 * 397;
    acc0_1 += input_11 * -279;
    acc0_1 += input_12 * 5;
    acc0_1 += input_13 * 322;
    acc0_1 += input_14 * 75;
    acc0_1 += input_15 * -452;
    acc0_1 += (dense_accum_t)(247 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_1 = acc0_1 >> (FRAC_DEFAULT);
    tmp_relu0_1 = (default_t)acc0_1;

    /* RELU ACTIVATION */
    z0[1] = tmp_relu0_1 > 0 ? tmp_relu0_1 : 0;

    /* ReLU Layer Iteration: 2 */
    acc0_2 = 0;
    /* Unrolled Dot Product */
    acc0_2 += input_0 * 325;
    acc0_2 += input_1 * 39;
    acc0_2 += input_2 * 419;
    acc0_2 += input_3 * 39;
    acc0_2 += input_4 * 130;
    acc0_2 += input_5 * -203;
    acc0_2 += input_6 * -51;
    acc0_2 += input_7 * -27;
    acc0_2 += input_8 * -71;
    acc0_2 += input_9 * 238;
    acc0_2 += input_10 * 147;
    acc0_2 += input_11 * 141;
    acc0_2 += input_12 * 88;
    acc0_2 += input_13 * -26;
    acc0_2 += input_14 * -283;
    acc0_2 += input_15 * 7;
    acc0_2 += (dense_accum_t)(293 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_2 = acc0_2 >> (FRAC_DEFAULT);
    tmp_relu0_2 = (default_t)acc0_2;

    /* RELU ACTIVATION */
    z0[2] = tmp_relu0_2 > 0 ? tmp_relu0_2 : 0;

    /* ReLU Layer Iteration: 3 */
    acc0_3 = 0;
    /* Unrolled Dot Product */
    acc0_3 += input_0 * 213;
    acc0_3 += input_1 * 222;
    acc0_3 += input_2 * -481;
    acc0_3 += input_3 * -411;
    acc0_3 += input_4 * -212;
    acc0_3 += input_5 * -370;
    acc0_3 += input_6 * -148;
    acc0_3 += input_7 * 274;
    acc0_3 += input_8 * -7;
    acc0_3 += input_9 * 46;
    acc0_3 += input_10 * -261;
    acc0_3 += input_11 * 1;
    acc0_3 += input_12 * 345;
    acc0_3 += input_13 * -156;
    acc0_3 += input_14 * -116;
    acc0_3 += input_15 * 244;
    acc0_3 += (dense_accum_t)(92 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_3 = acc0_3 >> (FRAC_DEFAULT);
    tmp_relu0_3 = (default_t)acc0_3;

    /* RELU ACTIVATION */
    z0[3] = tmp_relu0_3 > 0 ? tmp_relu0_3 : 0;

    /* ReLU Layer Iteration: 4 */
    acc0_4 = 0;
    /* Unrolled Dot Product */
    acc0_4 += input_0 * 168;
    acc0_4 += input_1 * -145;
    acc0_4 += input_2 * 370;
    acc0_4 += input_3 * 45;
    acc0_4 += input_4 * -109;
    acc0_4 += input_5 * -200;
    acc0_4 += input_6 * 239;
    acc0_4 += input_7 * -248;
    acc0_4 += input_8 * -2;
    acc0_4 += input_9 * -26;
    acc0_4 += input_10 * 1;
    acc0_4 += input_11 * -366;
    acc0_4 += input_12 * -17;
    acc0_4 += input_13 * 0;
    acc0_4 += input_14 * -724;
    acc0_4 += input_15 * 117;
    acc0_4 += (dense_accum_t)(202 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_4 = acc0_4 >> (FRAC_DEFAULT);
    tmp_relu0_4 = (default_t)acc0_4;

    /* RELU ACTIVATION */
    z0[4] = tmp_relu0_4 > 0 ? tmp_relu0_4 : 0;

    /* ReLU Layer Iteration: 5 */
    acc0_5 = 0;
    /* Unrolled Dot Product */
    acc0_5 += input_0 * -107;
    acc0_5 += input_1 * -247;
    acc0_5 += input_2 * -78;
    acc0_5 += input_3 * 9;
    acc0_5 += input_4 * -282;
    acc0_5 += input_5 * 1;
    acc0_5 += input_6 * -82;
    acc0_5 += input_7 * 207;
    acc0_5 += input_8 * -97;
    acc0_5 += input_9 * -3;
    acc0_5 += input_10 * 373;
    acc0_5 += input_11 * 5;
    acc0_5 += input_12 * -87;
    acc0_5 += input_13 * -142;
    acc0_5 += input_14 * -569;
    acc0_5 += input_15 * 0;
    acc0_5 += (dense_accum_t)(227 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_5 = acc0_5 >> (FRAC_DEFAULT);
    tmp_relu0_5 = (default_t)acc0_5;

    /* RELU ACTIVATION */
    z0[5] = tmp_relu0_5 > 0 ? tmp_relu0_5 : 0;

    /* ReLU Layer Iteration: 6 */
    acc0_6 = 0;
    /* Unrolled Dot Product */
    acc0_6 += input_0 * -2;
    acc0_6 += input_1 * -3;
    acc0_6 += input_2 * 865;
    acc0_6 += input_3 * 1093;
    acc0_6 += input_4 * 769;
    acc0_6 += input_5 * 0;
    acc0_6 += input_6 * -38;
    acc0_6 += input_7 * -214;
    acc0_6 += input_8 * 143;
    acc0_6 += input_9 * -182;
    acc0_6 += input_10 * 141;
    acc0_6 += input_11 * -172;
    acc0_6 += input_12 * -403;
    acc0_6 += input_13 * 390;
    acc0_6 += input_14 * 280;
    acc0_6 += input_15 * 0;
    acc0_6 += -(dense_accum_t)(37 << (FRAC_DEFAULT));

    acc0_6 = acc0_6 >> (FRAC_DEFAULT);
    tmp_relu0_6 = (default_t)acc0_6;

    /* RELU ACTIVATION */
    z0[6] = tmp_relu0_6 > 0 ? tmp_relu0_6 : 0;

    /* ReLU Layer Iteration: 7 */
    acc0_7 = 0;
    /* Unrolled Dot Product */
    acc0_7 += input_0 * -34;
    acc0_7 += input_1 * 35;
    acc0_7 += input_2 * 199;
    acc0_7 += input_3 * 310;
    acc0_7 += input_4 * -517;
    acc0_7 += input_5 * 138;
    acc0_7 += input_6 * -81;
    acc0_7 += input_7 * 154;
    acc0_7 += input_8 * 269;
    acc0_7 += input_9 * -79;
    acc0_7 += input_10 * -3;
    acc0_7 += input_11 * -31;
    acc0_7 += input_12 * -1;
    acc0_7 += input_13 * -8;
    acc0_7 += input_14 * 74;
    acc0_7 += input_15 * 33;
    acc0_7 += -(dense_accum_t)(457 << (FRAC_DEFAULT));

    acc0_7 = acc0_7 >> (FRAC_DEFAULT);
    tmp_relu0_7 = (default_t)acc0_7;

    /* RELU ACTIVATION */
    z0[7] = tmp_relu0_7 > 0 ? tmp_relu0_7 : 0;

    /* ReLU Layer Iteration: 8 */
    acc0_8 = 0;
    /* Unrolled Dot Product */
    acc0_8 += input_0 * -51;
    acc0_8 += input_1 * 1;
    acc0_8 += input_2 * 14;
    acc0_8 += input_3 * -9;
    acc0_8 += input_4 * 343;
    acc0_8 += input_5 * -354;
    acc0_8 += input_6 * -4;
    acc0_8 += input_7 * -243;
    acc0_8 += input_8 * 82;
    acc0_8 += input_9 * 31;
    acc0_8 += input_10 * 286;
    acc0_8 += input_11 * 229;
    acc0_8 += input_12 * -152;
    acc0_8 += input_13 * -93;
    acc0_8 += input_14 * 517;
    acc0_8 += input_15 * -303;
    acc0_8 += (dense_accum_t)(269 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_8 = acc0_8 >> (FRAC_DEFAULT);
    tmp_relu0_8 = (default_t)acc0_8;

    /* RELU ACTIVATION */
    z0[8] = tmp_relu0_8 > 0 ? tmp_relu0_8 : 0;

    /* ReLU Layer Iteration: 9 */
    acc0_9 = 0;
    /* Unrolled Dot Product */
    acc0_9 += input_0 * -28;
    acc0_9 += input_1 * 66;
    acc0_9 += input_2 * -215;
    acc0_9 += input_3 * -451;
    acc0_9 += input_4 * 56;
    acc0_9 += input_5 * -1;
    acc0_9 += input_6 * -205;
    acc0_9 += input_7 * -210;
    acc0_9 += input_8 * 173;
    acc0_9 += input_9 * 159;
    acc0_9 += input_10 * 210;
    acc0_9 += input_11 * 44;
    acc0_9 += input_12 * -208;
    acc0_9 += input_13 * -138;
    acc0_9 += input_14 * -664;
    acc0_9 += input_15 * -246;
    acc0_9 += (dense_accum_t)(241 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_9 = acc0_9 >> (FRAC_DEFAULT);
    tmp_relu0_9 = (default_t)acc0_9;

    /* RELU ACTIVATION */
    z0[9] = tmp_relu0_9 > 0 ? tmp_relu0_9 : 0;

    /* ReLU Layer Iteration: 10 */
    acc0_10 = 0;
    /* Unrolled Dot Product */
    acc0_10 += input_0 * 233;
    acc0_10 += input_1 * -101;
    acc0_10 += input_2 * 74;
    acc0_10 += input_3 * 312;
    acc0_10 += input_4 * -81;
    acc0_10 += input_5 * 181;
    acc0_10 += input_6 * 167;
    acc0_10 += input_7 * -182;
    acc0_10 += input_8 * 0;
    acc0_10 += input_9 * 248;
    acc0_10 += input_10 * 89;
    acc0_10 += input_11 * 203;
    acc0_10 += input_12 * 6;
    acc0_10 += input_13 * 72;
    acc0_10 += input_14 * 189;
    acc0_10 += input_15 * -5;
    acc0_10 += -(dense_accum_t)(86 << (FRAC_DEFAULT));

    acc0_10 = acc0_10 >> (FRAC_DEFAULT);
    tmp_relu0_10 = (default_t)acc0_10;

    /* RELU ACTIVATION */
    z0[10] = tmp_relu0_10 > 0 ? tmp_relu0_10 : 0;

    /* ReLU Layer Iteration: 11 */
    acc0_11 = 0;
    /* Unrolled Dot Product */
    acc0_11 += input_0 * -122;
    acc0_11 += input_1 * 160;
    acc0_11 += input_2 * 0;
    acc0_11 += input_3 * -466;
    acc0_11 += input_4 * -131;
    acc0_11 += input_5 * -158;
    acc0_11 += input_6 * -249;
    acc0_11 += input_7 * 221;
    acc0_11 += input_8 * 290;
    acc0_11 += input_9 * -195;
    acc0_11 += input_10 * -233;
    acc0_11 += input_11 * -349;
    acc0_11 += input_12 * -3;
    acc0_11 += input_13 * -38;
    acc0_11 += input_14 * 199;
    acc0_11 += input_15 * 117;
    acc0_11 += (dense_accum_t)(250 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_11 = acc0_11 >> (FRAC_DEFAULT);
    tmp_relu0_11 = (default_t)acc0_11;

    /* RELU ACTIVATION */
    z0[11] = tmp_relu0_11 > 0 ? tmp_relu0_11 : 0;

    /* ReLU Layer Iteration: 12 */
    acc0_12 = 0;
    /* Unrolled Dot Product */
    acc0_12 += input_0 * -2;
    acc0_12 += input_1 * 3;
    acc0_12 += input_2 * -313;
    acc0_12 += input_3 * 161;
    acc0_12 += input_4 * -104;
    acc0_12 += input_5 * 30;
    acc0_12 += input_6 * -335;
    acc0_12 += input_7 * -97;
    acc0_12 += input_8 * 265;
    acc0_12 += input_9 * -42;
    acc0_12 += input_10 * 139;
    acc0_12 += input_11 * 181;
    acc0_12 += input_12 * 209;
    acc0_12 += input_13 * 20;
    acc0_12 += input_14 * 251;
    acc0_12 += input_15 * 343;
    acc0_12 += (dense_accum_t)(1 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_12 = acc0_12 >> (FRAC_DEFAULT);
    tmp_relu0_12 = (default_t)acc0_12;

    /* RELU ACTIVATION */
    z0[12] = tmp_relu0_12 > 0 ? tmp_relu0_12 : 0;

    /* ReLU Layer Iteration: 13 */
    acc0_13 = 0;
    /* Unrolled Dot Product */
    acc0_13 += input_0 * 177;
    acc0_13 += input_1 * 564;
    acc0_13 += input_2 * 106;
    acc0_13 += input_3 * 57;
    acc0_13 += input_4 * 326;
    acc0_13 += input_5 * 243;
    acc0_13 += input_6 * 61;
    acc0_13 += input_7 * -69;
    acc0_13 += input_8 * -46;
    acc0_13 += input_9 * 53;
    acc0_13 += input_10 * -128;
    acc0_13 += input_11 * 198;
    acc0_13 += input_12 * -173;
    acc0_13 += input_13 * 8;
    acc0_13 += input_14 * -995;
    acc0_13 += input_15 * 299;
    acc0_13 += (dense_accum_t)(100 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_13 = acc0_13 >> (FRAC_DEFAULT);
    tmp_relu0_13 = (default_t)acc0_13;

    /* RELU ACTIVATION */
    z0[13] = tmp_relu0_13 > 0 ? tmp_relu0_13 : 0;

    /* ReLU Layer Iteration: 14 */
    acc0_14 = 0;
    /* Unrolled Dot Product */
    acc0_14 += input_0 * -12;
    acc0_14 += input_1 * -263;
    acc0_14 += input_2 * 181;
    acc0_14 += input_3 * -438;
    acc0_14 += input_4 * 372;
    acc0_14 += input_5 * 32;
    acc0_14 += input_6 * 0;
    acc0_14 += input_7 * -48;
    acc0_14 += input_8 * -200;
    acc0_14 += input_9 * -138;
    acc0_14 += input_10 * -73;
    acc0_14 += input_11 * 221;
    acc0_14 += input_12 * 0;
    acc0_14 += input_13 * -207;
    acc0_14 += input_14 * 65;
    acc0_14 += input_15 * -5;
    acc0_14 += (dense_accum_t)(268 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_14 = acc0_14 >> (FRAC_DEFAULT);
    tmp_relu0_14 = (default_t)acc0_14;

    /* RELU ACTIVATION */
    z0[14] = tmp_relu0_14 > 0 ? tmp_relu0_14 : 0;

    /* ReLU Layer Iteration: 15 */
    acc0_15 = 0;
    /* Unrolled Dot Product */
    acc0_15 += input_0 * 225;
    acc0_15 += input_1 * 386;
    acc0_15 += input_2 * 2;
    acc0_15 += input_3 * -1;
    acc0_15 += input_4 * -156;
    acc0_15 += input_5 * 66;
    acc0_15 += input_6 * -3;
    acc0_15 += input_7 * 12;
    acc0_15 += input_8 * -74;
    acc0_15 += input_9 * -275;
    acc0_15 += input_10 * -257;
    acc0_15 += input_11 * -313;
    acc0_15 += input_12 * -205;
    acc0_15 += input_13 * -314;
    acc0_15 += input_14 * 737;
    acc0_15 += input_15 * -359;
    acc0_15 += -(dense_accum_t)(39 << (FRAC_DEFAULT));

    acc0_15 = acc0_15 >> (FRAC_DEFAULT);
    tmp_relu0_15 = (default_t)acc0_15;

    /* RELU ACTIVATION */
    z0[15] = tmp_relu0_15 > 0 ? tmp_relu0_15 : 0;

    /* ReLU Layer Iteration: 16 */
    acc0_16 = 0;
    /* Unrolled Dot Product */
    acc0_16 += input_0 * -281;
    acc0_16 += input_1 * -117;
    acc0_16 += input_2 * -504;
    acc0_16 += input_3 * -691;
    acc0_16 += input_4 * 170;
    acc0_16 += input_5 * -264;
    acc0_16 += input_6 * 8;
    acc0_16 += input_7 * -45;
    acc0_16 += input_8 * -124;
    acc0_16 += input_9 * -1;
    acc0_16 += input_10 * -368;
    acc0_16 += input_11 * 100;
    acc0_16 += input_12 * 337;
    acc0_16 += input_13 * 2;
    acc0_16 += input_14 * 692;
    acc0_16 += input_15 * -245;
    acc0_16 += (dense_accum_t)(270 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_16 = acc0_16 >> (FRAC_DEFAULT);
    tmp_relu0_16 = (default_t)acc0_16;

    /* RELU ACTIVATION */
    z0[16] = tmp_relu0_16 > 0 ? tmp_relu0_16 : 0;

    /* ReLU Layer Iteration: 17 */
    acc0_17 = 0;
    /* Unrolled Dot Product */
    acc0_17 += input_0 * -29;
    acc0_17 += input_1 * -3;
    acc0_17 += input_2 * -14;
    acc0_17 += input_3 * -1;
    acc0_17 += input_4 * 105;
    acc0_17 += input_5 * 0;
    acc0_17 += input_6 * 0;
    acc0_17 += input_7 * 5;
    acc0_17 += input_8 * 0;
    acc0_17 += input_9 * -31;
    acc0_17 += input_10 * -1;
    acc0_17 += input_11 * 21;
    acc0_17 += input_12 * 0;
    acc0_17 += input_13 * -1;
    acc0_17 += input_14 * -83;
    acc0_17 += input_15 * 324;
    acc0_17 += -(dense_accum_t)(77 << (FRAC_DEFAULT));

    acc0_17 = acc0_17 >> (FRAC_DEFAULT);
    tmp_relu0_17 = (default_t)acc0_17;

    /* RELU ACTIVATION */
    z0[17] = tmp_relu0_17 > 0 ? tmp_relu0_17 : 0;

    /* ReLU Layer Iteration: 18 */
    acc0_18 = 0;
    /* Unrolled Dot Product */
    acc0_18 += input_0 * 223;
    acc0_18 += input_1 * 331;
    acc0_18 += input_2 * -173;
    acc0_18 += input_3 * -409;
    acc0_18 += input_4 * -364;
    acc0_18 += input_5 * -61;
    acc0_18 += input_6 * 18;
    acc0_18 += input_7 * -247;
    acc0_18 += input_8 * 219;
    acc0_18 += input_9 * 471;
    acc0_18 += input_10 * 178;
    acc0_18 += input_11 * -281;
    acc0_18 += input_12 * -346;
    acc0_18 += input_13 * -21;
    acc0_18 += input_14 * 574;
    acc0_18 += input_15 * 136;
    acc0_18 += (dense_accum_t)(216 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_18 = acc0_18 >> (FRAC_DEFAULT);
    tmp_relu0_18 = (default_t)acc0_18;

    /* RELU ACTIVATION */
    z0[18] = tmp_relu0_18 > 0 ? tmp_relu0_18 : 0;

    /* ReLU Layer Iteration: 19 */
    acc0_19 = 0;
    /* Unrolled Dot Product */
    acc0_19 += input_0 * 306;
    acc0_19 += input_1 * -108;
    acc0_19 += input_2 * 262;
    acc0_19 += input_3 * 0;
    acc0_19 += input_4 * -28;
    acc0_19 += input_5 * 313;
    acc0_19 += input_6 * 299;
    acc0_19 += input_7 * 244;
    acc0_19 += input_8 * 125;
    acc0_19 += input_9 * 288;
    acc0_19 += input_10 * 2;
    acc0_19 += input_11 * -91;
    acc0_19 += input_12 * 35;
    acc0_19 += input_13 * 8;
    acc0_19 += input_14 * 42;
    acc0_19 += input_15 * -307;
    acc0_19 += (dense_accum_t)(90 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_19 = acc0_19 >> (FRAC_DEFAULT);
    tmp_relu0_19 = (default_t)acc0_19;

    /* RELU ACTIVATION */
    z0[19] = tmp_relu0_19 > 0 ? tmp_relu0_19 : 0;

    /* ReLU Layer Iteration: 20 */
    acc0_20 = 0;
    /* Unrolled Dot Product */
    acc0_20 += input_0 * 38;
    acc0_20 += input_1 * 29;
    acc0_20 += input_2 * -56;
    acc0_20 += input_3 * -328;
    acc0_20 += input_4 * 402;
    acc0_20 += input_5 * -87;
    acc0_20 += input_6 * 20;
    acc0_20 += input_7 * 34;
    acc0_20 += input_8 * -46;
    acc0_20 += input_9 * 277;
    acc0_20 += input_10 * 5;
    acc0_20 += input_11 * 5;
    acc0_20 += input_12 * -266;
    acc0_20 += input_13 * 21;
    acc0_20 += input_14 * 1133;
    acc0_20 += input_15 * -31;
    acc0_20 += (dense_accum_t)(195 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_20 = acc0_20 >> (FRAC_DEFAULT);
    tmp_relu0_20 = (default_t)acc0_20;

    /* RELU ACTIVATION */
    z0[20] = tmp_relu0_20 > 0 ? tmp_relu0_20 : 0;

    /* ReLU Layer Iteration: 21 */
    acc0_21 = 0;
    /* Unrolled Dot Product */
    acc0_21 += input_0 * -37;
    acc0_21 += input_1 * 616;
    acc0_21 += input_2 * 0;
    acc0_21 += input_3 * -4;
    acc0_21 += input_4 * 99;
    acc0_21 += input_5 * 106;
    acc0_21 += input_6 * 51;
    acc0_21 += input_7 * -116;
    acc0_21 += input_8 * 1;
    acc0_21 += input_9 * 0;
    acc0_21 += input_10 * 148;
    acc0_21 += input_11 * 0;
    acc0_21 += input_12 * 297;
    acc0_21 += input_13 * 0;
    acc0_21 += input_14 * -3;
    acc0_21 += input_15 * 100;
    acc0_21 += (dense_accum_t)(112 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_21 = acc0_21 >> (FRAC_DEFAULT);
    tmp_relu0_21 = (default_t)acc0_21;

    /* RELU ACTIVATION */
    z0[21] = tmp_relu0_21 > 0 ? tmp_relu0_21 : 0;

    /* ReLU Layer Iteration: 22 */
    acc0_22 = 0;
    /* Unrolled Dot Product */
    acc0_22 += input_0 * -28;
    acc0_22 += input_1 * 616;
    acc0_22 += input_2 * 58;
    acc0_22 += input_3 * -3;
    acc0_22 += input_4 * 187;
    acc0_22 += input_5 * -138;
    acc0_22 += input_6 * 230;
    acc0_22 += input_7 * -99;
    acc0_22 += input_8 * -4;
    acc0_22 += input_9 * -160;
    acc0_22 += input_10 * 6;
    acc0_22 += input_11 * -203;
    acc0_22 += input_12 * 175;
    acc0_22 += input_13 * 129;
    acc0_22 += input_14 * -264;
    acc0_22 += input_15 * 1;
    acc0_22 += (dense_accum_t)(103 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_22 = acc0_22 >> (FRAC_DEFAULT);
    tmp_relu0_22 = (default_t)acc0_22;

    /* RELU ACTIVATION */
    z0[22] = tmp_relu0_22 > 0 ? tmp_relu0_22 : 0;

    /* ReLU Layer Iteration: 23 */
    acc0_23 = 0;
    /* Unrolled Dot Product */
    acc0_23 += input_0 * -92;
    acc0_23 += input_1 * 49;
    acc0_23 += input_2 * 258;
    acc0_23 += input_3 * 345;
    acc0_23 += input_4 * -190;
    acc0_23 += input_5 * 110;
    acc0_23 += input_6 * 222;
    acc0_23 += input_7 * -12;
    acc0_23 += input_8 * -17;
    acc0_23 += input_9 * -107;
    acc0_23 += input_10 * -299;
    acc0_23 += input_11 * -315;
    acc0_23 += input_12 * -43;
    acc0_23 += input_13 * 267;
    acc0_23 += input_14 * 349;
    acc0_23 += input_15 * 181;
    acc0_23 += -(dense_accum_t)(86 << (FRAC_DEFAULT));

    acc0_23 = acc0_23 >> (FRAC_DEFAULT);
    tmp_relu0_23 = (default_t)acc0_23;

    /* RELU ACTIVATION */
    z0[23] = tmp_relu0_23 > 0 ? tmp_relu0_23 : 0;

    /* ReLU Layer Iteration: 24 */
    acc0_24 = 0;
    /* Unrolled Dot Product */
    acc0_24 += input_0 * -380;
    acc0_24 += input_1 * 149;
    acc0_24 += input_2 * 113;
    acc0_24 += input_3 * -572;
    acc0_24 += input_4 * 183;
    acc0_24 += input_5 * -102;
    acc0_24 += input_6 * 245;
    acc0_24 += input_7 * -204;
    acc0_24 += input_8 * -183;
    acc0_24 += input_9 * -207;
    acc0_24 += input_10 * 23;
    acc0_24 += input_11 * -274;
    acc0_24 += input_12 * 346;
    acc0_24 += input_13 * -12;
    acc0_24 += input_14 * -494;
    acc0_24 += input_15 * -258;
    acc0_24 += (dense_accum_t)(343 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_24 = acc0_24 >> (FRAC_DEFAULT);
    tmp_relu0_24 = (default_t)acc0_24;

    /* RELU ACTIVATION */
    z0[24] = tmp_relu0_24 > 0 ? tmp_relu0_24 : 0;

    /* ReLU Layer Iteration: 25 */
    acc0_25 = 0;
    /* Unrolled Dot Product */
    acc0_25 += input_0 * 123;
    acc0_25 += input_1 * 197;
    acc0_25 += input_2 * -121;
    acc0_25 += input_3 * -378;
    acc0_25 += input_4 * 21;
    acc0_25 += input_5 * 0;
    acc0_25 += input_6 * 283;
    acc0_25 += input_7 * 264;
    acc0_25 += input_8 * -311;
    acc0_25 += input_9 * 146;
    acc0_25 += input_10 * -29;
    acc0_25 += input_11 * -9;
    acc0_25 += input_12 * -190;
    acc0_25 += input_13 * -381;
    acc0_25 += input_14 * -882;
    acc0_25 += input_15 * -105;
    acc0_25 += (dense_accum_t)(217 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_25 = acc0_25 >> (FRAC_DEFAULT);
    tmp_relu0_25 = (default_t)acc0_25;

    /* RELU ACTIVATION */
    z0[25] = tmp_relu0_25 > 0 ? tmp_relu0_25 : 0;

    /* ReLU Layer Iteration: 26 */
    acc0_26 = 0;
    /* Unrolled Dot Product */
    acc0_26 += input_0 * 609;
    acc0_26 += input_1 * 321;
    acc0_26 += input_2 * -270;
    acc0_26 += input_3 * -210;
    acc0_26 += input_4 * -448;
    acc0_26 += input_5 * 0;
    acc0_26 += input_6 * 57;
    acc0_26 += input_7 * -173;
    acc0_26 += input_8 * -23;
    acc0_26 += input_9 * 294;
    acc0_26 += input_10 * 29;
    acc0_26 += input_11 * 298;
    acc0_26 += input_12 * -61;
    acc0_26 += input_13 * 65;
    acc0_26 += input_14 * -1;
    acc0_26 += input_15 * -206;
    acc0_26 += -(dense_accum_t)(17 << (FRAC_DEFAULT));

    acc0_26 = acc0_26 >> (FRAC_DEFAULT);
    tmp_relu0_26 = (default_t)acc0_26;

    /* RELU ACTIVATION */
    z0[26] = tmp_relu0_26 > 0 ? tmp_relu0_26 : 0;

    /* ReLU Layer Iteration: 27 */
    acc0_27 = 0;
    /* Unrolled Dot Product */
    acc0_27 += input_0 * -50;
    acc0_27 += input_1 * 0;
    acc0_27 += input_2 * 200;
    acc0_27 += input_3 * 0;
    acc0_27 += input_4 * -325;
    acc0_27 += input_5 * 104;
    acc0_27 += input_6 * 119;
    acc0_27 += input_7 * 22;
    acc0_27 += input_8 * 1;
    acc0_27 += input_9 * -188;
    acc0_27 += input_10 * -43;
    acc0_27 += input_11 * 72;
    acc0_27 += input_12 * -178;
    acc0_27 += input_13 * 114;
    acc0_27 += input_14 * 817;
    acc0_27 += input_15 * 312;
    acc0_27 += -(dense_accum_t)(114 << (FRAC_DEFAULT));

    acc0_27 = acc0_27 >> (FRAC_DEFAULT);
    tmp_relu0_27 = (default_t)acc0_27;

    /* RELU ACTIVATION */
    z0[27] = tmp_relu0_27 > 0 ? tmp_relu0_27 : 0;

    /* ReLU Layer Iteration: 28 */
    acc0_28 = 0;
    /* Unrolled Dot Product */
    acc0_28 += input_0 * -405;
    acc0_28 += input_1 * -150;
    acc0_28 += input_2 * 28;
    acc0_28 += input_3 * 73;
    acc0_28 += input_4 * -351;
    acc0_28 += input_5 * 368;
    acc0_28 += input_6 * 270;
    acc0_28 += input_7 * 91;
    acc0_28 += input_8 * 16;
    acc0_28 += input_9 * -115;
    acc0_28 += input_10 * 39;
    acc0_28 += input_11 * 57;
    acc0_28 += input_12 * -99;
    acc0_28 += input_13 * -94;
    acc0_28 += input_14 * -293;
    acc0_28 += input_15 * -623;
    acc0_28 += -(dense_accum_t)(72 << (FRAC_DEFAULT));

    acc0_28 = acc0_28 >> (FRAC_DEFAULT);
    tmp_relu0_28 = (default_t)acc0_28;

    /* RELU ACTIVATION */
    z0[28] = tmp_relu0_28 > 0 ? tmp_relu0_28 : 0;

    /* ReLU Layer Iteration: 29 */
    acc0_29 = 0;
    /* Unrolled Dot Product */
    acc0_29 += input_0 * -150;
    acc0_29 += input_1 * -4;
    acc0_29 += input_2 * 17;
    acc0_29 += input_3 * 403;
    acc0_29 += input_4 * -315;
    acc0_29 += input_5 * 11;
    acc0_29 += input_6 * 150;
    acc0_29 += input_7 * 3;
    acc0_29 += input_8 * 76;
    acc0_29 += input_9 * -341;
    acc0_29 += input_10 * 0;
    acc0_29 += input_11 * -296;
    acc0_29 += input_12 * 10;
    acc0_29 += input_13 * -235;
    acc0_29 += input_14 * 140;
    acc0_29 += input_15 * 284;
    acc0_29 += -(dense_accum_t)(91 << (FRAC_DEFAULT));

    acc0_29 = acc0_29 >> (FRAC_DEFAULT);
    tmp_relu0_29 = (default_t)acc0_29;

    /* RELU ACTIVATION */
    z0[29] = tmp_relu0_29 > 0 ? tmp_relu0_29 : 0;

    /* ReLU Layer Iteration: 30 */
    acc0_30 = 0;
    /* Unrolled Dot Product */
    acc0_30 += input_0 * -148;
    acc0_30 += input_1 * 1;
    acc0_30 += input_2 * -444;
    acc0_30 += input_3 * -487;
    acc0_30 += input_4 * -97;
    acc0_30 += input_5 * -84;
    acc0_30 += input_6 * -216;
    acc0_30 += input_7 * -180;
    acc0_30 += input_8 * -257;
    acc0_30 += input_9 * -47;
    acc0_30 += input_10 * 356;
    acc0_30 += input_11 * 123;
    acc0_30 += input_12 * -90;
    acc0_30 += input_13 * 6;
    acc0_30 += input_14 * 3;
    acc0_30 += input_15 * -345;
    acc0_30 += (dense_accum_t)(42 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_30 = acc0_30 >> (FRAC_DEFAULT);
    tmp_relu0_30 = (default_t)acc0_30;

    /* RELU ACTIVATION */
    z0[30] = tmp_relu0_30 > 0 ? tmp_relu0_30 : 0;

    /* ReLU Layer Iteration: 31 */
    acc0_31 = 0;
    /* Unrolled Dot Product */
    acc0_31 += input_0 * 339;
    acc0_31 += input_1 * 71;
    acc0_31 += input_2 * -445;
    acc0_31 += input_3 * -301;
    acc0_31 += input_4 * -386;
    acc0_31 += input_5 * 16;
    acc0_31 += input_6 * 11;
    acc0_31 += input_7 * -63;
    acc0_31 += input_8 * 100;
    acc0_31 += input_9 * 1;
    acc0_31 += input_10 * 251;
    acc0_31 += input_11 * -108;
    acc0_31 += input_12 * -134;
    acc0_31 += input_13 * -122;
    acc0_31 += input_14 * 203;
    acc0_31 += input_15 * -68;
    acc0_31 += (dense_accum_t)(196 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_31 = acc0_31 >> (FRAC_DEFAULT);
    tmp_relu0_31 = (default_t)acc0_31;

    /* RELU ACTIVATION */
    z0[31] = tmp_relu0_31 > 0 ? tmp_relu0_31 : 0;

    /* ReLU Layer Iteration: 32 */
    acc0_32 = 0;
    /* Unrolled Dot Product */
    acc0_32 += input_0 * 31;
    acc0_32 += input_1 * 65;
    acc0_32 += input_2 * 0;
    acc0_32 += input_3 * 0;
    acc0_32 += input_4 * 0;
    acc0_32 += input_5 * -1;
    acc0_32 += input_6 * 110;
    acc0_32 += input_7 * 45;
    acc0_32 += input_8 * -1;
    acc0_32 += input_9 * 0;
    acc0_32 += input_10 * 0;
    acc0_32 += input_11 * -189;
    acc0_32 += input_12 * -43;
    acc0_32 += input_13 * -115;
    acc0_32 += input_14 * 0;
    acc0_32 += input_15 * -2;
    acc0_32 += -(dense_accum_t)(98 << (FRAC_DEFAULT));

    acc0_32 = acc0_32 >> (FRAC_DEFAULT);
    tmp_relu0_32 = (default_t)acc0_32;

    /* RELU ACTIVATION */
    z0[32] = tmp_relu0_32 > 0 ? tmp_relu0_32 : 0;

    /* ReLU Layer Iteration: 33 */
    acc0_33 = 0;
    /* Unrolled Dot Product */
    acc0_33 += input_0 * 57;
    acc0_33 += input_1 * 664;
    acc0_33 += input_2 * 406;
    acc0_33 += input_3 * -48;
    acc0_33 += input_4 * -1;
    acc0_33 += input_5 * 75;
    acc0_33 += input_6 * -130;
    acc0_33 += input_7 * 11;
    acc0_33 += input_8 * 165;
    acc0_33 += input_9 * -153;
    acc0_33 += input_10 * 139;
    acc0_33 += input_11 * -38;
    acc0_33 += input_12 * 134;
    acc0_33 += input_13 * -5;
    acc0_33 += input_14 * -91;
    acc0_33 += input_15 * -147;
    acc0_33 += (dense_accum_t)(87 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_33 = acc0_33 >> (FRAC_DEFAULT);
    tmp_relu0_33 = (default_t)acc0_33;

    /* RELU ACTIVATION */
    z0[33] = tmp_relu0_33 > 0 ? tmp_relu0_33 : 0;

    /* ReLU Layer Iteration: 34 */
    acc0_34 = 0;
    /* Unrolled Dot Product */
    acc0_34 += input_0 * -92;
    acc0_34 += input_1 * -15;
    acc0_34 += input_2 * 347;
    acc0_34 += input_3 * 408;
    acc0_34 += input_4 * 544;
    acc0_34 += input_5 * -355;
    acc0_34 += input_6 * -366;
    acc0_34 += input_7 * 197;
    acc0_34 += input_8 * 260;
    acc0_34 += input_9 * -56;
    acc0_34 += input_10 * -368;
    acc0_34 += input_11 * -154;
    acc0_34 += input_12 * 5;
    acc0_34 += input_13 * 79;
    acc0_34 += input_14 * -689;
    acc0_34 += input_15 * 53;
    acc0_34 += (dense_accum_t)(121 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_34 = acc0_34 >> (FRAC_DEFAULT);
    tmp_relu0_34 = (default_t)acc0_34;

    /* RELU ACTIVATION */
    z0[34] = tmp_relu0_34 > 0 ? tmp_relu0_34 : 0;

    /* ReLU Layer Iteration: 35 */
    acc0_35 = 0;
    /* Unrolled Dot Product */
    acc0_35 += input_0 * -44;
    acc0_35 += input_1 * 6;
    acc0_35 += input_2 * -172;
    acc0_35 += input_3 * -21;
    acc0_35 += input_4 * 241;
    acc0_35 += input_5 * 132;
    acc0_35 += input_6 * -31;
    acc0_35 += input_7 * -174;
    acc0_35 += input_8 * -3;
    acc0_35 += input_9 * 249;
    acc0_35 += input_10 * 2;
    acc0_35 += input_11 * -234;
    acc0_35 += input_12 * 0;
    acc0_35 += input_13 * -5;
    acc0_35 += input_14 * 56;
    acc0_35 += input_15 * -343;
    acc0_35 += -(dense_accum_t)(7 << (FRAC_DEFAULT));

    acc0_35 = acc0_35 >> (FRAC_DEFAULT);
    tmp_relu0_35 = (default_t)acc0_35;

    /* RELU ACTIVATION */
    z0[35] = tmp_relu0_35 > 0 ? tmp_relu0_35 : 0;

    /* ReLU Layer Iteration: 36 */
    acc0_36 = 0;
    /* Unrolled Dot Product */
    acc0_36 += input_0 * -377;
    acc0_36 += input_1 * -20;
    acc0_36 += input_2 * 329;
    acc0_36 += input_3 * -56;
    acc0_36 += input_4 * 561;
    acc0_36 += input_5 * -54;
    acc0_36 += input_6 * 34;
    acc0_36 += input_7 * 293;
    acc0_36 += input_8 * -156;
    acc0_36 += input_9 * -51;
    acc0_36 += input_10 * -382;
    acc0_36 += input_11 * -332;
    acc0_36 += input_12 * -86;
    acc0_36 += input_13 * -388;
    acc0_36 += input_14 * -720;
    acc0_36 += input_15 * 322;
    acc0_36 += (dense_accum_t)(77 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_36 = acc0_36 >> (FRAC_DEFAULT);
    tmp_relu0_36 = (default_t)acc0_36;

    /* RELU ACTIVATION */
    z0[36] = tmp_relu0_36 > 0 ? tmp_relu0_36 : 0;

    /* ReLU Layer Iteration: 37 */
    acc0_37 = 0;
    /* Unrolled Dot Product */
    acc0_37 += input_0 * 7;
    acc0_37 += input_1 * -233;
    acc0_37 += input_2 * -7;
    acc0_37 += input_3 * -156;
    acc0_37 += input_4 * -26;
    acc0_37 += input_5 * -185;
    acc0_37 += input_6 * 36;
    acc0_37 += input_7 * 0;
    acc0_37 += input_8 * -274;
    acc0_37 += input_9 * 321;
    acc0_37 += input_10 * -208;
    acc0_37 += input_11 * 326;
    acc0_37 += input_12 * -229;
    acc0_37 += input_13 * -207;
    acc0_37 += input_14 * 59;
    acc0_37 += input_15 * -291;
    acc0_37 += (dense_accum_t)(209 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_37 = acc0_37 >> (FRAC_DEFAULT);
    tmp_relu0_37 = (default_t)acc0_37;

    /* RELU ACTIVATION */
    z0[37] = tmp_relu0_37 > 0 ? tmp_relu0_37 : 0;

    /* ReLU Layer Iteration: 38 */
    acc0_38 = 0;
    /* Unrolled Dot Product */
    acc0_38 += input_0 * 1;
    acc0_38 += input_1 * 392;
    acc0_38 += input_2 * -59;
    acc0_38 += input_3 * -293;
    acc0_38 += input_4 * -10;
    acc0_38 += input_5 * 76;
    acc0_38 += input_6 * 36;
    acc0_38 += input_7 * 202;
    acc0_38 += input_8 * 223;
    acc0_38 += input_9 * 3;
    acc0_38 += input_10 * -204;
    acc0_38 += input_11 * -147;
    acc0_38 += input_12 * 8;
    acc0_38 += input_13 * -310;
    acc0_38 += input_14 * 491;
    acc0_38 += input_15 * 321;
    acc0_38 += -(dense_accum_t)(232 << (FRAC_DEFAULT));

    acc0_38 = acc0_38 >> (FRAC_DEFAULT);
    tmp_relu0_38 = (default_t)acc0_38;

    /* RELU ACTIVATION */
    z0[38] = tmp_relu0_38 > 0 ? tmp_relu0_38 : 0;

    /* ReLU Layer Iteration: 39 */
    acc0_39 = 0;
    /* Unrolled Dot Product */
    acc0_39 += input_0 * -4;
    acc0_39 += input_1 * -234;
    acc0_39 += input_2 * -129;
    acc0_39 += input_3 * 505;
    acc0_39 += input_4 * -281;
    acc0_39 += input_5 * 154;
    acc0_39 += input_6 * -1;
    acc0_39 += input_7 * 158;
    acc0_39 += input_8 * 176;
    acc0_39 += input_9 * -141;
    acc0_39 += input_10 * -7;
    acc0_39 += input_11 * -80;
    acc0_39 += input_12 * 344;
    acc0_39 += input_13 * -348;
    acc0_39 += input_14 * -166;
    acc0_39 += input_15 * 274;
    acc0_39 += -(dense_accum_t)(1 << (FRAC_DEFAULT));

    acc0_39 = acc0_39 >> (FRAC_DEFAULT);
    tmp_relu0_39 = (default_t)acc0_39;

    /* RELU ACTIVATION */
    z0[39] = tmp_relu0_39 > 0 ? tmp_relu0_39 : 0;

    /* ReLU Layer Iteration: 40 */
    acc0_40 = 0;
    /* Unrolled Dot Product */
    acc0_40 += input_0 * 123;
    acc0_40 += input_1 * 203;
    acc0_40 += input_2 * 7;
    acc0_40 += input_3 * -82;
    acc0_40 += input_4 * 576;
    acc0_40 += input_5 * 0;
    acc0_40 += input_6 * -226;
    acc0_40 += input_7 * -301;
    acc0_40 += input_8 * 273;
    acc0_40 += input_9 * 80;
    acc0_40 += input_10 * 18;
    acc0_40 += input_11 * -102;
    acc0_40 += input_12 * -183;
    acc0_40 += input_13 * 165;
    acc0_40 += input_14 * -1270;
    acc0_40 += input_15 * -166;
    acc0_40 += (dense_accum_t)(463 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_40 = acc0_40 >> (FRAC_DEFAULT);
    tmp_relu0_40 = (default_t)acc0_40;

    /* RELU ACTIVATION */
    z0[40] = tmp_relu0_40 > 0 ? tmp_relu0_40 : 0;

    /* ReLU Layer Iteration: 41 */
    acc0_41 = 0;
    /* Unrolled Dot Product */
    acc0_41 += input_0 * -2;
    acc0_41 += input_1 * -7;
    acc0_41 += input_2 * 169;
    acc0_41 += input_3 * -2;
    acc0_41 += input_4 * 364;
    acc0_41 += input_5 * -81;
    acc0_41 += input_6 * 126;
    acc0_41 += input_7 * -155;
    acc0_41 += input_8 * -119;
    acc0_41 += input_9 * 345;
    acc0_41 += input_10 * -119;
    acc0_41 += input_11 * 375;
    acc0_41 += input_12 * -72;
    acc0_41 += input_13 * -131;
    acc0_41 += input_14 * -677;
    acc0_41 += input_15 * -377;
    acc0_41 += (dense_accum_t)(28 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_41 = acc0_41 >> (FRAC_DEFAULT);
    tmp_relu0_41 = (default_t)acc0_41;

    /* RELU ACTIVATION */
    z0[41] = tmp_relu0_41 > 0 ? tmp_relu0_41 : 0;

    /* ReLU Layer Iteration: 42 */
    acc0_42 = 0;
    /* Unrolled Dot Product */
    acc0_42 += input_0 * 0;
    acc0_42 += input_1 * -86;
    acc0_42 += input_2 * 544;
    acc0_42 += input_3 * 157;
    acc0_42 += input_4 * 618;
    acc0_42 += input_5 * 120;
    acc0_42 += input_6 * -28;
    acc0_42 += input_7 * 122;
    acc0_42 += input_8 * -178;
    acc0_42 += input_9 * 35;
    acc0_42 += input_10 * -234;
    acc0_42 += input_11 * 24;
    acc0_42 += input_12 * -290;
    acc0_42 += input_13 * 16;
    acc0_42 += input_14 * 449;
    acc0_42 += input_15 * 41;
    acc0_42 += (dense_accum_t)(205 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_42 = acc0_42 >> (FRAC_DEFAULT);
    tmp_relu0_42 = (default_t)acc0_42;

    /* RELU ACTIVATION */
    z0[42] = tmp_relu0_42 > 0 ? tmp_relu0_42 : 0;

    /* ReLU Layer Iteration: 43 */
    acc0_43 = 0;
    /* Unrolled Dot Product */
    acc0_43 += input_0 * 78;
    acc0_43 += input_1 * -19;
    acc0_43 += input_2 * -635;
    acc0_43 += input_3 * 0;
    acc0_43 += input_4 * -314;
    acc0_43 += input_5 * -72;
    acc0_43 += input_6 * -219;
    acc0_43 += input_7 * -350;
    acc0_43 += input_8 * 157;
    acc0_43 += input_9 * -254;
    acc0_43 += input_10 * 204;
    acc0_43 += input_11 * -161;
    acc0_43 += input_12 * 287;
    acc0_43 += input_13 * 137;
    acc0_43 += input_14 * -211;
    acc0_43 += input_15 * 292;
    acc0_43 += -(dense_accum_t)(46 << (FRAC_DEFAULT));

    acc0_43 = acc0_43 >> (FRAC_DEFAULT);
    tmp_relu0_43 = (default_t)acc0_43;

    /* RELU ACTIVATION */
    z0[43] = tmp_relu0_43 > 0 ? tmp_relu0_43 : 0;

    /* ReLU Layer Iteration: 44 */
    acc0_44 = 0;
    /* Unrolled Dot Product */
    acc0_44 += input_0 * 231;
    acc0_44 += input_1 * -364;
    acc0_44 += input_2 * -248;
    acc0_44 += input_3 * 522;
    acc0_44 += input_4 * -33;
    acc0_44 += input_5 * -126;
    acc0_44 += input_6 * 68;
    acc0_44 += input_7 * 49;
    acc0_44 += input_8 * 23;
    acc0_44 += input_9 * -228;
    acc0_44 += input_10 * -68;
    acc0_44 += input_11 * -26;
    acc0_44 += input_12 * 161;
    acc0_44 += input_13 * -4;
    acc0_44 += input_14 * 111;
    acc0_44 += input_15 * 158;
    acc0_44 += -(dense_accum_t)(38 << (FRAC_DEFAULT));

    acc0_44 = acc0_44 >> (FRAC_DEFAULT);
    tmp_relu0_44 = (default_t)acc0_44;

    /* RELU ACTIVATION */
    z0[44] = tmp_relu0_44 > 0 ? tmp_relu0_44 : 0;

    /* ReLU Layer Iteration: 45 */
    acc0_45 = 0;
    /* Unrolled Dot Product */
    acc0_45 += input_0 * 122;
    acc0_45 += input_1 * 253;
    acc0_45 += input_2 * 0;
    acc0_45 += input_3 * -9;
    acc0_45 += input_4 * 450;
    acc0_45 += input_5 * -279;
    acc0_45 += input_6 * -29;
    acc0_45 += input_7 * 161;
    acc0_45 += input_8 * -172;
    acc0_45 += input_9 * 14;
    acc0_45 += input_10 * -108;
    acc0_45 += input_11 * -117;
    acc0_45 += input_12 * -152;
    acc0_45 += input_13 * 286;
    acc0_45 += input_14 * 1548;
    acc0_45 += input_15 * -163;
    acc0_45 += (dense_accum_t)(71 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_45 = acc0_45 >> (FRAC_DEFAULT);
    tmp_relu0_45 = (default_t)acc0_45;

    /* RELU ACTIVATION */
    z0[45] = tmp_relu0_45 > 0 ? tmp_relu0_45 : 0;

    /* ReLU Layer Iteration: 46 */
    acc0_46 = 0;
    /* Unrolled Dot Product */
    acc0_46 += input_0 * 168;
    acc0_46 += input_1 * 51;
    acc0_46 += input_2 * 627;
    acc0_46 += input_3 * 390;
    acc0_46 += input_4 * 323;
    acc0_46 += input_5 * -175;
    acc0_46 += input_6 * -30;
    acc0_46 += input_7 * -168;
    acc0_46 += input_8 * 217;
    acc0_46 += input_9 * -324;
    acc0_46 += input_10 * -25;
    acc0_46 += input_11 * -88;
    acc0_46 += input_12 * 0;
    acc0_46 += input_13 * -283;
    acc0_46 += input_14 * 6;
    acc0_46 += input_15 * -6;
    acc0_46 += (dense_accum_t)(101 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_46 = acc0_46 >> (FRAC_DEFAULT);
    tmp_relu0_46 = (default_t)acc0_46;

    /* RELU ACTIVATION */
    z0[46] = tmp_relu0_46 > 0 ? tmp_relu0_46 : 0;

    /* ReLU Layer Iteration: 47 */
    acc0_47 = 0;
    /* Unrolled Dot Product */
    acc0_47 += input_0 * -291;
    acc0_47 += input_1 * 191;
    acc0_47 += input_2 * 315;
    acc0_47 += input_3 * -157;
    acc0_47 += input_4 * 2;
    acc0_47 += input_5 * -1;
    acc0_47 += input_6 * 8;
    acc0_47 += input_7 * 97;
    acc0_47 += input_8 * -104;
    acc0_47 += input_9 * 101;
    acc0_47 += input_10 * -208;
    acc0_47 += input_11 * -128;
    acc0_47 += input_12 * -303;
    acc0_47 += input_13 * 2;
    acc0_47 += input_14 * -902;
    acc0_47 += input_15 * -7;
    acc0_47 += (dense_accum_t)(333 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_47 = acc0_47 >> (FRAC_DEFAULT);
    tmp_relu0_47 = (default_t)acc0_47;

    /* RELU ACTIVATION */
    z0[47] = tmp_relu0_47 > 0 ? tmp_relu0_47 : 0;

    /* ReLU Layer Iteration: 48 */
    acc0_48 = 0;
    /* Unrolled Dot Product */
    acc0_48 += input_0 * -118;
    acc0_48 += input_1 * -212;
    acc0_48 += input_2 * 369;
    acc0_48 += input_3 * 41;
    acc0_48 += input_4 * 84;
    acc0_48 += input_5 * 335;
    acc0_48 += input_6 * 29;
    acc0_48 += input_7 * 0;
    acc0_48 += input_8 * 51;
    acc0_48 += input_9 * -359;
    acc0_48 += input_10 * 52;
    acc0_48 += input_11 * 315;
    acc0_48 += input_12 * -337;
    acc0_48 += input_13 * -246;
    acc0_48 += input_14 * -644;
    acc0_48 += input_15 * -51;
    acc0_48 += (dense_accum_t)(169 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_48 = acc0_48 >> (FRAC_DEFAULT);
    tmp_relu0_48 = (default_t)acc0_48;

    /* RELU ACTIVATION */
    z0[48] = tmp_relu0_48 > 0 ? tmp_relu0_48 : 0;

    /* ReLU Layer Iteration: 49 */
    acc0_49 = 0;
    /* Unrolled Dot Product */
    acc0_49 += input_0 * 216;
    acc0_49 += input_1 * 169;
    acc0_49 += input_2 * -10;
    acc0_49 += input_3 * 77;
    acc0_49 += input_4 * 238;
    acc0_49 += input_5 * -1;
    acc0_49 += input_6 * 5;
    acc0_49 += input_7 * -246;
    acc0_49 += input_8 * 27;
    acc0_49 += input_9 * -63;
    acc0_49 += input_10 * -73;
    acc0_49 += input_11 * 155;
    acc0_49 += input_12 * 87;
    acc0_49 += input_13 * -1;
    acc0_49 += input_14 * 1014;
    acc0_49 += input_15 * -18;
    acc0_49 += (dense_accum_t)(164 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_49 = acc0_49 >> (FRAC_DEFAULT);
    tmp_relu0_49 = (default_t)acc0_49;

    /* RELU ACTIVATION */
    z0[49] = tmp_relu0_49 > 0 ? tmp_relu0_49 : 0;

    /* ReLU Layer Iteration: 50 */
    acc0_50 = 0;
    /* Unrolled Dot Product */
    acc0_50 += input_0 * 44;
    acc0_50 += input_1 * -1;
    acc0_50 += input_2 * -686;
    acc0_50 += input_3 * -612;
    acc0_50 += input_4 * -196;
    acc0_50 += input_5 * -48;
    acc0_50 += input_6 * -143;
    acc0_50 += input_7 * 168;
    acc0_50 += input_8 * -166;
    acc0_50 += input_9 * 258;
    acc0_50 += input_10 * -12;
    acc0_50 += input_11 * -41;
    acc0_50 += input_12 * -190;
    acc0_50 += input_13 * -20;
    acc0_50 += input_14 * 406;
    acc0_50 += input_15 * 457;
    acc0_50 += (dense_accum_t)(47 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_50 = acc0_50 >> (FRAC_DEFAULT);
    tmp_relu0_50 = (default_t)acc0_50;

    /* RELU ACTIVATION */
    z0[50] = tmp_relu0_50 > 0 ? tmp_relu0_50 : 0;

    /* ReLU Layer Iteration: 51 */
    acc0_51 = 0;
    /* Unrolled Dot Product */
    acc0_51 += input_0 * 101;
    acc0_51 += input_1 * 15;
    acc0_51 += input_2 * -183;
    acc0_51 += input_3 * -37;
    acc0_51 += input_4 * -82;
    acc0_51 += input_5 * 63;
    acc0_51 += input_6 * -20;
    acc0_51 += input_7 * 248;
    acc0_51 += input_8 * -19;
    acc0_51 += input_9 * -1;
    acc0_51 += input_10 * 279;
    acc0_51 += input_11 * -531;
    acc0_51 += input_12 * -213;
    acc0_51 += input_13 * -325;
    acc0_51 += input_14 * 276;
    acc0_51 += input_15 * 295;
    acc0_51 += (dense_accum_t)(61 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_51 = acc0_51 >> (FRAC_DEFAULT);
    tmp_relu0_51 = (default_t)acc0_51;

    /* RELU ACTIVATION */
    z0[51] = tmp_relu0_51 > 0 ? tmp_relu0_51 : 0;

    /* ReLU Layer Iteration: 52 */
    acc0_52 = 0;
    /* Unrolled Dot Product */
    acc0_52 += input_0 * -388;
    acc0_52 += input_1 * -242;
    acc0_52 += input_2 * 19;
    acc0_52 += input_3 * 275;
    acc0_52 += input_4 * 14;
    acc0_52 += input_5 * 46;
    acc0_52 += input_6 * -5;
    acc0_52 += input_7 * -160;
    acc0_52 += input_8 * 186;
    acc0_52 += input_9 * 318;
    acc0_52 += input_10 * 2;
    acc0_52 += input_11 * -40;
    acc0_52 += input_12 * 317;
    acc0_52 += input_13 * -152;
    acc0_52 += input_14 * -821;
    acc0_52 += input_15 * 368;
    acc0_52 += (dense_accum_t)(218 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_52 = acc0_52 >> (FRAC_DEFAULT);
    tmp_relu0_52 = (default_t)acc0_52;

    /* RELU ACTIVATION */
    z0[52] = tmp_relu0_52 > 0 ? tmp_relu0_52 : 0;

    /* ReLU Layer Iteration: 53 */
    acc0_53 = 0;
    /* Unrolled Dot Product */
    acc0_53 += input_0 * -298;
    acc0_53 += input_1 * -120;
    acc0_53 += input_2 * -446;
    acc0_53 += input_3 * -597;
    acc0_53 += input_4 * -474;
    acc0_53 += input_5 * 0;
    acc0_53 += input_6 * -11;
    acc0_53 += input_7 * -193;
    acc0_53 += input_8 * 306;
    acc0_53 += input_9 * -333;
    acc0_53 += input_10 * -244;
    acc0_53 += input_11 * -96;
    acc0_53 += input_12 * 50;
    acc0_53 += input_13 * -11;
    acc0_53 += input_14 * -154;
    acc0_53 += input_15 * -120;
    acc0_53 += (dense_accum_t)(153 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_53 = acc0_53 >> (FRAC_DEFAULT);
    tmp_relu0_53 = (default_t)acc0_53;

    /* RELU ACTIVATION */
    z0[53] = tmp_relu0_53 > 0 ? tmp_relu0_53 : 0;

    /* ReLU Layer Iteration: 54 */
    acc0_54 = 0;
    /* Unrolled Dot Product */
    acc0_54 += input_0 * 11;
    acc0_54 += input_1 * 2;
    acc0_54 += input_2 * -126;
    acc0_54 += input_3 * 3;
    acc0_54 += input_4 * 3;
    acc0_54 += input_5 * 16;
    acc0_54 += input_6 * 0;
    acc0_54 += input_7 * 184;
    acc0_54 += input_8 * 153;
    acc0_54 += input_9 * 0;
    acc0_54 += input_10 * 138;
    acc0_54 += input_11 * 6;
    acc0_54 += input_12 * 392;
    acc0_54 += input_13 * 12;
    acc0_54 += input_14 * -96;
    acc0_54 += input_15 * -15;
    acc0_54 += -(dense_accum_t)(83 << (FRAC_DEFAULT));

    acc0_54 = acc0_54 >> (FRAC_DEFAULT);
    tmp_relu0_54 = (default_t)acc0_54;

    /* RELU ACTIVATION */
    z0[54] = tmp_relu0_54 > 0 ? tmp_relu0_54 : 0;

    /* ReLU Layer Iteration: 55 */
    acc0_55 = 0;
    /* Unrolled Dot Product */
    acc0_55 += input_0 * -238;
    acc0_55 += input_1 * 168;
    acc0_55 += input_2 * -2;
    acc0_55 += input_3 * -138;
    acc0_55 += input_4 * -319;
    acc0_55 += input_5 * -34;
    acc0_55 += input_6 * -149;
    acc0_55 += input_7 * -3;
    acc0_55 += input_8 * 355;
    acc0_55 += input_9 * -138;
    acc0_55 += input_10 * 107;
    acc0_55 += input_11 * 62;
    acc0_55 += input_12 * 348;
    acc0_55 += input_13 * 182;
    acc0_55 += input_14 * 560;
    acc0_55 += input_15 * 296;
    acc0_55 += (dense_accum_t)(148 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_55 = acc0_55 >> (FRAC_DEFAULT);
    tmp_relu0_55 = (default_t)acc0_55;

    /* RELU ACTIVATION */
    z0[55] = tmp_relu0_55 > 0 ? tmp_relu0_55 : 0;

    /* ReLU Layer Iteration: 56 */
    acc0_56 = 0;
    /* Unrolled Dot Product */
    acc0_56 += input_0 * 211;
    acc0_56 += input_1 * 512;
    acc0_56 += input_2 * -91;
    acc0_56 += input_3 * -468;
    acc0_56 += input_4 * 42;
    acc0_56 += input_5 * -92;
    acc0_56 += input_6 * -3;
    acc0_56 += input_7 * -9;
    acc0_56 += input_8 * -76;
    acc0_56 += input_9 * 443;
    acc0_56 += input_10 * -56;
    acc0_56 += input_11 * -186;
    acc0_56 += input_12 * 93;
    acc0_56 += input_13 * -166;
    acc0_56 += input_14 * 394;
    acc0_56 += input_15 * 15;
    acc0_56 += (dense_accum_t)(293 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_56 = acc0_56 >> (FRAC_DEFAULT);
    tmp_relu0_56 = (default_t)acc0_56;

    /* RELU ACTIVATION */
    z0[56] = tmp_relu0_56 > 0 ? tmp_relu0_56 : 0;

    /* ReLU Layer Iteration: 57 */
    acc0_57 = 0;
    /* Unrolled Dot Product */
    acc0_57 += input_0 * -173;
    acc0_57 += input_1 * -160;
    acc0_57 += input_2 * 551;
    acc0_57 += input_3 * 632;
    acc0_57 += input_4 * 454;
    acc0_57 += input_5 * 0;
    acc0_57 += input_6 * 270;
    acc0_57 += input_7 * -12;
    acc0_57 += input_8 * -211;
    acc0_57 += input_9 * 81;
    acc0_57 += input_10 * 141;
    acc0_57 += input_11 * -103;
    acc0_57 += input_12 * -350;
    acc0_57 += input_13 * 47;
    acc0_57 += input_14 * 663;
    acc0_57 += input_15 * -165;
    acc0_57 += -(dense_accum_t)(96 << (FRAC_DEFAULT));

    acc0_57 = acc0_57 >> (FRAC_DEFAULT);
    tmp_relu0_57 = (default_t)acc0_57;

    /* RELU ACTIVATION */
    z0[57] = tmp_relu0_57 > 0 ? tmp_relu0_57 : 0;

    /* ReLU Layer Iteration: 58 */
    acc0_58 = 0;
    /* Unrolled Dot Product */
    acc0_58 += input_0 * 183;
    acc0_58 += input_1 * 318;
    acc0_58 += input_2 * 422;
    acc0_58 += input_3 * 200;
    acc0_58 += input_4 * 252;
    acc0_58 += input_5 * -286;
    acc0_58 += input_6 * -8;
    acc0_58 += input_7 * -205;
    acc0_58 += input_8 * 267;
    acc0_58 += input_9 * -300;
    acc0_58 += input_10 * 534;
    acc0_58 += input_11 * 23;
    acc0_58 += input_12 * 25;
    acc0_58 += input_13 * -298;
    acc0_58 += input_14 * -331;
    acc0_58 += input_15 * -247;
    acc0_58 += (dense_accum_t)(268 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_58 = acc0_58 >> (FRAC_DEFAULT);
    tmp_relu0_58 = (default_t)acc0_58;

    /* RELU ACTIVATION */
    z0[58] = tmp_relu0_58 > 0 ? tmp_relu0_58 : 0;

    /* ReLU Layer Iteration: 59 */
    acc0_59 = 0;
    /* Unrolled Dot Product */
    acc0_59 += input_0 * 346;
    acc0_59 += input_1 * 237;
    acc0_59 += input_2 * 323;
    acc0_59 += input_3 * 48;
    acc0_59 += input_4 * 406;
    acc0_59 += input_5 * -375;
    acc0_59 += input_6 * -270;
    acc0_59 += input_7 * 72;
    acc0_59 += input_8 * 41;
    acc0_59 += input_9 * -183;
    acc0_59 += input_10 * -195;
    acc0_59 += input_11 * -178;
    acc0_59 += input_12 * 303;
    acc0_59 += input_13 * 174;
    acc0_59 += input_14 * -822;
    acc0_59 += input_15 * 208;
    acc0_59 += (dense_accum_t)(280 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_59 = acc0_59 >> (FRAC_DEFAULT);
    tmp_relu0_59 = (default_t)acc0_59;

    /* RELU ACTIVATION */
    z0[59] = tmp_relu0_59 > 0 ? tmp_relu0_59 : 0;

    /* ReLU Layer Iteration: 60 */
    acc0_60 = 0;
    /* Unrolled Dot Product */
    acc0_60 += input_0 * -111;
    acc0_60 += input_1 * -19;
    acc0_60 += input_2 * -147;
    acc0_60 += input_3 * 103;
    acc0_60 += input_4 * -13;
    acc0_60 += input_5 * -465;
    acc0_60 += input_6 * 273;
    acc0_60 += input_7 * -337;
    acc0_60 += input_8 * 221;
    acc0_60 += input_9 * 85;
    acc0_60 += input_10 * -62;
    acc0_60 += input_11 * 4;
    acc0_60 += input_12 * -230;
    acc0_60 += input_13 * 107;
    acc0_60 += input_14 * 703;
    acc0_60 += input_15 * 21;
    acc0_60 += -(dense_accum_t)(18 << (FRAC_DEFAULT));

    acc0_60 = acc0_60 >> (FRAC_DEFAULT);
    tmp_relu0_60 = (default_t)acc0_60;

    /* RELU ACTIVATION */
    z0[60] = tmp_relu0_60 > 0 ? tmp_relu0_60 : 0;

    /* ReLU Layer Iteration: 61 */
    acc0_61 = 0;
    /* Unrolled Dot Product */
    acc0_61 += input_0 * -195;
    acc0_61 += input_1 * 66;
    acc0_61 += input_2 * -41;
    acc0_61 += input_3 * 466;
    acc0_61 += input_4 * 262;
    acc0_61 += input_5 * -30;
    acc0_61 += input_6 * 170;
    acc0_61 += input_7 * -237;
    acc0_61 += input_8 * -90;
    acc0_61 += input_9 * 62;
    acc0_61 += input_10 * 0;
    acc0_61 += input_11 * -36;
    acc0_61 += input_12 * -1;
    acc0_61 += input_13 * -122;
    acc0_61 += input_14 * 987;
    acc0_61 += input_15 * -58;
    acc0_61 += -(dense_accum_t)(28 << (FRAC_DEFAULT));

    acc0_61 = acc0_61 >> (FRAC_DEFAULT);
    tmp_relu0_61 = (default_t)acc0_61;

    /* RELU ACTIVATION */
    z0[61] = tmp_relu0_61 > 0 ? tmp_relu0_61 : 0;

    /* ReLU Layer Iteration: 62 */
    acc0_62 = 0;
    /* Unrolled Dot Product */
    acc0_62 += input_0 * -273;
    acc0_62 += input_1 * 101;
    acc0_62 += input_2 * 408;
    acc0_62 += input_3 * -1;
    acc0_62 += input_4 * -31;
    acc0_62 += input_5 * -319;
    acc0_62 += input_6 * -34;
    acc0_62 += input_7 * -367;
    acc0_62 += input_8 * 0;
    acc0_62 += input_9 * 92;
    acc0_62 += input_10 * -177;
    acc0_62 += input_11 * 202;
    acc0_62 += input_12 * -397;
    acc0_62 += input_13 * 185;
    acc0_62 += input_14 * 581;
    acc0_62 += input_15 * -319;
    acc0_62 += (dense_accum_t)(72 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_62 = acc0_62 >> (FRAC_DEFAULT);
    tmp_relu0_62 = (default_t)acc0_62;

    /* RELU ACTIVATION */
    z0[62] = tmp_relu0_62 > 0 ? tmp_relu0_62 : 0;

    /* ReLU Layer Iteration: 63 */
    acc0_63 = 0;
    /* Unrolled Dot Product */
    acc0_63 += input_0 * -115;
    acc0_63 += input_1 * -12;
    acc0_63 += input_2 * 253;
    acc0_63 += input_3 * -2;
    acc0_63 += input_4 * 80;
    acc0_63 += input_5 * 0;
    acc0_63 += input_6 * -190;
    acc0_63 += input_7 * 270;
    acc0_63 += input_8 * 146;
    acc0_63 += input_9 * 255;
    acc0_63 += input_10 * -5;
    acc0_63 += input_11 * 60;
    acc0_63 += input_12 * -4;
    acc0_63 += input_13 * 193;
    acc0_63 += input_14 * 3;
    acc0_63 += input_15 * -547;
    acc0_63 += (dense_accum_t)(358 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc0_63 = acc0_63 >> (FRAC_DEFAULT);
    tmp_relu0_63 = (default_t)acc0_63;

    /* RELU ACTIVATION */
    z0[63] = tmp_relu0_63 > 0 ? tmp_relu0_63 : 0;

    // ===========================================================================
    // Layer 1: Dense ReLU
    dense_accum_t acc1;
    default_t tmp_relu1 = 0;
    for (int j = 0; j < N_LAYER_5; j++) {
        acc1 = 0;
        /* Unrolled Dot Product */
        for (int i = 0; i < N_LAYER_2; i++) {
            acc1 += z0[i] * w1[j][i];
        }
        acc1 += (dense_accum_t)(b1[j] << (FRAC_DEFAULT));

        /* TRUNCATE */
        acc1 = acc1 >> (FRAC_DEFAULT);
        tmp_relu1 = (default_t)acc1;

        /* RELU ACTIVATION */
        z1[j] = tmp_relu1 > 0 ? tmp_relu1 : 0;
    }

    // ===========================================================================
    // Layer 2: Dense ReLU
    dense_accum_t acc2;
    default_t tmp_relu2 = 0;
    for (int j = 0; j < N_LAYER_8; j++) {
        acc2 = 0;
        /* Unrolled Dot Product */
        for (int i = 0; i < N_LAYER_5; i++) {
            acc2 += z1[i] * w2[j][i];
        }
        acc2 += (dense_accum_t)(b2[j] << (FRAC_DEFAULT));

        /* TRUNCATE */
        acc2 = acc2 >> (FRAC_DEFAULT);
        tmp_relu2 = (default_t)acc2;

        /* RELU ACTIVATION */
        z2[j] = tmp_relu2 > 0 ? tmp_relu2 : 0;
    }

    // ===========================================================================
    // Layer 3: Dense Argmax
    dense_accum_t acc3;
    default_t tmp_max3 = -(1 << (NB_DEFAULT - 1));
    /* Argmax Layer Iteration: 0 */
    acc3 = 0;
    /* Unrolled Dot Product */
    acc3 += z2[0] * -396;
    acc3 += z2[1] * 42;
    acc3 += z2[2] * -266;
    acc3 += z2[3] * -16;
    acc3 += z2[4] * -163;
    acc3 += z2[5] * 9;
    acc3 += z2[6] * 571;
    acc3 += z2[7] * 255;
    acc3 += z2[8] * -242;
    acc3 += z2[9] * 81;
    acc3 += z2[10] * 127;
    acc3 += z2[11] * 62;
    acc3 += z2[12] * 39;
    acc3 += z2[13] * 441;
    acc3 += z2[14] * -53;
    acc3 += z2[15] * -454;
    acc3 += z2[16] * 401;
    acc3 += z2[17] * -423;
    acc3 += z2[18] * -125;
    acc3 += z2[19] * 61;
    acc3 += z2[20] * 253;
    acc3 += z2[21] * 342;
    acc3 += z2[22] * 617;
    acc3 += z2[23] * 1029;
    acc3 += z2[24] * 0;
    acc3 += z2[25] * -646;
    acc3 += z2[26] * 82;
    acc3 += z2[27] * -12;
    acc3 += z2[28] * 271;
    acc3 += z2[29] * 385;
    acc3 += z2[30] * -265;
    acc3 += z2[31] * -410;
    acc3 += -(dense_accum_t)(130 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc3 = acc3 >> (FRAC_DEFAULT);
    tmp_max3 = ((default_t)acc3 > tmp_max3) ? (default_t)acc3 : tmp_max3;
    z3[0] = (default_t)acc3;

    /* Argmax Layer Iteration: 1 */
    acc3 = 0;
    /* Unrolled Dot Product */
    acc3 += z2[0] * -314;
    acc3 += z2[1] * 71;
    acc3 += z2[2] * -380;
    acc3 += z2[3] * 246;
    acc3 += z2[4] * -40;
    acc3 += z2[5] * 60;
    acc3 += z2[6] * 270;
    acc3 += z2[7] * -18;
    acc3 += z2[8] * 253;
    acc3 += z2[9] * -246;
    acc3 += z2[10] * 476;
    acc3 += z2[11] * 327;
    acc3 += z2[12] * 250;
    acc3 += z2[13] * 4;
    acc3 += z2[14] * -349;
    acc3 += z2[15] * -246;
    acc3 += z2[16] * 284;
    acc3 += z2[17] * 257;
    acc3 += z2[18] * 115;
    acc3 += z2[19] * 199;
    acc3 += z2[20] * -500;
    acc3 += z2[21] * -141;
    acc3 += z2[22] * 422;
    acc3 += z2[23] * -322;
    acc3 += z2[24] * -136;
    acc3 += z2[25] * 138;
    acc3 += z2[26] * 192;
    acc3 += z2[27] * 0;
    acc3 += z2[28] * -331;
    acc3 += z2[29] * -326;
    acc3 += z2[30] * 12;
    acc3 += z2[31] * -42;
    acc3 += (dense_accum_t)(74 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc3 = acc3 >> (FRAC_DEFAULT);
    tmp_max3 = ((default_t)acc3 > tmp_max3) ? (default_t)acc3 : tmp_max3;
    z3[1] = (default_t)acc3;

    /* Argmax Layer Iteration: 2 */
    acc3 = 0;
    /* Unrolled Dot Product */
    acc3 += z2[0] * -405;
    acc3 += z2[1] * -430;
    acc3 += z2[2] * 232;
    acc3 += z2[3] * 556;
    acc3 += z2[4] * -170;
    acc3 += z2[5] * 538;
    acc3 += z2[6] * -204;
    acc3 += z2[7] * -364;
    acc3 += z2[8] * 308;
    acc3 += z2[9] * 175;
    acc3 += z2[10] * -17;
    acc3 += z2[11] * -263;
    acc3 += z2[12] * -224;
    acc3 += z2[13] * -93;
    acc3 += z2[14] * 162;
    acc3 += z2[15] * 314;
    acc3 += z2[16] * -26;
    acc3 += z2[17] * -356;
    acc3 += z2[18] * 265;
    acc3 += z2[19] * -411;
    acc3 += z2[20] * -253;
    acc3 += z2[21] * 291;
    acc3 += z2[22] * -631;
    acc3 += z2[23] * 323;
    acc3 += z2[24] * -17;
    acc3 += z2[25] * 425;
    acc3 += z2[26] * -72;
    acc3 += z2[27] * 101;
    acc3 += z2[28] * 55;
    acc3 += z2[29] * 46;
    acc3 += z2[30] * -244;
    acc3 += z2[31] * -269;
    acc3 += -(dense_accum_t)(95 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc3 = acc3 >> (FRAC_DEFAULT);
    tmp_max3 = ((default_t)acc3 > tmp_max3) ? (default_t)acc3 : tmp_max3;
    z3[2] = (default_t)acc3;

    /* Argmax Layer Iteration: 3 */
    acc3 = 0;
    /* Unrolled Dot Product */
    acc3 += z2[0] * 41;
    acc3 += z2[1] * -410;
    acc3 += z2[2] * 603;
    acc3 += z2[3] * 688;
    acc3 += z2[4] * -35;
    acc3 += z2[5] * 53;
    acc3 += z2[6] * -373;
    acc3 += z2[7] * 190;
    acc3 += z2[8] * -937;
    acc3 += z2[9] * 149;
    acc3 += z2[10] * -503;
    acc3 += z2[11] * -463;
    acc3 += z2[12] * 155;
    acc3 += z2[13] * -666;
    acc3 += z2[14] * 214;
    acc3 += z2[15] * 132;
    acc3 += z2[16] * 523;
    acc3 += z2[17] * 522;
    acc3 += z2[18] * -513;
    acc3 += z2[19] * -30;
    acc3 += z2[20] * 398;
    acc3 += z2[21] * -74;
    acc3 += z2[22] * 0;
    acc3 += z2[23] * -1560;
    acc3 += z2[24] * 0;
    acc3 += z2[25] * 202;
    acc3 += z2[26] * -268;
    acc3 += z2[27] * -201;
    acc3 += z2[28] * -448;
    acc3 += z2[29] * 152;
    acc3 += z2[30] * 461;
    acc3 += z2[31] * 231;
    acc3 += -(dense_accum_t)(88 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc3 = acc3 >> (FRAC_DEFAULT);
    tmp_max3 = ((default_t)acc3 > tmp_max3) ? (default_t)acc3 : tmp_max3;
    z3[3] = (default_t)acc3;

    /* Argmax Layer Iteration: 4 */
    acc3 = 0;
    /* Unrolled Dot Product */
    acc3 += z2[0] * 545;
    acc3 += z2[1] * -311;
    acc3 += z2[2] * -286;
    acc3 += z2[3] * -844;
    acc3 += z2[4] * 337;
    acc3 += z2[5] * -712;
    acc3 += z2[6] * -312;
    acc3 += z2[7] * -57;
    acc3 += z2[8] * 180;
    acc3 += z2[9] * -694;
    acc3 += z2[10] * 139;
    acc3 += z2[11] * 60;
    acc3 += z2[12] * 77;
    acc3 += z2[13] * -555;
    acc3 += z2[14] * 387;
    acc3 += z2[15] * 444;
    acc3 += z2[16] * -840;
    acc3 += z2[17] * -417;
    acc3 += z2[18] * 47;
    acc3 += z2[19] * 59;
    acc3 += z2[20] * 129;
    acc3 += z2[21] * -2;
    acc3 += z2[22] * 0;
    acc3 += z2[23] * -1235;
    acc3 += z2[24] * -14;
    acc3 += z2[25] * -175;
    acc3 += z2[26] * 438;
    acc3 += z2[27] * 186;
    acc3 += z2[28] * -64;
    acc3 += z2[29] * -747;
    acc3 += z2[30] * -430;
    acc3 += z2[31] * 326;
    acc3 += (dense_accum_t)(159 << (FRAC_DEFAULT));

    /* TRUNCATE */
    acc3 = acc3 >> (FRAC_DEFAULT);
    tmp_max3 = ((default_t)acc3 > tmp_max3) ? (default_t)acc3 : tmp_max3;
    z3[4] = (default_t)acc3;

    /* Argmax */
    z3[0] = (tmp_max3 == z3[0]) ? (1 << (FRAC_DEFAULT)) : 0;
    z3[1] = (tmp_max3 == z3[1]) ? (1 << (FRAC_DEFAULT)) : 0;
    z3[2] = (tmp_max3 == z3[2]) ? (1 << (FRAC_DEFAULT)) : 0;
    z3[3] = (tmp_max3 == z3[3]) ? (1 << (FRAC_DEFAULT)) : 0;
    z3[4] = (tmp_max3 == z3[4]) ? (1 << (FRAC_DEFAULT)) : 0;
}

int main(void) {
    default_t input[N_INPUT_1_1] = {
        -122, 415, -1065, -844, -773, -594, 2034, 1574, 2034, 646, 392, -206, 1085, 418, -1044, -184
    };

    default_t z0[N_LAYER_2];

    default_t w1[N_LAYER_5][N_LAYER_2] = {
        {87,204,40,-108,3,-77,-359,0,186,0,0,25,-187,0,288,-43,328,1,145,64,-3,97,63,-52,306,0,-61,-173,0,-118,108,185,-97,1,1,-2,1,228,-118,1,0,18,-188,-158,-22,-4,0,0,16,-2,-29,-2,-84,9,38,-32,79,-6,25,16,1,-31,2,33},
        {-3,-35,50,135,85,9,-2,0,-10,93,-6,126,2,3,0,56,-4,3,75,56,74,221,118,248,76,47,38,-57,-435,119,-126,107,10,61,-28,0,91,-101,1,105,-96,-134,-3,-1,68,-120,-17,-75,58,-140,331,274,35,-119,161,237,244,62,-1,1,-118,-89,-100,-125},
        {-42,1,2,33,-44,70,0,0,-50,-25,2,107,127,-60,48,16,-217,57,1,-44,-89,209,5,106,-77,-97,164,2,0,151,-141,-36,1,168,0,-1,187,9,93,274,-226,-84,-6,308,44,224,140,-112,8,2,352,91,195,10,-25,186,-2,167,-15,-211,-58,19,-3,-4},
        {1,166,69,4,0,-36,234,0,3,-189,-4,-61,0,4,5,63,49,0,0,247,-6,2,4,-23,-1,-140,204,0,-196,136,121,-47,-15,207,107,-18,66,-1,0,48,0,67,-53,35,141,-153,136,-103,1,64,-19,0,-17,32,7,2,20,-1,282,110,1,111,-1,25},
        {114,90,17,-2,-69,3,-22,-2,0,-26,-126,-17,0,150,53,76,-90,0,-1,43,0,215,209,-41,1,-10,71,-191,0,-4,6,-1,-1,123,4,0,77,-2,-1,-3,2,6,-1,-2,-5,-55,71,-41,-3,0,-2,-3,178,-260,74,72,119,-11,243,2,0,-163,-7,123},
        {0,58,38,-221,7,145,-53,-216,49,178,142,40,-221,-5,313,-89,-68,-114,-53,189,95,-1,0,-7,0,0,131,0,303,-55,-95,2,-41,0,203,-2,162,5,0,-298,83,25,1,2,-39,8,115,140,112,-28,-134,-144,-410,73,0,-11,-237,191,18,-23,52,-105,106,116},
        {-1,167,1,72,0,12,4,0,-120,145,7,-38,-34,10,0,-99,-167,2,-3,174,-119,38,0,0,-4,216,223,1,-173,-5,65,-8,-1,226,-164,0,-57,80,1,98,6,1,-10,13,1,0,-1,59,79,-7,1,-2,6,0,150,2,-1,5,-4,123,-73,5,-56,3},
        {18,-301,0,176,-21,2,-24,-103,-1,65,5,14,50,-95,105,1,130,-2,44,2,131,99,-2,39,79,-88,114,150,1,-45,127,81,5,-5,0,1,198,-36,129,0,-87,-10,43,196,-258,91,-114,299,58,0,75,-1,4,199,-15,266,-50,131,-255,-288,-3,0,110,0},
        {23,23,0,48,0,138,-3,0,-40,-1,-1,144,0,34,37,-39,-2,-1,127,94,-155,177,-1,-7,-1,209,112,-8,77,-36,98,181,0,-2,0,3,16,0,-2,-1,246,-1,-62,60,-1,-503,-172,-3,1,-303,207,-2,-1,-1,-72,108,-1,0,-187,0,-3,-10,-6,0},
        {-13,34,8,-1,-42,0,253,0,128,-286,3,165,2,0,1,0,94,0,43,-1,38,2,0,123,-178,-236,0,1,-2,70,-63,0,0,1,0,0,2,51,-2,-1,-393,0,4,-4,-2,374,0,-2,-57,163,0,-2,2,-2,-8,2,155,259,0,-280,56,1,189,0},
        {6,15,1,-537,45,-142,-303,1,50,1,-113,2,-8,3,130,13,0,-24,-30,4,0,1,7,-201,232,2,-178,-73,2,1,-45,30,2,-1,188,0,137,5,-50,-48,214,1,41,-433,2,-107,19,149,26,-73,-261,3,267,-7,2,-14,-2,0,72,145,-55,-228,4,134},
        {82,-18,-3,146,-239,-150,479,0,153,-313,-4,-35,-107,-125,2,233,113,-1,49,-59,330,1,4,84,-62,-346,1,-6,1,-3,1,-93,0,1,-29,1,-140,52,14,-6,-661,-126,107,1,108,423,228,-220,-254,62,138,0,-15,-202,4,2,115,140,31,-274,364,222,281,-72},
        {182,43,157,-12,18,-178,11,0,267,3,2,-18,-47,2,56,26,3,-26,2,-55,3,3,43,-287,63,0,-96,-246,-206,1,212,54,1,6,2,0,162,-90,1,-2,23,1,-62,0,0,1,233,150,2,3,0,2,57,36,25,-40,5,-318,305,487,1,0,119,14},
        {6,-35,3,-219,148,-5,215,-487,238,0,154,80,119,-118,64,26,9,-1,-57,0,107,0,-1,-20,1,-3,-4,0,-1,-139,1,0,100,71,1,171,9,11,-1,-23,-132,-3,255,2,-181,104,108,60,26,0,4,-33,0,-67,-160,-86,-152,141,16,0,-13,-138,4,8},
        {120,208,-87,-229,2,178,-872,0,285,313,-79,264,-112,2,40,61,111,49,-1,129,140,0,73,-315,324,78,-1,-503,1,1,112,86,0,27,17,23,19,127,-7,-18,130,0,-66,-84,-6,1,20,75,3,118,-94,0,143,0,14,96,28,-493,57,33,-86,-5,87,302},
        {144,70,-2,-477,-7,-263,-193,-249,420,-92,0,55,-2,-280,266,264,167,0,114,7,211,138,0,-77,161,-341,-7,-141,-4,-4,46,19,0,8,-25,136,-193,-1,3,-155,-448,-30,234,8,-4,6,-46,-33,-3,75,-217,2,-98,-113,1,118,43,-42,81,-186,47,90,83,158},
        {-23,17,129,-38,15,220,207,-55,-16,-102,-17,-204,-92,385,106,65,-536,0,-457,0,-116,43,59,0,4,-48,1,-261,97,94,-226,-30,-1,0,563,0,206,-141,-369,-2,390,108,256,29,-50,-356,82,230,210,0,-515,-98,260,-117,-39,-172,-161,-23,316,206,-89,-16,120,-4},
        {-200,87,151,-51,179,35,83,-57,1,103,-48,-9,-109,306,3,-135,-447,31,-400,66,34,47,299,-29,-3,50,-13,-31,-1,-132,-143,-160,-2,171,167,0,64,1,-3,40,409,229,14,-100,3,-207,223,148,330,53,-288,-88,116,-53,9,-145,-254,-6,342,394,-164,-29,112,0},
        {0,-81,1,0,-1,0,40,0,2,0,91,-73,2,2,-78,-3,-1,2,0,3,45,-62,0,178,0,1,0,3,2,0,0,0,0,2,1,0,0,0,145,3,0,0,59,0,2,117,0,3,0,1,0,0,1,0,3,-6,-1,198,0,0,24,22,-42,3},
        {-32,-86,-47,-1,-2,-2,-60,-358,1,-2,112,-180,2,-105,-1,-8,0,17,0,15,105,2,-168,168,0,-3,-2,-30,0,-131,-1,-2,0,-14,-39,232,-179,0,4,-331,-3,0,52,-1,-554,296,-44,335,197,197,4,-11,-188,-5,-210,-46,-143,49,-86,-10,-3,1,110,202},
        {-44,0,120,0,8,-135,-30,-549,167,0,2,283,0,-62,57,112,45,-65,135,-8,73,49,-2,118,98,2,-66,16,85,-20,-289,-116,1,-1,26,27,143,6,-78,-73,1,-1,260,-77,-340,341,24,277,246,80,-66,-1,-27,33,-113,142,0,32,-40,3,-88,-1,269,143},
        {20,3,-46,118,-2,-222,-62,0,0,-3,2,190,2,-44,-7,121,104,-51,306,11,-3,137,14,-62,-84,3,200,-34,-108,-63,43,48,3,66,-26,3,-205,-1,3,0,-245,-75,-6,-2,12,-3,-16,-40,-57,-63,100,1,-103,56,-14,97,166,-10,0,2,-114,-33,-7,4},
        {-76,3,10,-151,76,-40,148,-2,59,199,-157,-51,-9,317,63,55,-315,0,-303,65,-67,52,-5,3,322,164,-218,-162,213,115,118,-313,-1,-71,289,1,40,-171,-341,78,211,33,-112,-307,-4,-105,305,123,-3,11,-438,-34,266,-15,143,-32,-286,-3,273,310,-160,-140,-40,41},
        {322,-1,-19,53,-129,45,-46,79,-4,-2,-14,124,115,-4,-2,-10,459,3,335,0,94,65,-29,-164,-3,-77,1,50,0,99,136,194,2,-33,-81,-85,-269,71,0,1,-291,-300,0,16,32,239,-5,-90,-247,-75,224,117,-158,36,-1,134,198,-56,-269,-378,-4,146,-4,-26},
        {3,74,1,-12,-309,-153,93,0,179,-333,-3,20,-2,1,11,183,355,-19,267,45,90,0,29,46,-138,-333,-38,-38,0,-87,-85,-4,111,6,-114,2,20,4,0,-34,-700,-32,151,-12,-2,430,27,-236,-52,296,56,139,-18,-165,-33,-5,246,255,-1,-340,248,2,229,-94},
        {15,1,-20,0,94,-24,-1,0,2,153,0,88,86,69,24,-19,14,-72,200,145,-563,-27,157,1,181,246,42,0,-75,1,206,225,1,18,3,2,-81,22,57,1,168,0,-418,0,51,-402,-86,89,89,-587,-2,104,6,1,-4,29,21,-1,4,212,-191,-678,-281,34},
        {166,22,-212,80,-217,-33,-153,200,121,38,-77,-6,80,-370,0,51,568,0,1,56,209,0,-79,0,11,-88,-86,23,22,221,306,195,0,-105,-351,-44,-230,248,127,54,-109,-61,-65,2,0,57,15,-71,-106,242,82,0,4,159,63,261,248,2,-52,-301,77,89,-1,-2},
        {33,19,-151,-184,-21,-240,134,1,61,-33,0,-98,42,-28,-1,231,223,-2,47,9,242,32,4,0,-115,-349,1,-3,-40,112,-50,0,-35,-119,-338,39,-238,109,50,18,-555,-43,5,0,9,400,-114,-136,-277,49,13,136,16,-74,5,16,143,291,98,-288,244,316,213,0},
        {-153,-206,1,15,-1,16,-47,0,-246,0,-5,51,5,2,-21,201,-1,0,2,-18,-115,-5,1,35,107,117,76,136,0,33,4,0,5,0,84,3,242,-102,156,-2,5,0,-46,0,-2,4,54,18,0,2,214,58,157,15,8,2,2,-76,-94,0,-154,2,0,-64},
        {-19,80,194,34,-1,-185,21,0,1,-130,-174,33,2,103,0,309,-389,2,52,-1,-111,9,99,2,142,42,2,4,-105,16,0,63,24,262,168,0,-28,-3,-88,-1,118,-109,3,-17,85,-23,135,-5,-8,81,-269,13,-43,-220,3,92,277,-4,456,311,0,0,88,3},
        {-1,308,0,120,-4,-101,7,0,0,-1,1,0,-4,4,0,33,1,2,50,110,-531,283,150,1,73,59,61,0,-60,-29,113,0,0,188,-20,4,-5,3,0,-3,15,13,-75,-213,76,-582,4,0,3,-397,-182,-4,-328,-145,3,13,218,0,202,68,-202,-196,-1,4},
        {0,130,10,-24,93,2,-1096,0,143,84,1,-37,-77,-1,-44,0,0,1,-3,210,4,58,203,2,-2,223,225,0,0,-38,248,26,-4,10,-70,-1,-116,40,16,0,105,0,-277,83,2,24,-1,0,120,2,-89,0,-248,159,-60,99,231,-382,0,26,5,12,-71,0}
    };
    default_t b1[N_LAYER_5] = {
        320,-25,-98,145,122,85,-86,11,38,-35,331,79,454,139,358,287,3,109,-291,-77,122,-80,145,111,-64,-52,135,-93,-86,131,-21,94
    };
    default_t z1[N_LAYER_5];

    default_t w2[N_LAYER_8][N_LAYER_5] = {
        {212,-215,-4,0,0,224,-359,38,65,1,179,480,-53,78,477,100,-597,301,205,274,0,-413,185,2,-91,-16,317,545,-239,-257,-210,55},
        {31,-5,-39,318,37,-4,194,-136,147,-101,-4,136,0,-294,-48,-152,-40,-303,-91,-175,-59,0,-182,73,273,329,-48,-97,-8,220,45,312},
        {-470,-36,-8,0,-173,121,0,72,0,357,-1,216,-397,342,-141,0,44,2,0,182,199,-1,1,-82,439,0,-242,-80,389,-93,0,0},
        {319,201,5,12,-38,-5,78,111,-182,-440,212,-560,343,0,185,447,-457,146,0,0,7,35,60,-27,-966,-332,-188,-1202,112,165,-1,116},
        {243,-1,-129,0,0,9,-67,108,88,0,182,97,1,0,113,-12,-185,63,0,-4,0,-58,111,2,0,174,3,155,-1,-15,-2,-93},
        {3,199,10,21,-57,19,0,-183,-45,-182,104,-539,389,2,49,529,43,139,0,2,22,126,149,-150,-693,228,-92,-1143,54,312,3,204},
        {201,208,295,197,74,-53,13,-138,-140,368,-102,52,0,-316,169,-190,-2,-321,1,-47,-199,318,-666,288,1,265,203,-137,-5,231,205,0},
        {-2,229,-14,80,173,-37,257,296,240,-93,1,-50,-91,161,58,60,-98,-173,0,-7,-1,-3,-262,27,-146,4,3,-41,-1,1,-26,91},
        {-94,6,91,208,2,157,-94,102,2,-51,51,144,5,416,-477,303,425,309,-38,169,94,-138,153,-85,-135,2,-158,-78,0,21,1,-374},
        {-137,52,94,-8,-16,294,-1,197,0,283,-453,53,-328,12,-549,46,482,226,-5,41,362,0,-2,0,400,0,-132,0,0,22,0,0},
        {336,288,-74,371,115,180,305,-109,357,302,-193,491,63,-265,-342,-107,282,354,-267,-429,-3,1,-33,-153,343,357,0,181,-25,473,572,194},
        {-33,128,132,77,321,-226,2,-143,7,255,1,359,300,-111,-207,-15,-131,6,66,-474,5,0,52,0,240,105,179,218,-231,49,202,-170},
        {104,-204,1,299,348,94,2,-294,1,0,0,0,162,-149,0,-26,-1,-132,78,-122,47,-3,0,4,36,0,205,1,-17,100,-2,132},
        {225,149,292,120,1,-368,0,-1,271,74,-111,-138,201,-190,-248,-17,-105,-244,-237,-13,95,414,-635,406,0,-27,169,282,-1,119,358,-87},
        {159,-2,-1,119,-93,245,-182,282,4,0,476,-74,123,179,57,187,-234,-3,247,551,307,0,163,1,-244,-91,27,246,182,-15,-434,-242},
        {54,-136,-293,71,-515,94,-383,181,-371,124,278,217,323,-5,455,393,22,-156,-24,165,335,73,68,53,8,-112,-64,46,4,38,-321,-84},
        {-3,-7,95,331,3,-146,0,-42,-1,224,-734,457,-132,-140,-829,71,330,-142,-11,-57,-214,-1,-2,343,418,0,139,12,-5,396,0,-1},
        {326,-113,-268,0,6,63,-54,104,-209,-201,-76,-819,-60,-152,320,219,-1114,262,0,-4,-3,-2,-12,-22,-560,-381,262,-6,-30,-257,-71,359},
        {-301,0,38,114,1,8,0,-52,0,0,0,213,123,-74,-311,3,274,353,-210,184,337,2,21,-66,197,176,-81,-27,29,212,0,-295},
        {11,0,-1,4,82,206,163,106,27,-1,4,-79,6,2,35,-497,164,89,1,0,-69,-1,13,3,-1,7,-50,0,-2,0,117,238},
        {280,85,-103,-216,149,-168,229,0,3,-313,103,-494,380,49,199,605,26,-338,0,-1,319,-5,12,385,-280,-290,453,-146,0,4,-227,129},
        {-15,5,165,110,-6,-158,0,372,0,0,168,0,-109,1,11,-7,280,-98,-97,190,144,249,-107,0,1,187,-109,-85,196,-64,1,-121},
        {-67,-41,1,1,35,2,63,-67,37,0,1,1,-31,2,-4,-206,471,200,2,5,-161,0,373,-1,-47,-1,-151,3,-2,23,3,-2},
        {-172,-1,12,25,296,-14,165,-215,15,-1,-55,-164,43,282,-185,19,495,512,-2,-1,-66,0,617,-879,-6,120,-373,-131,192,331,-50,1},
        {0,0,0,0,0,0,0,70,111,0,0,0,0,0,53,0,0,0,0,0,5,0,0,0,0,0,44,0,0,0,0,0},
        {-15,-134,182,54,-46,367,42,-79,-36,93,-389,-74,23,258,215,226,155,225,221,93,236,-302,277,-234,-194,-102,-361,-334,215,261,137,202},
        {217,37,-2,208,62,182,0,0,179,-4,62,357,201,0,-38,-230,-4,68,0,-184,36,0,165,83,1,270,113,390,-1,0,236,-305},
        {11,0,-2,0,-1,58,-1,55,84,0,49,1,80,0,-46,1,0,0,-34,0,4,0,54,-243,0,170,-36,0,0,1,1,-109},
        {0,14,278,-63,109,-224,-1,450,208,22,188,-1,24,86,-57,-6,156,110,-6,3,198,120,-231,236,-38,287,-105,-1,244,224,2,-300},
        {-207,268,236,3,-3,0,-109,57,259,283,-208,139,-356,-130,-232,0,88,-253,1,-48,341,24,-648,263,59,286,310,321,40,12,0,124},
        {94,48,-1,-238,82,0,168,-105,18,-8,-1,-152,136,-174,469,149,-219,0,99,-1,-172,-32,75,100,-463,-507,55,-127,0,-1,-14,290},
        {281,-380,-237,172,-2,121,81,-121,114,0,263,-3,222,175,456,0,-331,188,194,-3,0,-372,417,75,151,-260,252,17,-205,-1,-196,-80}
    };
    default_t b2[N_LAYER_8] = {
        158,28,10,164,160,63,-98,70,368,-195,142,210,68,-36,45,147,-113,63,134,100,140,-106,-310,-56,-155,37,197,106,1,-54,28,131
    };
    default_t z2[N_LAYER_8];

    default_t z3[N_LAYER_11];

    CALL_KERNEL(jet_tagging_inline,
        input,
        z0,
        w1, b1, z1,
        w2, b2, z2,
        z3
    );
    return 0;
}
