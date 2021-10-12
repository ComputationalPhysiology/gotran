#include "math.h"

// Compute the right hand side of the maleckar_2009 ODE
void rhs(const double *__restrict states, const double t, const double *__restrict parameters,
         double *values)
{

    // Assign states
    const double m = states[0];
    const double h1 = states[1];
    const double h2 = states[2];
    const double d_L = states[3];
    const double f_L1 = states[4];
    const double f_L2 = states[5];
    const double r = states[6];
    const double s = states[7];
    const double a_ur = states[8];
    const double i_ur = states[9];
    const double n = states[10];
    const double pa = states[11];
    const double O_C = states[12];
    const double O_TC = states[13];
    const double O_TMgC = states[14];
    const double O_TMgMg = states[15];
    const double F1 = states[17];
    const double F2 = states[18];
    const double O_Calse = states[19];
    const double Ca_up = states[20];
    const double Ca_rel = states[21];
    const double Ca_c = states[22];
    const double K_c = states[23];
    const double Na_c = states[24];
    const double V = states[25];
    const double K_i = states[26];
    const double Na_i = states[27];
    const double Ca_i = states[28];
    const double Ca_d = states[29];

    // Assign parameters
    const double P_Na = parameters[0];
    const double E_Ca_app = parameters[1];
    const double g_Ca_L = parameters[2];
    const double k_Ca = parameters[3];
    const double g_t = parameters[4];
    const double g_kur = parameters[5];
    const double g_K1 = parameters[6];
    const double g_Kr = parameters[7];
    const double g_Ks = parameters[8];
    const double g_B_Ca = parameters[9];
    const double g_B_Na = parameters[10];
    const double K_NaK_K = parameters[11];
    const double i_NaK_max = parameters[12];
    const double pow_K_NaK_Na_15 = parameters[13];
    const double i_CaP_max = parameters[14];
    const double k_CaP = parameters[15];
    const double K_NaCa = parameters[16];
    const double d_NaCa = parameters[17];
    const double gamma_Na = parameters[18];
    const double ACh = parameters[19];
    const double Mg_i = parameters[20];
    const double I_up_max = parameters[21];
    const double Vol_rel = parameters[22];
    const double Vol_up = parameters[23];
    const double alpha_rel = parameters[24];
    const double k_cyca = parameters[25];
    const double k_rel_d = parameters[26];
    const double k_rel_i = parameters[27];
    const double k_srca = parameters[28];
    const double k_xcs = parameters[29];
    const double r_recov = parameters[30];
    const double tau_tr = parameters[31];
    const double Ca_b = parameters[32];
    const double K_b = parameters[33];
    const double Na_b = parameters[34];
    const double Vol_c = parameters[35];
    const double tau_Ca = parameters[36];
    const double tau_K = parameters[37];
    const double tau_Na = parameters[38];
    const double Cm = parameters[39];
    const double F = parameters[40];
    const double R = parameters[41];
    const double T = parameters[42];
    const double stim_amplitude = parameters[43];
    const double stim_duration = parameters[44];
    const double stim_offset = parameters[45];
    const double stim_period = parameters[46];
    const double Vol_d = parameters[47];
    const double Vol_i = parameters[48];
    const double phi_Na_en = parameters[49];
    const double tau_di = parameters[50];

    // Expressions for the Sodium current component
    const double E_Na = R * T * log(Na_c / Na_i) / F;
    const double i_Na = P_Na * (F * F) * (m * m * m) * (-1. + exp(F * (-E_Na + V) / (R * T)))
                        * (0.1 * h2 + 0.9 * h1) * Na_c * V / (R * T * (-1. + exp(F * V / (R * T))));

    // Expressions for the m gate component
    const double m_factor = 0.887847222222222 + 0.0347222222222222 * V;
    const double m_infinity = 1.0 / (1. + 0.0367620699824072 * exp(-0.121802679658952 * V));
    const double tau_m = 2.4e-5 + 4.2e-5 * exp(-(m_factor * m_factor));
    values[0] = (-m + m_infinity) / tau_m;

    // Expressions for the H1 gate component
    const double h_infinity = 1.0 / (1. + 162754.791419004 * exp(0.188679245283019 * V));
    const double h_factor = 1.0 / (1. + 58032.0080361162 * exp(0.3125 * V));
    const double tau_h1 = 0.0003 + 0.03 * h_factor;
    values[1] = (-h1 + h_infinity) / tau_h1;

    // Expressions for the H2 gate component
    const double tau_h2 = 0.003 + 0.12 * h_factor;
    values[2] = (-h2 + h_infinity) / tau_h2;

    // Expressions for the L_type Ca channel component
    const double f_Ca = Ca_d / (k_Ca + Ca_d);
    const double i_Ca_L = g_Ca_L * (-E_Ca_app + V) * ((1. - f_Ca) * f_L2 + f_Ca * f_L1) * d_L;

    // Expressions for the d_L gate component
    const double d_L_infinity = 1.0 / (1. + 0.211882344332697 * exp(-0.172413793103448 * V));
    const double d_L_factor = 7. / 6. + V / 30.;
    const double tau_d_L = 0.002 + 0.0027 * exp(-(d_L_factor * d_L_factor));
    values[3] = (-d_L + d_L_infinity) / tau_d_L;

    // Expressions for the f_L1 gate component
    const double f_L_infinity = 1.0 / (1. + 47.4252567480916 * exp(0.140845070422535 * V));
    const double f_L_factor = 40. + V;
    const double tau_f_L1 = 0.01 + 0.161 * exp(-0.00482253086419753 * (f_L_factor * f_L_factor));
    values[4] = (-f_L1 + f_L_infinity) / tau_f_L1;

    // Expressions for the f_L2 gate component
    const double tau_f_L2 = 0.0626 + 1.3323 * exp(-0.00495933346558223 * (f_L_factor * f_L_factor));
    values[5] = (-f_L2 + f_L_infinity) / tau_f_L2;

    // Expressions for the Ca independent transient outward K current component
    const double E_K = R * T * log(K_c / K_i) / F;
    const double i_t = g_t * (-E_K + V) * r * s;

    // Expressions for the r gate component
    const double r_infinity = 1.0 / (1. + exp(1. / 11. - V / 11.));
    const double tau_r = 0.0015 + 0.0035 * exp(-(V * V) / 900.);
    values[6] = (-r + r_infinity) / tau_r;

    // Expressions for the s gate component
    const double s_infinity = 1.0 / (1. + 33.8432351130073 * exp(0.0869565217391304 * V));
    const double s_factor = 3.30233524526686 + 0.0629615871356885 * V;
    const double tau_s = 0.01414 + 0.025635 * exp(-(s_factor * s_factor));
    values[7] = (-s + s_infinity) / tau_s;

    // Expressions for the Ultra rapid K current component
    const double i_Kur = g_kur * (-E_K + V) * a_ur * i_ur;

    // Expressions for the Aur gate component
    const double a_ur_infinity = 1.0 / (1. + 0.49774149722499 * exp(-0.116279069767442 * V));
    const double tau_a_ur = 0.0005 + 0.009 / (1. + exp(5. / 12. + V / 12.));
    values[8] = (-a_ur + a_ur_infinity) / tau_a_ur;

    // Expressions for the Iur gate component
    const double i_ur_infinity = 1.0 / (1. + 2.11700001661267 * exp(V / 10.));
    const double tau_i_ur = 3.05 + 0.59 / (1. + exp(6. + V / 10.));
    values[9] = (-i_ur + i_ur_infinity) / tau_i_ur;

    // Expressions for the Inward rectifier component
    const double i_K1 = g_K1 * pow(K_c, 0.4457) * (-E_K + V)
                        / (1. + exp(F * (5.4 + 1.5 * V - 1.5 * E_K) / (R * T)));

    // Expressions for the n gate component
    const double n_infinity = 1.0 / (1. + 4.79191026127248 * exp(-0.078740157480315 * V));
    const double n_factor = -1. + V / 20.;
    const double tau_n = 0.7 + 0.4 * exp(-(n_factor * n_factor));
    values[10] = (-n + n_infinity) / tau_n;

    // Expressions for the Pa gate component
    const double p_a_infinity = 1.0 / (1. + exp(-5. / 2. - V / 6.));
    const double pa_factor = 0.907115443521505 + 0.0450458566821024 * V;
    const double tau_pa = 0.03118 + 0.21718 * exp(-(pa_factor * pa_factor));
    values[11] = (-pa + p_a_infinity) / tau_pa;

    // Expressions for the Pi gate component
    const double pip = 1.0 / (1. + exp(55. / 24. + V / 24.));

    // Expressions for the Background currents component
    const double i_B_Na = g_B_Na * (-E_Na + V);
    const double E_Ca = R * T * log(Ca_c / Ca_i) / (2. * F);
    const double i_B_Ca = g_B_Ca * (-E_Ca + V);

    // Expressions for the Sodium potassium pump component
    const double pow_Na_i_15 = pow(Na_i, 1.5);
    const double i_NaK = i_NaK_max * (150. + V) * K_c * pow_Na_i_15
                         / ((200. + V) * (K_NaK_K + K_c) * (pow_K_NaK_Na_15 + pow_Na_i_15));

    // Expressions for the Sarcolemmal calcium pump current component
    const double i_CaP = i_CaP_max * Ca_i / (k_CaP + Ca_i);

    // Expressions for the Na Ca ion exchanger current component
    const double i_NaCa =
            K_NaCa
            * ((Na_i * Na_i * Na_i) * Ca_c * exp(F * gamma_Na * V / (R * T))
               - (Na_c * Na_c * Na_c) * Ca_i * exp(F * (-1. + gamma_Na) * V / (R * T)))
            / (1. + d_NaCa * ((Na_c * Na_c * Na_c) * Ca_i + (Na_i * Na_i * Na_i) * Ca_c));

    // Expressions for the ACh dependent K current component
    const double i_KACh =
            10. * Cm * (0.0517 + 0.4516 / (1. + 31.9788795036608 * exp(0.0582072176949942 * V)))
            * (-E_K + V) / (1. + 9.13652 * pow(ACh, -0.477811));

    // Expressions for the Intracellular Ca buffering component
    const double J_O_C = -476. * O_C + 200000. * (1. - O_C) * Ca_i;
    const double J_O_TC = -392. * O_TC + 78400. * (1. - O_TC) * Ca_i;
    const double J_O_TMgC = -6.6 * O_TMgC + 200000. * (1. - O_TMgC - O_TMgMg) * Ca_i;
    const double J_O_TMgMg = -666. * O_TMgMg + 2000. * Mg_i * (1. - O_TMgC - O_TMgMg);
    values[12] = J_O_C;
    values[13] = J_O_TC;
    values[14] = J_O_TMgC;
    values[15] = J_O_TMgMg;
    const double J_O = 0.045 * J_O_C + 0.08 * J_O_TC + 0.16 * J_O_TMgC;
    values[16] = J_O;

    // Expressions for the Ca handling by the SR component
    const double r_Ca_d_term = Ca_d / (k_rel_d + Ca_d);
    const double r_Ca_i_term = Ca_i / (k_rel_i + Ca_i);
    const double r_Ca_d_factor = pow(r_Ca_d_term, 4.);
    const double r_Ca_i_factor = pow(r_Ca_i_term, 4.);
    const double r_act = 203.8 * r_Ca_d_factor + 203.8 * r_Ca_i_factor;
    const double r_inact = 33.96 + 339.6 * r_Ca_i_factor;
    values[17] = r_recov * (1. - F1 - F2) - F1 * r_act;
    values[18] = F1 * r_act - F2 * r_inact;
    const double i_rel_f2 = F2 / (0.25 + F2);
    const double i_rel_factor = (i_rel_f2 * i_rel_f2);
    const double i_rel = alpha_rel * (-Ca_i + Ca_rel) * i_rel_factor;
    const double i_up = I_up_max * (Ca_i / k_cyca - (k_xcs * k_xcs) * Ca_up / k_srca)
                        / ((k_cyca + Ca_i) / k_cyca + k_xcs * (k_srca + Ca_up) / k_srca);
    const double i_tr = F * Vol_rel * (-2. * Ca_rel + 2. * Ca_up) / tau_tr;
    const double J_O_Calse = -400. * O_Calse + 480. * (1. - O_Calse) * Ca_rel;
    values[19] = J_O_Calse;
    values[20] = (-i_tr + i_up) / (2. * F * Vol_up);
    values[21] = -31. * J_O_Calse + (-i_rel + i_tr) / (2. * F * Vol_rel);

    // Expressions for the Delayed rectifier K currents component
    const double i_Ks = g_Ks * (-E_K + V) * n;
    const double i_Kr = g_Kr * (-E_K + V) * pa * pip;

    // Expressions for the Cleft space ion concentrations component
    values[22] =
            (Ca_b - Ca_c) / tau_Ca + (-2. * i_NaCa + i_B_Ca + i_CaP + i_Ca_L) / (2. * F * Vol_c);
    values[23] =
            (K_b - K_c) / tau_K + (-2. * i_NaK + i_K1 + i_Kr + i_Ks + i_Kur + i_t) / (F * Vol_c);
    values[24] = (Na_b - Na_c) / tau_Na
                 + (phi_Na_en + 3. * i_NaCa + 3. * i_NaK + i_B_Na + i_Na) / (F * Vol_c);

    // Expressions for the Membrane component
    const double past = stim_period * floor(t / stim_period);
    const double i_Stim =
            (t - past <= stim_duration + stim_offset && t - past >= stim_offset ? stim_amplitude
                                                                                : 0.);
    const double I = (i_B_Ca + i_B_Na + i_CaP + i_Ca_L + i_K1 + i_KACh + i_Kr + i_Ks + i_Kur + i_Na
                      + i_NaCa + i_NaK + i_t)
                             / Cm
                     + i_Stim;
    values[25] = -1000. * I;

    // Expressions for the Intracellular ion concentrations component
    values[26] = (-i_K1 - i_Kr - i_Ks - i_Kur - i_t + 2. * i_NaK - Cm * i_Stim) / (F * Vol_i);
    values[27] = (-phi_Na_en - i_B_Na - i_Na - 3. * i_NaCa - 3. * i_NaK) / (F * Vol_i);
    const double i_di = F * Vol_d * (-2. * Ca_i + 2. * Ca_d) / tau_di;
    values[28] = -J_O + (-i_B_Ca - i_CaP - i_up + 2. * i_NaCa + i_di + i_rel) / (2. * F * Vol_i);
    values[29] = (-i_Ca_L - i_di) / (2. * F * Vol_d);
}