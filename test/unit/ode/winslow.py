__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2012-05-07 -- 2013-04-23"
__copyright__ = "Copyright (C) 2012 " + __author__
__license__  = "GNU LGPL Version 3.0 or later"

import unittest
from gotran import *

class Creation(unittest.TestCase):

    def setUp(self):
        # Store ode
        ode = ODE("winslow")
        ode.clear()
        
        # States

        ode.add_states("Membrane",
                       V=ScalarParam(-35, ge=-200, le=50))
        ode.add_states("Na current I_Na",
                       m=2.4676e-4, h=0.99869, j=0.99887)
        ode.add_states("Rapid-activating delayed rectifier K current I_Kr",
                       xKr=0.6935)
        ode.add_states("Slow-activating delayed rectifier K current I_Ks",
                       xKs=1.4589e-4)
        ode.add_states("Transient outward K current I_to",
                       xto1=3.742e-5, yto1=1)
        ode.add_states("Intracellular K",
                       K_i=159.48)
        ode.add_states("Intracellular Ca",
                       Ca_i=8.464e-5, Ca_NSR=0.2620, Ca_ss=1.315e-4, Ca_JSR=0.2616, \
                       LTRPNCa=5.5443e-3, HTRPNCa=136.64e-3)
        ode.add_states("RyR Channel",
                       C1_RyR=0.4929, O1_RyR=6.027e-4, O2_RyR=2.882e-9, C2_RyR=0.5065)

        ode.add_states("L-type Ca Channel", 
                       yCa=0.7959)
        
        lcc = ode.add_markov_model("lcc", "L-type Ca Channel", algebraic_sum=1.0,
                                   C0=0.99802,
                                   C1=4.6456e-6,
                                   C2=1.9544e-6,
                                   C3=0,
                                   C4=0,
                                   Open=0,
                                   CCa0=1.9734e-3,
                                   CCa1=0,
                                   CCa2=0,
                                   CCa3=0,
                                   CCa4=0)

        ode.add_parameters("Cell geometry",
                           ist=0,
                           C_sc  = 1.00,
                           A_cap = 1.534e-4,
                           V_myo = 25.84e-6,
                           V_JSR = 0.16e-6,
                           V_NSR = 2.1e-6,
                           V_ss  = 1.2e-9,
                           )
        
        ode.add_parameters("Ionic concentrations",
                           K_o  = 4.0,
                           Na_o = 138.0,
                           Ca_o = 2.0,
                           Na_i = 10.0,
                           )
        
        ode.add_parameters("Na current I_Na",
                           G_NaMax = 12.8,
                           )          
        
        ode.add_parameters("Rapid-activating delayed rectifier K current I_Kr",
                           G_KrMax = 0.0034,
                           )
        
        ode.add_parameters("Slow-activating delayed rectifier K current I_Ks",
                           G_KsMax = 0.00271,
                           )
        
        ode.add_parameters("Transient outward K current I_to",
                           G_toMax = 0.23815,
                           )
        
        ode.add_parameters("Time-Independent K current I_ti",
                           G_tiMax = 2.8,
                           K_mK1   = 13.0,
                           )
        
        ode.add_parameters("Plateau current I_Kp",
                           G_KpMax = 0.002216,
                           )
        
        ode.add_parameters("NCX Current I_NaCa",
                           k_NaCa  = 0.30,
                           K_mNa   = 87.5,
                           K_mCa   = 1.38,
                           k_sat   = 0.2,
                           eta     = 0.35,
                           )
        
        ode.add_parameters("Na-K pump current I_NaK",
                           I_NaKMax= 0.693,
                           K_mNai  = 10.0,
                           K_mKo   = 1.5,
                           )
        
        ode.add_parameters("Sarcolemmal Ca pump current I_pCa",
                           I_pCaMax= 0.05,
                           K_mpCa  = 0.00005,
                           )
        
        ode.add_parameters("Ca background current I_bCa",
                           G_bCaMax= 0.0003842,
                           )
        
        ode.add_parameters("Na background current I_bNa",
                           G_bNaMax= 0.0031,
                           )
        
        ode.add_parameters("Intracellular Ca", 
                           tau_tr  = 0.5747,
                           tau_xfer= 26.7,
                           LTRPNtot   = 70e-3,
                           HTRPNtot   = 140e-3,
                           khtrpn_plus = 20.0,
                           khtrpn_minus= 66.0e-6,
                           kltrpn_plus = 40.0,
                           kltrpn_minus= 0.040,
                           CMDNtot    = 50e-3,
                           CSQNtot    = 15.0,
                           EGTAtot    = 10.0,
                           KmCMDN     = 2.38e-3,
                           KmCSQN     = 0.8,
                           KmEGTA     = 0.15e-3,
                           )
        
        ode.add_parameters("L-type Ca Channel",
                           fL     = 0.3,
                           gL     = 2.0,
                           bL     = 2.0,
                           aL     = 2.0,
                           omega  = 0.01,
                           PCa = 3.125e-4,
                           PK  = 5.79e-7,
                           ICahalf= -0.265,
                           )
        
        ode.add_parameters("RyR Channel",
                           kaplus = 0.01215,
                           kaminus= 0.576,
                           kbplus = 0.00405,
                           kbminus= 1.930,
                           kcplus = 0.100,
                           kcminus= 0.0008,
                           v_1     = 1.8,
                           ncoop   = 4.0,
                           mcoop   = 3.0,
                           )
        ode.add_parameters("SERCA2a Pump",
                           K_fb    = 0.168e-3,
                           K_rb    = 3.29,
                           K_SR    = 1.0,
                           N_fb    = 1.2,
                           N_rb    = 1.0,
                           v_maxf  = 0.813e-4,
                           v_maxr  = 0.318e-3,
                           )
        
        ode.set_component("Membrane")
        ode.add_comment("Constants")
        ode.F       = 96.500
        ode.T       = 310.
        ode.R       = 8.314
        ode.RTonF   = ode.R*ode.T/ode.F
        ode.FonRT   = ode.F/(ode.R*ode.T)
        ode.VFonRT = ode.V*ode.FonRT
        ode.expVFonRT = exp(ode.VFonRT)
        
        ode.set_component("Ionic concentrations")
        ode.E_Na = ode.RTonF*log(ode.Na_o/ode.Na_i)
        ode.E_k  = ode.RTonF*log(ode.K_o/ode.K_i)
        ode.E_Ca  = ode.RTonF/2*log(ode.Ca_o/ode.Ca_i)

        ode.set_component("Na current I_Na")
        ode.I_Na = ode.G_NaMax*ode.m**3*ode.h*ode.j*(ode.V-ode.E_Na)

        ode.a_h = Conditional(Ge(ode.V, -40), 0.0, 0.135*exp((80+ode.V)/(-6.8)))
        ode.b_h = Conditional(Ge(ode.V, -40), 1./(0.13*(1+exp((ode.V+10.66)/(-11.1)))), \
                              3.56*exp(0.079*ode.V)+3.1e5*exp(0.35*ode.V))
        
        ode.a_j = Conditional(Ge(ode.V, -40), 0.0, (-1.2714e5*exp(0.2444*ode.V)-\
                                                3.474e-5*exp(-0.04391*ode.V))\
                              *(ode.V+37.78)/(1+exp(0.311*(ode.V+79.23))))
        ode.b_j = Conditional(Ge(ode.V, -40), \
                              0.3*exp(-2.535e-7*ode.V)/(1+exp(-0.1*(ode.V+32))),
                              0.1212*exp(-0.01052*ode.V)/(1+exp(-0.1378*(ode.V+40.14))))

        ode.a_m = 0.32*(ode.V+47.13)/(1-exp(-0.1*(ode.V+47.13)))
        ode.b_m = 0.08*exp(-ode.V/11)
        
        ode.set_component("Rapid-activating delayed rectifier K current I_Kr")
        
        ode.k12 = exp(-5.495+0.1691*ode.V)
        ode.k21 = exp(-7.677-0.0128*ode.V)
        ode.xKr_inf = ode.k12/(ode.k12+ode.k21)
        ode.tau_xKr = 1.0/(ode.k12+ode.k21) + 27.0
        ode.dxKr = (ode.xKr_inf-ode.xKr)/ode.tau_xKr
        
        ode.R_V  = 1./(1+1.4945*exp(0.0446*ode.V))
        ode.f_k  = sqrt(ode.K_o/4)
        
        ode.I_Kr = ode.G_KrMax*ode.f_k*ode.R_V*ode.xKr*(ode.V-ode.E_k)
        
        ode.set_component("Slow-activating delayed rectifier K current I_Ks")
        ode.xKs_inf = 1.0/(1.0+exp(-(ode.V-24.70)/13.60))
        ode.tau_xKs = 1.0/( 7.19e-5*(ode.V-10.0)/(1.0-exp(-0.1480*(ode.V-10.0))) \
                            + 1.31e-4*(ode.V-10.0)/(exp(0.06870*(ode.V-10.0))-1.0))
        ode.dxKs = (ode.xKs_inf-ode.xKs)/ode.tau_xKs
        
        ode.E_Ks = ode.RTonF*log((ode.K_o+0.01833*ode.Na_o)/(ode.K_i+0.01833*ode.Na_i))
        
        ode.I_Ks = ode.G_KsMax*(ode.xKs**2)*(ode.V-ode.E_Ks)
        
        ode.set_component("Transient outward K current I_to")
        ode.alpha_xto1 = 0.04516*exp(0.03577*ode.V)
        ode.beta_xto1  = 0.0989*exp(-0.06237*ode.V)
        ode.a1 = 1.0+0.051335*exp(-(ode.V+33.5)/5.0)
        ode.alpha_yto1 = 0.005415*exp(-(ode.V+33.5)/5.0)/ode.a1
        ode.a1 = 1.0+0.051335*exp((ode.V+33.5)/5.0)
        ode.beta_yto1 = 0.005415*exp((ode.V+33.5)/5.0)/ode.a1
        
        ode.dxto1 = ode.alpha_xto1*(1.e0-ode.xto1)-ode.beta_xto1*ode.xto1
        ode.dyto1 = ode.alpha_yto1*(1.e0-ode.yto1)-ode.beta_yto1*ode.yto1
        
        ode.I_to = ode.G_toMax*ode.xto1*ode.yto1*(ode.V-ode.E_k)
                
        ode.set_component("Time-Independent K current I_ti")
        ode.K_tiUnlim = 1./(2+exp(1.5*(ode.V-ode.E_k)*ode.FonRT))
        
        ode.I_ti = ode.G_tiMax*ode.K_tiUnlim*(ode.K_o/(ode.K_o+ode.K_mK1))*(ode.V-ode.E_k)
        
        ode.set_component("Plateau current I_Kp")
        ode.K_p  = 1./(1+exp((7.488-ode.V)/5.98))
        ode.I_Kp = ode.G_KpMax*ode.K_p*(ode.V-ode.E_k)
        
        ode.set_component("NCX Current I_NaCa")
        ode.I_NaCa = ode.k_NaCa*(5000/(ode.K_mNa**3+ode.Na_o**3))*(1./(ode.K_mCa+ode.Ca_o))*\
                 (1./(1+ode.k_sat*exp((ode.eta-1)*ode.VFonRT)))*\
                 (exp(ode.eta*ode.VFonRT)*ode.Na_i**3*ode.Ca_o-\
                  exp((ode.eta-1)*ode.VFonRT)*ode.Na_o**3*ode.Ca_i)
        
        
        ode.set_component("Na-K pump current I_NaK")
        ode.sigma = 1./7*(exp(ode.Na_o/67.3)-1)
        ode.f_NaK = 1./(1+0.1245*exp(-0.1*ode.VFonRT)+0.0365*ode.sigma*exp(-ode.VFonRT))
        
        ode.I_NaK = ode.I_NaKMax*ode.f_NaK*1./(1+(ode.K_mNai/ode.Na_i)**1.5)*\
                    ode.K_o/(ode.K_o+ode.K_mKo)
        
        
        ode.set_component("Sarcolemmal Ca pump current I_pCa")
        ode.I_pCa = ode.I_pCaMax*ode.Ca_i/(ode.K_mpCa+ode.Ca_i)
        
        ode.set_component("Ca background current I_bCa")
        ode.I_bCa = ode.G_bCaMax*(ode.V-ode.E_Ca)
        
        
        ode.set_component("Na background current I_bNa")
        ode.I_bNa = ode.G_bNaMax*(ode.V-ode.E_Na)
        
        ode.set_component("L-type Ca Channel")
        
        ode.alpha      = 0.4*exp((ode.V+2.0)/10.0)
        ode.beta       = 0.05*exp(-(ode.V+2.0)/13.0)
        ode.gamma = 0.10375*ode.Ca_ss

        # Help list of LCC closed state variables
        normal_mode_states = [ode.C0, ode.C1, ode.C2, ode.C3, ode.C4]
        Ca_mode_states = [ode.CCa0, ode.CCa1, ode.CCa2, ode.CCa3, ode.CCa4]
        
        for ind, (s0, s1) in enumerate(zip(normal_mode_states[:-1], \
                                           normal_mode_states[1:])):
            lcc[s0, s1] = (4-ind)*ode.alpha
            lcc[s1, s0] = (1+ind)*ode.beta
        
        for ind, (s0, s1) in enumerate(zip(Ca_mode_states[:-1], \
                                           Ca_mode_states[1:])):
            lcc[s0, s1] = (4-ind)*ode.alpha*ode.aL
            lcc[s1, s0] = (1+ind)*ode.beta/ode.bL
        
        for ind, (normal, Ca_mode) in enumerate(zip(normal_mode_states, \
                                                    Ca_mode_states)):
        
            lcc[normal, Ca_mode] = ode.gamma*ode.aL**ind
            lcc[Ca_mode, normal] = ode.omega/ode.bL**ind
            
        lcc[ode.C4, ode.Open] = ode.fL
        lcc[ode.Open, ode.C4] = ode.gL
        
        ode.yCa_inf = 0.80/(1.0+exp((ode.V + 12.5)/5.0)) + 0.2
        ode.tau_yCa = 20.0 + 600.0 / (1.0 + exp((ode.V+20.0)/9.50))
        ode.dyCa = (ode.yCa_inf-ode.yCa)/ode.tau_yCa
        
        ode.VFsqonRT=(1000.0*ode.F)*ode.VFonRT
        
        ode.a1 =  1.0e-3*exp(2.0*ode.VFonRT)-ode.Ca_o*0.341 
        ode.a2 =  exp(2.0*ode.VFonRT)-1.0
        ode.ICamax = ode.PCa*4.0*ode.VFsqonRT*(ode.a1/ode.a2)
        ode.I_Ca = ode.ICamax*ode.yCa*ode.Open
        
        ode.PKprime = ode.PK/(1.0+(Min(0.0, ode.ICamax)/ode.ICahalf ))
        ode.a1 = ode.K_i*ode.expVFonRT-ode.K_o
        ode.a2 = ode.expVFonRT-1.0
        ode.I_CaK = ode.PKprime*ode.Open*ode.yCa*ode.VFsqonRT*(ode.a1/ode.a2)
        
        ode.set_component("RyR Channel")
        ode.a1 = (ode.Ca_ss*1000.0)**ode.mcoop
        ode.a2 = (ode.Ca_ss*1000.0)**ode.ncoop
        ode.dC1_RyR = -ode.kaplus*ode.a2*ode.C1_RyR + ode.kaminus*ode.O1_RyR
        ode.dO2_RyR =  ode.kbplus*ode.a1*ode.O1_RyR - ode.kbminus*ode.O2_RyR
        ode.dC2_RyR =  ode.kcplus*ode.O1_RyR - ode.kcminus*ode.C2_RyR 
        ode.dO1_RyR = -(ode.dC1_RyR + ode.dO2_RyR + ode.dC2_RyR)
        
        ode.J_rel = ode.v_1*(ode.O1_RyR+ode.O2_RyR)*(ode.Ca_JSR-ode.Ca_ss)
        
        ode.set_component("SERCA2a Pump")
        ode.f_b = (ode.Ca_i/ode.K_fb)**ode.N_fb
        ode.r_b = (ode.Ca_NSR/ode.K_rb)**ode.N_rb
        
        ode.J_up = ode.K_SR*(ode.v_maxf*ode.f_b-ode.v_maxr*ode.r_b)/(1+ode.f_b+ode.r_b)
        
        ode.set_component("Intracellular Ca")
        ode.J_tr = (ode.Ca_NSR-ode.Ca_JSR)/ode.tau_tr
        ode.J_xfer = (ode.Ca_ss-ode.Ca_i)/ode.tau_xfer
        
        
        ode.a1 = ode.kltrpn_minus * ode.LTRPNCa
        ode.dLTRPNCa = ode.kltrpn_plus*ode.Ca_i*(1.0 - ode.LTRPNCa) - ode.a1
        
        ode.a1 = ode.khtrpn_minus * ode.HTRPNCa
        ode.dHTRPNCa = ode.khtrpn_plus*ode.Ca_i*(1.0 - ode.HTRPNCa) - ode.a1
        	
        ode.J_trpn = ode.LTRPNtot*ode.dLTRPNCa+ode.HTRPNtot*ode.dHTRPNCa
         
        ode.a1 = ode.CMDNtot*ode.KmCMDN/((ode.Ca_ss+ode.KmCMDN)**2.0)
        #a2 = 0
        ode.beta_ss = 1.e0/(1.0+ode.a1)#+a2) 
        		
        ode.a1 = ode.CSQNtot*ode.KmCSQN/((ode.Ca_JSR+ode.KmCSQN)**2.0)
        ode.beta_JSR = 1.0/(1.0+ode.a1)
        
        ode.a1 = ode.CMDNtot*ode.KmCMDN/((ode.Ca_i+ode.KmCMDN)**2.e0)
        #a2 = 0
        ode.beta_i = 1.0/(1.0+ode.a1)#+a2)
        
        ode.dV_dt = -(ode.I_Na+ode.I_Ca+ode.I_CaK+ode.I_Kr+ode.I_Ks+ode.I_to+ode.I_ti+ode.I_Kp+ode.I_NaCa+ode.I_NaK+ode.I_pCa+ode.I_bCa+ode.I_bNa+ode.ist)
        ode.dm_dt = Conditional(Ge(ode.V, -90), ode.a_m*(1-ode.m)-ode.b_m*ode.m, 0.0)
        ode.dh_dt = ode.a_h*(1-ode.h)-ode.b_h*ode.h
        ode.dj_dt = ode.a_j*(1-ode.j)-ode.b_j*ode.j
        ode.dxKr_dt = ode.dxKr
        ode.dxKs_dt = ode.dxKs
        ode.dxto1_dt = ode.dxto1
        ode.dyto1_dt = ode.dyto1
        ode.dK_i_dt = -(ode.I_Kr+ode.I_Ks+ode.I_to+ode.I_ti+ode.I_Kp+ode.I_CaK-2*ode.I_NaK)*ode.A_cap*ode.C_sc/(ode.V_myo*1000*ode.F)
        ode.dCa_i_dt = ode.beta_i*(ode.J_xfer-ode.J_up-ode.J_trpn-(ode.I_bCa-2*ode.I_NaCa+ode.I_pCa)*ode.A_cap*ode.C_sc/(2*ode.V_myo*1000*ode.F))
        ode.dCa_NSR_dt = ode.J_up*ode.V_myo/ode.V_NSR-ode.J_tr*ode.V_JSR/ode.V_NSR
        ode.dCa_ss_dt = ode.beta_ss*(ode.J_rel*ode.V_JSR/ode.V_ss-ode.J_xfer*ode.V_myo/ode.V_ss-ode.I_Ca*ode.A_cap*ode.C_sc/(2*ode.V_ss*1000*ode.F))
        ode.dCa_JSR_dt = ode.beta_JSR*(ode.J_tr-ode.J_rel)
        ode.dC1_RyR_dt = ode.dC1_RyR
        ode.dO1_RyR_dt = ode.dO1_RyR
        ode.dO2_RyR_dt = ode.dO2_RyR
        ode.dC2_RyR_dt = ode.dC2_RyR
        ode.dyCa_dt = ode.dyCa
        ode.dLTRPNCa_dt = ode.dLTRPNCa
        ode.dHTRPNCa_dt = ode.dHTRPNCa

        assert(ode.is_complete)
        self.ode = ode

    def test_load_and_equality(self):
        """
        Test ODE loading from file and its equality with python created ones
        """

        ode = load_ode("winslow")
        self.assertTrue(ode == self.ode)
        self.assertNotEqual(id(ode), id(self.ode))

    def test_functionality(self):
        ode = self.ode
        expr = ode.expand_intermediate(ode.dO2_RyR)
        self.assertEqual(expr, -ode.O2_RyR*ode.kbminus + \
                         (1000.0*ode.Ca_ss)**ode.mcoop*ode.O1_RyR*ode.kbplus)
        self.assertRaises(GotranException, ode.expand_intermediate, ode.O2_RyR)
        
    def test_completness(self):
        """
        Test copletness of an ODE
        """
        self.assertTrue(self.ode.is_complete)
        
    def test_members(self):
        """
        Test that ODE has the correct members
        """
        ode = self.ode
        states = ["V", "m", "h", "j", "xKr", "xKs", "xto1", "yto1", "K_i", "Ca_i", \
                  "Ca_NSR", "Ca_ss", "Ca_JSR", "C1_RyR", "O1_RyR", "O2_RyR", \
                  "C2_RyR", "C1", "C2", "C3", "C4", "Open", "CCa0", "CCa1", \
                  "CCa2", "CCa3", "CCa4", "yCa", "LTRPNCa", "HTRPNCa"]

        parameters = ["ist", "C_sc", "A_cap", "V_myo", "V_JSR", "V_NSR", "V_ss", \
                      "K_o", "Na_o", "Ca_o", "Na_i", "G_KrMax", "G_KsMax", "G_toMax", \
                      "G_tiMax", "G_KpMax", "G_NaMax", "k_NaCa", "K_mNa", "K_mCa", \
                      "K_mK1", "k_sat", "eta", "I_NaKMax", "K_mNai", "K_mKo", \
                      "I_pCaMax", "K_mpCa", "G_bCaMax", "G_bNaMax", "v_1", "K_fb", \
                      "K_rb", "K_SR", "N_fb", "N_rb", "v_maxf", "v_maxr", "tau_tr", \
                      "tau_xfer", "kaplus", "kaminus", "kbplus", "kbminus", "kcplus", \
                      "kcminus", "ncoop", "mcoop", "fL", "gL", "bL", "aL", "omega", \
                      "PCa", "PK", "ICahalf", "LTRPNtot", "HTRPNtot", "khtrpn_plus", \
                      "khtrpn_minus", "kltrpn_plus", "kltrpn_minus", "CMDNtot", \
                      "CSQNtot", "EGTAtot", "KmCMDN", "KmCSQN", "KmEGTA"]

        components = ["winslow", "Membrane", "Na current I_Na", \
                      "Rapid-activating delayed rectifier K current I_Kr", \
                      "Slow-activating delayed rectifier K current I_Ks", \
                      "Transient outward K current I_to", "Intracellular K", \
                      "Intracellular Ca", "RyR Channel", "L-type Ca Channel", \
                      "Cell geometry", "Ionic concentrations", \
                      "Time-Independent K current I_ti", "Plateau current I_Kp", \
                      "NCX Current I_NaCa", "Na-K pump current I_NaK", \
                      "Sarcolemmal Ca pump current I_pCa", \
                      "Ca background current I_bCa", "Na background current I_bNa", \
                      "SERCA2a Pump"]
        
        self.assertTrue(all(ode.has_state(state) for state in states))
        self.assertTrue(all(state.name in states for state in ode.states))
        
        self.assertTrue(all(ode.has_parameter(param) for param in \
                            parameters))
        self.assertTrue(all(param.name in parameters for param in \
                            ode.parameters))

        self.assertTrue(all(ode.has_component(comp) for comp in components))
        self.assertTrue(all(comp in components for comp in ode.components))

    def test_components(self):
        
        ode = self.ode
        components = {"winslow":
                      dict(states=[], parameters=[], variables=["time", "dt"],
                           intermediates=[], derivatives=[],
                           external_object_dep=[], external_component_dep=[]),
                      
                      "Membrane":
                      dict(states=["V"],
                           parameters=[],
                           variables=[],
                           intermediates=["F", "T", "R", "RTonF", "FonRT", 
                                          "VFonRT", "expVFonRT"],
                           derivatives=["dV_dt"],
                           external_object_dep=["I_Ca", "I_NaCa", "I_Ks", "I_to",
                                                "I_CaK", "I_NaK", "I_Kp", "I_ti",
                                                "I_bNa", "ist", "I_pCa", "I_Kr",
                                                "I_Na", "I_bCa"],
                           external_component_dep=['Time-Independent K current I_ti',
                            'Cell geometry', 'L-type Ca Channel', 'Plateau current I_Kp',
                            'Sarcolemmal Ca pump current I_pCa', 'Na-K pump current I_NaK',
                            'Transient outward K current I_to', 'Na current I_Na',
                            'Rapid-activating delayed rectifier K current I_Kr',
                            'NCX Current I_NaCa', 'Ca background current I_bCa',
                            'Na background current I_bNa',
                            'Slow-activating delayed rectifier K current I_Ks']), 

                      "Na current I_Na":
                      dict(states=["m", "h", "j"],
                           parameters=["G_NaMax"],
                           variables=[],
                           intermediates=['I_Na', 'a_h', 'b_h', 'a_j', 'b_j', 'a_m',
                                          'b_m'],
                           derivatives=['dm_dt', 'dh_dt', 'dj_dt'],
                           external_object_dep=["E_Na", "V"],
                           external_component_dep=['Ionic concentrations', 'Membrane']), 

                      "Rapid-activating delayed rectifier K current I_Kr":
                      dict(states=['xKr'],
                           parameters=['G_KrMax'],
                           variables=[],
                           intermediates=['k12', 'k21', 'xKr_inf', 'tau_xKr', 'dxKr',
                                          'R_V', 'f_k', 'I_Kr'],
                           derivatives=['dxKr_dt'],
                           external_object_dep=['E_k', 'V', 'K_o'],
                           external_component_dep=['Ionic concentrations', 'Membrane']),
                      
                      "Slow-activating delayed rectifier K current I_Ks":
                      dict(states=['xKs'],
                           parameters=['G_KsMax'],
                           variables=[],
                           intermediates=['xKs_inf', 'tau_xKs', 'dxKs', 'E_Ks', 'I_Ks'],
                           derivatives=['dxKs_dt'],
                           external_object_dep=['RTonF', 'K_i', 'Na_i', 'Na_o', 'V', 'K_o'],
                           external_component_dep=['Ionic concentrations',
                                                   'Intracellular K', 'Membrane']),  

                      "Transient outward K current I_to":
                      dict(states=['xto1', 'yto1'],
                           parameters=['G_toMax'],
                           variables=[],
                           intermediates=['alpha_xto1', 'beta_xto1', 'a1', 'alpha_yto1',
                                          'a1', 'beta_yto1', 'dxto1', 'dyto1', 'I_to'],
                           derivatives=['dxto1_dt', 'dyto1_dt'],
                           external_object_dep=['V', 'E_k'],
                           external_component_dep=['Ionic concentrations', 'Membrane']), 

                      "Intracellular K":
                      dict(states=['K_i'],
                           parameters=[],
                           variables=[],
                           intermediates=[],
                           derivatives=['dK_i_dt'],
                           external_object_dep=['F', 'I_Ks', 'C_sc', 'I_Kp', 'I_to',
                                                'V_myo', 'I_NaK', 'A_cap', 'I_CaK',
                                                'I_Kr', 'I_ti'],
                           external_component_dep=['Time-Independent K current I_ti',
                            'Cell geometry', 'L-type Ca Channel', 'Plateau current I_Kp',
                            'Na-K pump current I_NaK', 'Transient outward K current I_to',
                            'Rapid-activating delayed rectifier K current I_Kr',
                            'Slow-activating delayed rectifier K current I_Ks',
                            'Membrane']),  

                      "Intracellular Ca":
                      dict(states=['Ca_JSR', 'Ca_NSR', 'Ca_i', 'Ca_ss', 'HTRPNCa', 
                                   'LTRPNCa'],
                           parameters=['CMDNtot', 'CSQNtot', 'EGTAtot', 'HTRPNtot',
                                       'KmCMDN', 'KmCSQN', 'KmEGTA', 'LTRPNtot',
                                       'khtrpn_minus', 'khtrpn_plus', 'kltrpn_minus',
                                       'kltrpn_plus', 'tau_tr', 'tau_xfer'],
                           variables=[],
                           intermediates=['J_tr', 'J_xfer', 'a1', 'dLTRPNCa', 'a1',
                                          'dHTRPNCa', 'J_trpn', 'a1', 'beta_ss', 'a1',
                                          'beta_JSR', 'a1', 'beta_i'],
                           derivatives=['dCa_i_dt', 'dCa_NSR_dt', 'dCa_ss_dt',
                                        'dCa_JSR_dt', 'dLTRPNCa_dt', 'dHTRPNCa_dt'],
                           external_object_dep=['F', 'J_up', 'I_bCa', 'I_Ca', 'V_JSR',
                                                'V_myo', 'V_ss', 'A_cap', 'C_sc',
                                                'V_NSR', 'I_pCa', 'J_rel', 'I_NaCa'],
                           external_component_dep=['Membrane', 'Cell geometry',
                            'L-type Ca Channel', 'Sarcolemmal Ca pump current I_pCa',
                            'SERCA2a Pump', 'RyR Channel', 'Ca background current I_bCa',
                            'NCX Current I_NaCa']), 

                      "RyR Channel":
                      dict(states=['C1_RyR', 'C2_RyR', 'O1_RyR', 'O2_RyR'],
                           parameters=['kaminus', 'kaplus', 'kbminus', 'kbplus',
                                       'kcminus', 'kcplus', 'mcoop', 'ncoop', 'v_1'],
                           variables=[],
                           intermediates=['a1', 'a2', 'dC1_RyR', 'dO2_RyR', 'dC2_RyR',
                                          'dO1_RyR', 'J_rel'],
                           derivatives=['dC1_RyR_dt', 'dO1_RyR_dt', 'dO2_RyR_dt',
                                        'dC2_RyR_dt'],
                           external_object_dep=['Ca_ss', 'Ca_JSR'],
                           external_component_dep=['Intracellular Ca']),

                      "L-type Ca Channel":
                      dict(states=['C1', 'C2', 'C3', 'C4', 'CCa0', 'CCa1',
                                   'CCa2', 'CCa3', 'CCa4', 'Open', 'yCa'],
                           parameters=['ICahalf', 'PCa', 'PK', 'aL', 'bL', 'fL',
                                       'gL', 'omega'],
                           variables=[],
                           intermediates=['C0', 'alpha', 'beta', 'alpha_prime', 'beta_prime',
                            'gamma', 'C0_to_C1', 'C1_to_C2', 'C2_to_C3', 'C3_to_C4',
                            'CCa0_to_CCa1', 'CCa1_to_CCa2', 'CCa2_to_CCa3',
                            'CCa3_to_CCa4', 'C1_to_C0', 'C2_to_C1', 'C3_to_C2',
                            'C4_to_C3', 'CCa1_to_CCa0', 'CCa2_to_CCa1', 'CCa3_to_CCa2',
                            'CCa4_to_CCa3', 'gamma', 'C0_to_CCa0', 'C1_to_CCa1',
                            'C2_to_CCa2', 'C3_to_CCa3', 'C4_to_CCa4', 'CCa0_to_C0',
                            'CCa1_to_C1', 'CCa2_to_C2', 'CCa3_to_C3', 'CCa4_to_C4',
                            'a1', 'a2', 'dC0', 'a1', 'a2', 'dC1', 'a1', 'a2', 'dC2',
                            'a1', 'a2', 'dC3', 'a1', 'a2', 'dC4', 'dOpen', 'a1', 'a2',
                            'dCCa0', 'a1', 'a2', 'dCCa1', 'a1', 'a2', 'dCCa2', 'a1',
                            'a2', 'dCCa3', 'a1', 'a2', 'dCCa4', 'yCa_inf', 'tau_yCa',
                            'dyCa', 'VFsqonRT', 'a1', 'a2', 'ICamax', 'I_Ca',
                            'PKprime', 'a1', 'a2', 'I_CaK'],
                           derivatives=['dC0_dt', 'dC1_dt', 'dC2_dt', 'dC3_dt',
                                        'dC4_dt', 'dOpen_dt', 'dCCa0_dt', 'dCCa1_dt',
                                        'dCCa2_dt', 'dCCa3_dt', 'dCCa4_dt', 'dyCa_dt'],
                           external_object_dep=['F', 'VFonRT', 'expVFonRT', 'Ca_o',
                                                'Ca_ss', 'K_i', 'K_o', 'V'],
                           external_component_dep=['Ionic concentrations',
                            'Intracellular K', 'Intracellular Ca', 'Membrane']),  

                      "Cell geometry":
                      dict(states=[],
                           parameters=['A_cap', 'C_sc', 'V_JSR', 'V_NSR', 'V_myo',
                                       'V_ss', 'ist'],
                           variables=[],
                           intermediates=[],
                           derivatives=[],
                           external_object_dep=[],
                           external_component_dep=[]), 

                      "Ionic concentrations":
                      dict(states=[],
                           parameters=['Ca_o', 'K_o', 'Na_i', 'Na_o'],
                           variables=[],
                           intermediates=['E_Na', 'E_k', 'E_Ca'],
                           derivatives=[],
                           external_object_dep=['RTonF', 'K_i', 'Ca_i'],
                           external_component_dep=['Membrane', 'Intracellular K',
                                                   'Intracellular Ca']),  

                      "Time-Independent K current I_ti":
                      dict(states=[],
                           parameters=['G_tiMax', 'K_mK1'],
                           variables=[],
                           intermediates=['K_tiUnlim', 'I_ti'],
                           derivatives=[],
                           external_object_dep=['K_o', 'V', 'E_k', 'FonRT'],
                           external_component_dep=['Ionic concentrations', 'Membrane']), 

                      "Plateau current I_Kp":
                      dict(states=[],
                           parameters=['G_KpMax'],
                           variables=[],
                           intermediates=['K_p', 'I_Kp'],
                           derivatives=[],
                           external_object_dep=['V', 'E_k'],
                           external_component_dep=['Ionic concentrations', 'Membrane']),  

                      "NCX Current I_NaCa":
                      dict(states=[],
                           parameters=['K_mCa', 'K_mNa', 'eta', 'k_NaCa', 'k_sat'],
                           variables=[],
                           intermediates=['I_NaCa'],
                           derivatives=[],
                           external_object_dep=['Na_i', 'VFonRT', 'Ca_i',
                                                'Na_o', 'Ca_o', 'V'],
                           external_component_dep=['Ionic concentrations',
                            'Intracellular Ca', 'Membrane']),

                      "Na-K pump current I_NaK":
                      dict(states=[],
                           parameters=['I_NaKMax', 'K_mKo', 'K_mNai'],
                           variables=[],
                           intermediates=['sigma', 'f_NaK', 'I_NaK'],
                           derivatives=[],
                           external_object_dep=['Na_o', 'Na_i', 'VFonRT', 'K_o'],
                           external_component_dep=['Ionic concentrations', 'Membrane']),  

                      "Sarcolemmal Ca pump current I_pCa":
                      dict(states=[],
                           parameters=['I_pCaMax', 'K_mpCa'],
                           variables=[],
                           intermediates=['I_pCa'],
                           derivatives=[],
                           external_object_dep=["Ca_i"],
                           external_component_dep=['Intracellular Ca']),  

                      "Ca background current I_bCa":
                      dict(states=[],
                           parameters=['G_bCaMax'],
                           variables=[],
                           intermediates=['I_bCa'],
                           derivatives=[],
                           external_object_dep=['E_Ca', 'V'],
                           external_component_dep=['Ionic concentrations', 'Membrane']), 

                      "Na background current I_bNa":
                      dict(states=[],
                           parameters=['G_bNaMax'],
                           variables=[],
                           intermediates=['I_bNa'],
                           derivatives=[],
                           external_object_dep=['E_Na', 'V'],
                           external_component_dep=['Ionic concentrations', 'Membrane']),  

                      "SERCA2a Pump":
                      dict(states=[],
                           parameters=['K_SR', 'K_fb', 'K_rb', 'N_fb', 'N_rb',
                                       'v_maxf', 'v_maxr'],
                           variables=[],
                           intermediates=['f_b', 'r_b', 'J_up'],
                           derivatives=[],
                           external_object_dep=['Ca_i', 'Ca_NSR'],
                           external_component_dep=['Intracellular Ca']), 
                      }

        for comp_name, dep in components.items():
            for what in ["states", "parameters", "variables",
                         "intermediates", "derivatives",
                         "external_object_dep", "external_component_dep"]:
                for obj in getattr(ode.components[comp_name], what):
                    if isinstance(obj, ODEObject):
                        self.assertTrue(obj.name in dep[what], \
                                        msg="{0}!={1}".format(obj.name,dep[what]))
                    elif isinstance(obj, str):
                        self.assertTrue(obj in dep[what])
                    else:
                        print "NO!"
        

    def test_extraction_and_subode(self):
        ode = self.ode

        # Extract all K related stuff
        odek = ode.extract_components(\
            "K_comp", "Rapid-activating delayed rectifier K current I_Kr", \
            "Slow-activating delayed rectifier K current I_Ks", \
            "Transient outward K current I_to", "Intracellular K",\
            "Time-Independent K current I_ti", "Plateau current I_Kp")

        # Extract all Na related stuff
        odeca = ode.extract_components(\
            "Ca_comp", "Intracellular Ca", "RyR Channel", "L-type Ca Channel", \
            "Sarcolemmal Ca pump current I_pCa", \
            "Ca background current I_bCa", "SERCA2a Pump")

        # Extract all Ca related stuff
        odena = ode.extract_components(\
            "Na_comp", "Na current I_Na", "NCX Current I_NaCa", "Na-K pump current I_NaK", \
            "Na background current I_bNa")
        
        ode_dup = ode.extract_components("winslow_dup", "Membrane", "Cell geometry", \
                                         "Ionic concentrations")
        
        ode_dup.add_subode(odek, prefix="")
        ode_dup.add_subode(odeca, prefix="")
        ode_dup.add_subode(odena, prefix="")

        self.assertTrue(ode_dup.is_complete)
        self.assertTrue(ode==ode_dup)

        ode_subode = load_ode("winslow_subode")
        self.assertTrue(ode==ode_subode)
        
    def test_python_code_gen(self):
        """
        Test generation of code
        """

        import numpy as np
        from gotran.codegeneration.codegenerator import \
             CodeGenerator, ODERepresentation
        from gotran.codegeneration.compilemodule import compile_module
        
        keep, use_cse, numerals, use_names = (1,0,0,1)

        gen = CodeGenerator(ODERepresentation(self.ode,
                                              keep_intermediates=keep, \
                                              use_cse=use_cse,
                                              parameter_numerals=numerals,\
                                              use_names=use_names))

        exec(gen.init_states_code())
        exec(gen.init_param_code())
        exec(gen.dy_code())

        parameters = default_parameters()
        states = init_values()
        dy_jit = np.asarray(states).copy()
        dy_correct = rhs(states, 0.0, parameters)

        for keep, use_cse, numerals, use_names in \
                [(1,0,0,1), (1,0,0,0), \
                 (1,0,1,1), (1,0,1,0), \
                 (0,0,0,1), (0,0,0,0), \
                 (0,0,1,1), (0,0,1,0), \
                 (0,1,0,1), (0,1,0,0), \
                 (0,1,1,1), (0,1,1,0)]:

            oderepr = ODERepresentation(self.ode,
                                        keep_intermediates=keep, \
                                        use_cse=use_cse,
                                        parameter_numerals=numerals,\
                                        use_names=use_names)
            gen = CodeGenerator(oderepr)
            jit_oderepr = compile_module(oderepr)

            # Execute code
            exec(gen.dy_code())
            if numerals:
                dy_eval = rhs(states, 0.0)
                jit_oderepr.rhs(states, 0.0, dy_jit)
            else:
                dy_eval = rhs(states, 0.0, parameters)
                jit_oderepr.rhs(states, 0.0, parameters, dy_jit)

            self.assertTrue(np.sum(np.abs((dy_eval-dy_correct))) < 1e-12)
            self.assertTrue(np.sum(np.abs((dy_jit-dy_correct))) < 1e-12)
            
    def test_matlab_python_code(self):
        from gotran.codegeneration.codegenerator import \
             MatlabCodeGenerator, ODERepresentation
        
        keep, use_cse, numerals, use_names = (1,0,0,1)

        gen = MatlabCodeGenerator(ODERepresentation(self.ode,
                                                    keep_intermediates=keep, \
                                                    use_cse=use_cse,
                                                    parameter_numerals=numerals,\
                                                    use_names=use_names))

        open("winslowInit.m", "w").write(gen.default_value_code())
        open("winslow.m", "w").write(gen.dy_code())
        
if __name__ == "__main__":
    unittest.main()
