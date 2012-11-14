__author__ = "Johan Hake (hake.dev@gmail.com)"
__date__ = "2012-05-07 -- 2012-10-24"
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
        ode.add_states("Na+ current I_Na",
                       m=2.4676e-4, h=0.99869, j=0.99887)
        ode.add_states("Rapid-activating delayed rectifier K+ current I_Kr",
                       xKr=0.6935)
        ode.add_states("Slow-activating delayed rectifier K+ current I_Ks",
                       xKs=1.4589e-4)
        ode.add_states("Transient outward K+ current I_to",
                       xto1=3.742e-5, yto1=1)
        ode.add_states("Intracellular K fluxes",
                       K_i=159.48)
        ode.add_states("Intracellular Ca fluxes",
                       Ca_i=8.464e-5, Ca_NSR=0.2620, Ca_ss=1.315e-4, Ca_JSR=0.2616)
        ode.add_states("RyR Channel",
                       C1_RyR=0.4929, O1_RyR=6.027e-4, O2_RyR=2.882e-9, C2_RyR=0.5065)
        ode.add_states("L-type Ca Channel", 
                       C0=0.99802, C1=1.9544e-6, C2=0, C3=0, C4=0, Open=0,
                       CCa0=1.9734e-3, CCa1=0, CCa2=0, CCa3=0, CCa4=0, yCa=0.7959)
        ode.add_states("Ca buffers",
                       LTRPNCa=5.5443e-3, HTRPNCa=136.64e-3)
        
        ode.add_comment("Constants")
        ode.F       = 96.500
        ode.T       = 310
        ode.R       = 8.314
        ode.RTonF   = ode.R*ode.T/ode.F
        ode.FonRT   = ode.F/(ode.R*ode.T)
        
        ode.add_parameters("Cell geometry parameters",
                           ist=0,
                           C_sc  = 1.00,
                           A_cap = 1.534e-4,
                           V_myo = 25.84e-6,
                           V_JSR = 0.16e-6,
                           V_NSR = 2.1e-6,
                           V_ss  = 1.2e-9,
                           )
        
        ode.add_parameters("Standard ionic concentrations",
                           K_o  = 4.0,
                           Na_o = 138.0,
                           Ca_o = 2.0,
                           Na_i = 10.0,
                           )
        
        ode.add_parameters("Membrane current parameters",
                           G_KrMax = 0.0034,
                           G_KsMax = 0.00271,
                           G_toMax = 0.23815,
                           G_tiMax = 2.8,
                           G_KpMax = 0.002216,
                           G_NaMax = 12.8,
                           k_NaCa  = 0.30,
                           K_mNa   = 87.5,
                           K_mCa   = 1.38,
                           K_mK1   = 13.0,
                           k_sat   = 0.2,
                           eta     = 0.35,
                           I_NaKMax= 0.693,
                           K_mNai  = 10.0,
                           K_mKo   = 1.5,
                           I_pCaMax= 0.05,
                           K_mpCa  = 0.00005,
                           G_bCaMax= 0.0003842,
                           G_bNaMax= 0.0031,
                           )
        
        ode.add_parameters("SR parameters", 
                           v_1     = 1.8,
                           K_fb    = 0.168e-3,
                           K_rb    = 3.29,
                           K_SR    = 1.0,
                           N_fb    = 1.2,
                           N_rb    = 1.0,
                           v_maxf  = 0.813e-4,
                           v_maxr  = 0.318e-3,
                           tau_tr  = 0.5747,
                           tau_xfer= 26.7,
                           kaplus = 0.01215,
                           kaminus= 0.576,
                           kbplus = 0.00405,
                           kbminus= 1.930,
                           kcplus = 0.100,
                           kcminus= 0.0008,
                           ncoop   = 4.0,
                           mcoop   = 3.0,
                           )
        
        ode.add_parameters("L-type Ca Channel parameters",
                           fL     = 0.3,
                           gL     = 2.0,
                           bL     = 2.0,
                           aL     = 2.0,
                           omega   = 0.01,
                           PCa = 3.125e-4,
                           PK  = 5.79e-7,
                           ICahalf= -0.265,
                           )
        
        ode.add_parameters("Buffering parameters",
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
        
        ode.add_comment("Help variables")
        ode.VFonRT = ode.V*ode.FonRT
        ode.expVFonRT = exp(ode.VFonRT)
        
        ode.add_comment("I Membrane currents")
        
        ode.add_comment("Na+ current I_Na")
        
        ode.E_Na = ode.RTonF*log(ode.Na_o/ode.Na_i)
        ode.I_Na = ode.G_NaMax*ode.m**3*ode.h*ode.j*(ode.V-ode.E_Na)
        
        ode.a_h = Conditional(Ge(ode.V, -40), 0.0, 0.135*exp((80+ode.V)/(-6.8)))
        ode.b_h = Conditional(Ge(ode.V, -40), 1/(0.13*(1+exp((ode.V+10.66)/(-11.1)))), \
                              3.56*exp(0.079*ode.V)+3.1e5*exp(0.35*ode.V))
        
        ode.a_j = Conditional(Ge(ode.V, -40), 0.0, (-1.2714e5*exp(0.2444*ode.V)-\
                                                3.474e-5*exp(-0.04391*ode.V))\
                              *(ode.V+37.78)/(1+exp(0.311*(ode.V+79.23))))
        ode.b_j = Conditional(Ge(ode.V, -40), \
                              0.3*exp(-2.535e-7*ode.V)/(1+exp(-0.1*(ode.V+32))),
                              0.1212*exp(-0.01052*ode.V)/(1+exp(-0.1378*(ode.V+40.14))))


        ode.a_m = 0.32*(ode.V+47.13)/(1-exp(-0.1*(ode.V+47.13)))
        ode.b_m = 0.08*exp(-ode.V/11)
        ode.dm = Conditional(Ge(ode.V, -90), ode.a_m*(1-ode.m)-ode.b_m*ode.m, 0.0)
        
        ode.add_comment("Rapid-activating delayed rectifier K+ current I_Kr")
        
        ode.k12 = exp(-5.495+0.1691*ode.V)
        ode.k21 = exp(-7.677-0.0128*ode.V)
        ode.xKr_inf = ode.k12/(ode.k12+ode.k21)
        ode.tau_xKr = 1.0/(ode.k12+ode.k21) + 27.0
        ode.dxKr = (ode.xKr_inf-ode.xKr)/ode.tau_xKr
        
        ode.E_k  = ode.RTonF*log(ode.K_o/ode.K_i)
        ode.R_V  = 1/(1+1.4945*exp(0.0446*ode.V))
        ode.f_k  = sqrt(ode.K_o/4)
        
        ode.I_Kr = ode.G_KrMax*ode.f_k*ode.R_V*ode.xKr*(ode.V-ode.E_k)
        
        ode.add_comment("Slow-activating delayed rectifier K+ current I_Ks")
        ode.xKs_inf = 1.0/(1.0+exp(-(ode.V-24.70)/13.60))
        ode.tau_xKs = 1.0/( 7.19e-5*(ode.V-10.0)/(1.0-exp(-0.1480*(ode.V-10.0))) \
                            + 1.31e-4*(ode.V-10.0)/(exp(0.06870*(ode.V-10.0))-1.0))
        ode.dxKs = (ode.xKs_inf-ode.xKs)/ode.tau_xKs
        
        ode.E_Ks = ode.RTonF*log((ode.K_o+0.01833*ode.Na_o)/(ode.K_i+0.01833*ode.Na_i))
        
        ode.I_Ks = ode.G_KsMax*(ode.xKs**2)*(ode.V-ode.E_Ks)
        
        
        ode.add_comment("Transient outward K+ current I_to")
        ode.alpha_xto1 = 0.04516*exp(0.03577*ode.V)
        ode.beta_xto1  = 0.0989*exp(-0.06237*ode.V)
        ode.a1 = 1.0+0.051335*exp(-(ode.V+33.5)/5.0)
        ode.alpha_yto1 = 0.005415*exp(-(ode.V+33.5)/5.0)/ode.a1
        ode.a1 = 1.0+0.051335*exp((ode.V+33.5)/5.0)
        ode.beta_yto1 = 0.005415*exp((ode.V+33.5)/5.0)/ode.a1
        
        ode.dxto1 = ode.alpha_xto1*(1.e0-ode.xto1)-ode.beta_xto1*ode.xto1
        ode.dyto1 = ode.alpha_yto1*(1.e0-ode.yto1)-ode.beta_yto1*ode.yto1
        
        ode.I_to = ode.G_toMax*ode.xto1*ode.yto1*(ode.V-ode.E_k)
                
        ode.add_comment("Time-Independent K+ current I_ti")
        ode.K_tiUnlim = 1/(2+exp(1.5*(ode.V-ode.E_k)*ode.FonRT))
        
        ode.I_ti = ode.G_tiMax*ode.K_tiUnlim*(ode.K_o/(ode.K_o+ode.K_mK1))*(ode.V-ode.E_k)
        
        ode.add_comment("Plateau current I_Kp")
        ode.K_p  = 1/(1+exp((7.488-ode.V)/5.98))
        ode.I_Kp = ode.G_KpMax*ode.K_p*(ode.V-ode.E_k)
        
        ode.add_comment("NCX Current I_NaCa")
        ode.I_NaCa = ode.k_NaCa*(5000/(ode.K_mNa**3+ode.Na_o**3))*(1/(ode.K_mCa+ode.Ca_o))*\
                 (1/(1+ode.k_sat*exp((ode.eta-1)*ode.VFonRT)))*\
                 (exp(ode.eta*ode.VFonRT)*ode.Na_i**3*ode.Ca_o-\
                  exp((ode.eta-1)*ode.VFonRT)*ode.Na_o**3*ode.Ca_i)
        
        
        ode.add_comment("Na+-K+ pump current I_NaK")
        ode.sigma = 1/7*(exp(ode.Na_o/67.3)-1)
        ode.f_NaK = 1/(1+0.1245*exp(-0.1*ode.VFonRT)+0.0365*ode.sigma*exp(-ode.VFonRT))
        
        ode.I_NaK = ode.I_NaKMax*ode.f_NaK*1/(1+(ode.K_mNai/ode.Na_i)**1.5)*\
                    ode.K_o/(ode.K_o+ode.K_mKo)
        
        
        ode.add_comment("Sarcolemmal Ca2+ pump current I_pCa")
        ode.I_pCa = ode.I_pCaMax*ode.Ca_i/(ode.K_mpCa+ode.Ca_i)
        
        ode.add_comment("Ca2+ background current I_bCa")
        ode.E_Ca  = ode.RTonF/2*log(ode.Ca_o/ode.Ca_i)
        ode.I_bCa = ode.G_bCaMax*(ode.V-ode.E_Ca)
        
        
        ode.add_comment("Na+ background current I_bNa")
        ode.I_bNa = ode.G_bNaMax*(ode.V-ode.E_Na)
        
        
        ode.add_comment("II Ca2+ handling mechanisms")
        ode.add_comment("L-type Ca2+ current I_Ca")
        
        ode.alpha      = 0.4*exp((ode.V+2.0)/10.0)
        ode.beta       = 0.05*exp(-(ode.V+2.0)/13.0)
        ode.alpha_prime = ode.alpha * ode.aL
        ode.beta_prime  = ode.beta/ode.bL
        ode.gamma = 0.10375*ode.Ca_ss

        ode.C0_to_C1 = 4.e0*ode.alpha
        ode.C1_to_C2 = 3.e0*ode.alpha
        ode.C2_to_C3 = 2.e0*ode.alpha
        ode.C3_to_C4 =      ode.alpha
        
        ode.CCa0_to_CCa1 = 4.e0*ode.alpha_prime
        ode.CCa1_to_CCa2 = 3.e0*ode.alpha_prime
        ode.CCa2_to_CCa3 = 2.e0*ode.alpha_prime
        ode.CCa3_to_CCa4 =      ode.alpha_prime
        
        ode.C1_to_C0 =      ode.beta
        ode.C2_to_C1 = 2.e0*ode.beta
        ode.C3_to_C2 = 3.e0*ode.beta
        ode.C4_to_C3 = 4.e0*ode.beta
        
        ode.CCa1_to_CCa0 =      ode.beta_prime
        ode.CCa2_to_CCa1 = 2.e0*ode.beta_prime
        ode.CCa3_to_CCa2 = 3.e0*ode.beta_prime
        ode.CCa4_to_CCa3 = 4.e0*ode.beta_prime
        		
        ode.gamma =   0.10375e0*ode.Ca_ss
        
        ode.C0_to_CCa0 = ode.gamma		
        ode.C1_to_CCa1 = ode.aL*ode.C0_to_CCa0	
        ode.C2_to_CCa2 = ode.aL*ode.C1_to_CCa1	
        ode.C3_to_CCa3 = ode.aL*ode.C2_to_CCa2	
        ode.C4_to_CCa4 = ode.aL*ode.C3_to_CCa3	
        		
        ode.CCa0_to_C0 = ode.omega		
        ode.CCa1_to_C1 = ode.CCa0_to_C0/ode.bL	
        ode.CCa2_to_C2 = ode.CCa1_to_C1/ode.bL	
        ode.CCa3_to_C3 = ode.CCa2_to_C2/ode.bL	
        ode.CCa4_to_C4 = ode.CCa3_to_C3/ode.bL	
        
        ode.a1 = (ode.C0_to_C1+ode.C0_to_CCa0)*ode.C0
        ode.a2 = ode.C1_to_C0*ode.C1 + ode.CCa0_to_C0*ode.CCa0
        ode.dC0 = ode.a2 - ode.a1
        
        ode.a1 = (ode.C1_to_C0+ode.C1_to_C2+ode.C1_to_CCa1)*ode.C1
        ode.a2 = ode.C0_to_C1*ode.C0 + ode.C2_to_C1*ode.C2 + ode.CCa1_to_C1*ode.CCa1
        ode.dC1 = ode.a2 - ode.a1
        
        ode.a1 = (ode.C2_to_C1+ode.C2_to_C3+ode.C2_to_CCa2)*ode.C2
        ode.a2 = ode.C1_to_C2*ode.C1 + ode.C3_to_C2*ode.C3 + ode.CCa2_to_C2*ode.CCa2
        ode.dC2 = ode.a2 - ode.a1
        
        ode.a1 = (ode.C3_to_C2+ode.C3_to_C4+ode.C3_to_CCa3)*ode.C3
        ode.a2 = ode.C2_to_C3*ode.C2 + ode.C4_to_C3*ode.C4 + ode.CCa3_to_C3*ode.CCa3
        ode.dC3 = ode.a2 - ode.a1
        
        ode.a1 = (ode.C4_to_C3+ode.fL+ode.C4_to_CCa4)*ode.C4
        ode.a2 = ode.C3_to_C4*ode.C3 + ode.gL*ode.Open + ode.CCa4_to_C4*ode.CCa4
        ode.dC4 = ode.a2 - ode.a1
        
        ode.dOpen =  ode.fL*ode.C4 - ode.gL*ode.Open
        
        ode.a1 = (ode.CCa0_to_CCa1+ode.CCa0_to_C0)*ode.CCa0
        ode.a2 = ode.CCa1_to_CCa0*ode.CCa1 + ode.C0_to_CCa0*ode.C0
        ode.dCCa0 = ode.a2 - ode.a1
        
        ode.a1 = (ode.CCa1_to_CCa0+ode.CCa1_to_CCa2+ode.CCa1_to_C1)*ode.CCa1
        ode.a2 = ode.CCa0_to_CCa1*ode.CCa0 + ode.CCa2_to_CCa1*ode.CCa2 + ode.C1_to_CCa1*ode.C1
        ode.dCCa1 = ode.a2 - ode.a1
        
        ode.a1 = (ode.CCa2_to_CCa1+ode.CCa2_to_CCa3+ode.CCa2_to_C2)*ode.CCa2
        ode.a2 = ode.CCa1_to_CCa2*ode.CCa1 + ode.CCa3_to_CCa2*ode.CCa3 + ode.C2_to_CCa2*ode.C2
        ode.dCCa2 = ode.a2 - ode.a1
        
        ode.a1 = (ode.CCa3_to_CCa2+ode.CCa3_to_CCa4+ode.CCa3_to_C3)*ode.CCa3
        ode.a2 = ode.CCa2_to_CCa3*ode.CCa2 + ode.CCa4_to_CCa3*ode.CCa4 + ode.C3_to_CCa3*ode.C3
        ode.dCCa3 = ode.a2 - ode.a1
        
        ode.a1 = (ode.CCa4_to_CCa3+ode.CCa4_to_C4)*ode.CCa4
        ode.a2 = ode.CCa3_to_CCa4*ode.CCa3 + ode.C4_to_CCa4*ode.C4
        ode.dCCa4 = ode.a2 - ode.a1
        
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
        
        ode.add_comment("RyR Channel")
        ode.a1 = (ode.Ca_ss*1000.0)**ode.mcoop
        ode.a2 = (ode.Ca_ss*1000.0)**ode.ncoop
        ode.dC1_RyR = -ode.kaplus*ode.a2*ode.C1_RyR + ode.kaminus*ode.O1_RyR
        ode.dO2_RyR =  ode.kbplus*ode.a1*ode.O1_RyR - ode.kbminus*ode.O2_RyR
        ode.dC2_RyR =  ode.kcplus*ode.O1_RyR - ode.kcminus*ode.C2_RyR 
        ode.dO1_RyR = -(ode.dC1_RyR + ode.dO2_RyR + ode.dC2_RyR)
        
        ode.J_rel = ode.v_1*(ode.O1_RyR+ode.O2_RyR)*(ode.Ca_JSR-ode.Ca_ss)
        
        
        ode.add_comment("SERCA2a Pump")
        ode.f_b = (ode.Ca_i/ode.K_fb)**ode.N_fb
        ode.r_b = (ode.Ca_NSR/ode.K_rb)**ode.N_rb
        
        ode.J_up = ode.K_SR*(ode.v_maxf*ode.f_b-ode.v_maxr*ode.r_b)/(1+ode.f_b+ode.r_b)
        
        ode.add_comment("Intracellular Ca fluxes")
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
        
        ode.add_comment("initial stimulating current I_st")
        
        ode.add_comment("The ODE system")

        ode.diff(ode.V      , -(ode.I_Na+ode.I_Ca+ode.I_CaK+ode.I_Kr+ode.I_Ks+ode.I_to+ode.I_ti+ode.I_Kp+ode.I_NaCa+ode.I_NaK+ode.I_pCa+ode.I_bCa+ode.I_bNa+ode.ist))
        ode.diff(ode.m      , ode.dm)
        ode.diff(ode.h      , ode.a_h*(1-ode.h)-ode.b_h*ode.h)
        ode.diff(ode.j      , ode.a_j*(1-ode.j)-ode.b_j*ode.j)
        ode.diff(ode.xKr    , ode.dxKr)
        ode.diff(ode.xKs    , ode.dxKs)
        ode.diff(ode.xto1   , ode.dxto1)
        ode.diff(ode.yto1   , ode.dyto1)
        ode.diff(ode.K_i    , -(ode.I_Kr+ode.I_Ks+ode.I_to+ode.I_ti+ode.I_Kp+ode.I_CaK-2*ode.I_NaK)*ode.A_cap*ode.C_sc/(ode.V_myo*1000*ode.F))
        ode.diff(ode.Ca_i   , ode.beta_i*(ode.J_xfer-ode.J_up-ode.J_trpn-(ode.I_bCa-2*ode.I_NaCa+ode.I_pCa)*ode.A_cap*ode.C_sc/(2*ode.V_myo*1000*ode.F)))
        ode.diff(ode.Ca_NSR , ode.J_up*ode.V_myo/ode.V_NSR-ode.J_tr*ode.V_JSR/ode.V_NSR)
        ode.diff(ode.Ca_ss  , ode.beta_ss*(ode.J_rel*ode.V_JSR/ode.V_ss-ode.J_xfer*ode.V_myo/ode.V_ss-ode.I_Ca*ode.A_cap*ode.C_sc/(2*ode.V_ss*1000*ode.F)))
        ode.diff(ode.Ca_JSR , ode.beta_JSR*(ode.J_tr-ode.J_rel))
        ode.diff(ode.C1_RyR , ode.dC1_RyR)
        ode.diff(ode.O1_RyR , ode.dO1_RyR)
        ode.diff(ode.O2_RyR , ode.dO2_RyR)
        ode.diff(ode.C2_RyR , ode.dC2_RyR)
        ode.diff(ode.C0     , ode.dC0)
        ode.diff(ode.C1     , ode.dC1)
        ode.diff(ode.C2     , ode.dC2)
        ode.diff(ode.C3     , ode.dC3)
        ode.diff(ode.C4     , ode.dC4)
        ode.diff(ode.Open   , ode.dOpen)
        ode.diff(ode.CCa0   , ode.dCCa0)
        ode.diff(ode.CCa1   , ode.dCCa1)
        ode.diff(ode.CCa2   , ode.dCCa2)
        ode.diff(ode.CCa3   , ode.dCCa3)
        ode.diff(ode.CCa4   , ode.dCCa4)
        ode.diff(ode.yCa    , ode.dyCa)
        ode.diff(ode.LTRPNCa, ode.dLTRPNCa)
        ode.diff(ode.HTRPNCa, ode.dHTRPNCa)

        assert(ode.is_complete)
        self.ode = ode

    def bla_test_load_and_equality(self):
        """
        Test ODE loading from file and its equality with python created ones
        """

        ode = load_ode("winslow")
        self.assertTrue(ode == self.ode)
        self.assertNotEqual(id(ode), id(self.ode))
        
        ode = load_ode("winslow", small_change=True)

        # FIXME: Comment in when comparison works
        #self.assertFalse(ode == self.ode)

    #def test_attributes(self):
    #    """
    #    Test ODE definition using attributes
    #    """
    #    ode = ODE("winslow2")
    #    ode.clear()
    #    
    #    # States
    #    ode.add_state("e", 0.0)
    #    ode.add_state("g", 0.0)
    #    
    #    # parameters
    #    ode.add_parameter("v_rest", -85.0)
    #    ode.add_parameter("v_peak", 35.0)
    #    ode.add_parameter("time_constant", 1.0)
    #    
    #    # Local Python variables
    #    a = 0.1
    #    gs = 8.0
    #    ga = gs
    #    M1 = 0.07
    #    M2 = 0.3
    #    eps1 = 0.01
    #    
    #    # Local compuations
    #    E = (ode.e-ode.v_rest)/(ode.v_peak-ode.v_rest)
    #    eps = eps1 + M1*ode.g/(ode.e+M2)
    #    
    #    ode.diff(ode.e, -ode.time_constant*(ode.v_peak-ode.v_rest)*\
    #             (ga*E*(E-a)*(E-1) + E*ode.g))
    #    ode.diff(ode.g, 0.25*ode.time_constant*eps*(-ode.g - gs*ode.e*(E-a-1)))
    #
    #    self.assertTrue(ode == self.ode)

    def bla_test_completness(self):
        """
        Test copletness of an ODE
        """
        self.assertTrue(self.ode.is_complete)
        
        ode = ODE("winslow")
        self.assertTrue(ode.is_empty)
        
    def bla_test_members(self):
        """
        Test that ODE has the correct members
        """
        ode = self.ode
        states = ["V", "m", "h", "j", "xKr", "xKs", "xto1", "yto1", "K_i", "Ca_i", \
                  "Ca_NSR", "Ca_ss", "Ca_JSR", "C1_RyR", "O1_RyR", "O2_RyR", \
                  "C2_RyR", "C0", "C1", "C2", "C3", "C4", "Open", "CCa0", "CCa1", \
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

        
        #self.assertTrue(all(ode.has_state(state) for state in states))
        #self.assertTrue(all(ode.has_parameter(param) for param in \
        #                    parameters))
        
    def bla_test_python_code_gen(self):
        """
        Test generation of code
        """

        import numpy as np
        from gotran.codegeneration.codegenerator import \
             CodeGenerator, ODERepresentation
        from gotran.codegeneration.compilemodule import jit
        
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
        dy_correct = rhs(0.0, states, parameters)

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
            jit_oderepr = jit(oderepr)

            # Execute code
            exec(gen.dy_code())
            if numerals:
                dy_eval = rhs(0.0, states)
                jit_oderepr.rhs(0.0, states, dy_jit)
            else:
                dy_eval = rhs(0.0, states, parameters)
                jit_oderepr.rhs(0.0, states, parameters, dy_jit)

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
