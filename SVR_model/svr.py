import numpy as np

linalg.lstsq(a, b[, rcond]) - метод наименьших квадратов.

'''The SV model is an extended version of the GMP model. Based on the GMP model, it adds to
#a conjugation of input signal with special time delay. Removing the duplicates, the SV model can be
#written as'''

y_sv0 = 0.
def y_sv(n):
    for k in range (1,K_a+1,2):
        for m in range(0,M_a+1,1):
            y_sv0 +=a_km*x(n-m)*abs(x(n-m))**(k-1)
    for k in range (3,K_b+1,2):
        for m in range(0,M_b+1,1):
            for l in range(1,L_b+1):
                y_sv0 +=b_kml*x(n-m)*abs(x(n-m+l))**(k-1)
    for k in range (3,K_c+1,2):
        for m in range(0,M_c+1,1):
            for l in range(1,L_c+1):
                y_sv0 +=c_kml*x(n-m)*abs(x(n-m+l))**(k-1)

    for k in range (3,K_d+1,2):
        for m in range(0,M_d+1,1):
            for l in range(1,L_d+1):
                y_sv0 +=d_kml*x(n-m)*(x(n-m+l)**2)*abs(x(n-m-l))**(k-3)
    for k in range (3,K_e+1,2):
        for m in range(0,M_e+1,1):
            for l in range(1,L_e+1):
                y_sv0 +=e_kml**x(n-m)*(x(n-m+l)**2)*abs(x(n-m+l))**(k-3)
    return y_sv0;
