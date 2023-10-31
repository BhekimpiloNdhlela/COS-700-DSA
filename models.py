###################################################################################################################
# Author         : Bhekimpilo Ndhlela and Seani Rananga
# Date           : 09 August 2023
# Algorithms name: Energy spectral efficiency tradeoff in downlink OFDMA network; Distributed channel
# Reference      : https://onlinelibrary.wiley.com/doi/abs/10.1002/dac.2725; https://dl.acm.org/citation.cfm?id=1582596;
#                  assignemnt in cognitive radio network; Proposed Approach Based on GFDM for
#                  Maximizing Throughput in Underlay Cognitive Radio Network.
# Description    : This program calculates the spectral efficiency and data rate the three algorithms against
#                  Power(dBm) and Bandwidth(dBm) for a downlink OFDMA single circular cell network with one
#                  base station(BS) and 10 user equipments(UE) 
###################################################################################################################
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------
# The link model Here we define variables that will be used to calculate SNR
# the maximal transmit power ranges form -30 to 30 dBm and we use step 2,we have 10
B = 12.81 # Pathloss constant
sigma = 3.76 # Pathloss exponent constant
sm = 1 # lognormal fading variable constant
G = 9  # dB # Power gain constant of the transmit and receive antenna

# constant complex additive white Gaussian noise power
sigmaSquaredDBM = -100  # dBm
sigmaSquaredWatts = 1e-13  # Watts

fm = 1 # small-scale fading channel constant
K = 10 # Available number of channels

# Transmit power of K channels is uniformly distributed between 10 dBm and 30 dBm
PowerDBM = np.linspace(-30, 30, 10)
print("PowerDBM", PowerDBM)
PowerWatts = 10 ** (PowerDBM / 10) / 1000  

# The bandwidth of K channels is uniformly distributed between 32000 Hz and 160000 Hz
Bandwidth = np.linspace(36000, 160000, 10)
BandwidthKHz = np.linspace(36, 160, 10)

# Interference of K channels is uniformly distributed between -120 dBm and -40 dBm
InterferenceDBM = np.linspace(-120, -40, 10)
InterferenceWatts = 10 ** (InterferenceDBM / 10) / 1000  # converting DBM to watts


Pb = 10 ** -7 # Bit error probability
C = 10 ** -6 # constant
d = 10 ** 6 # Power decay factor (d^-?)

kb = (-2 / 3) * np.log2(Pb / 2) # calculating kb

# other constants
sg = 100000  # sub channel activity
Ws = 30000  # subcarrier distance in MHz
i = 0  # subchannel
Hi = 1  # power gain of data channel of SU
b = 100  # factor of Qam mapper
Pm = 0  # optimum power transmitted by SU m over sub channel
Hm = 1  # channel power gain between SU i transmitter and other SU m receiver
phi = 1  # PU activity
Pp = np.linspace(-25, 30, 10)  # power transmitted by PU
PowerWattsP = 10 ** (Pp / 10) / 100
PowerWattsPm = 10 ** (Pp / 10) / 100
Hp = 1  # channel power gain between PU transmitter and SU receiver i
NF = 0.1  # noise factor
No = 0.1  # noise power density of additive white Gaussian noise with zero mean and variance

# Assuming P1 is the transmit power for SU one, which can be taken from PowerWatts[0]
P1 = PowerWatts[0]

# Assuming H1 is the channel gain for SU one, which can be taken from Hi (as provided)
H1 = Hi

# Assuming I1 is the interference for SU one, which can be taken from InterferenceWatts[0]
I1 = InterferenceWatts[0]

# Calculate SNR for SU one
SNR_SU1 = P1 * H1 / (sigmaSquaredWatts + I1)

# Calculate Spectral Efficiency for SU one using Shannon's formula
SE_SU1 = np.log2(1 + SNR_SU1)
# print(f"Spectral Efficiency for SU one: {SE_SU1} bps/Hz")
# # ----------------------------------------------------------------------------------------------------

# Circle function
def circle_function(x_BS, y_BS, x_UE, y_UE, n_EU, radius):
    # This is a basic translation of the circle function and the main structure.
    plt.figure(1)
    np.random.seed(1)
    x_UE, y_UE  = np.random.randn(), np.random.randn()

    circle_theta = np.linspace(0, 2*np.pi, 100)
    xunit = radius * np.cos(circle_theta) + x_BS
    yunit = radius * np.sin(circle_theta) + y_BS
    plt.plot(xunit, yunit)
    plt.scatter(x_BS, y_BS, color='g')

    theta = np.random.rand(n_EU) * 2 * np.pi
    r = np.sqrt(np.random.rand(n_EU)) * radius
    x = x_UE + r * np.cos(theta)
    y = y_UE + r * np.sin(theta)
    plt.scatter(x, y, marker='.')
    plt.title(label="Network model with 10 UE randomly distributed", fontsize=15, color="black")
    d = np.sqrt((x_BS - x)**2 + (y_BS - y)**2)
    d = np.sort(d)[::-1]
    plt.show()
    return d

def get_total_se_and_dr(power_watt, sigma_scalar, d):
    """
    function used to get the total spectral efficiency and data rate
    (se, dr)
    """
    SE, DR = 0.0, 0.0
    for i in range(len(d)):
        SE_temp = np.log2(1 + ((power_watt) * (fm**2 * (G * B * (sigma_scalar) * sm)) / sigmaSquaredWatts))
        SE += SE_temp

        DR_temp = Bandwidth[0] * np.log2(1 + ((power_watt) * (fm**2 * (G * B * (sigma_scalar) * sm)) / sigmaSquaredWatts))
        DR += DR_temp
    return SE, DR

def plot_data_vs_power(x_values, y_values_list, colors, labels, y_label, title, figure_number):
    plt.figure(figure_number)
    for y_values, color, label in zip(y_values_list, colors, labels):
        plt.plot(x_values, y_values, color, label=label)
    
    plt.grid(which='both')
    plt.legend(loc='upper left')
    plt.xlabel('Power (dBm)')
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, which='both')
    plt.minorticks_on()
    plt.tight_layout()
    plt.show()

def calculate_values(bandwidth, PowerWatts, kb, d):
    DR, SE = [], []
    for power in PowerWatts:
        value = bandwidth * np.log2(1 + (1/kb) * (power / (1e-8 * d)))
        DR.append(value)
        SE.append(value)
    return DR, SE

# Main execution
if __name__ == "__main__":
    # Call the circle function
    d = circle_function(0, 0, 0, 0, 10, 1000)

    total_se_vector = np.zeros(10, dtype=float) # vector to store the total spectral efficiencies
    total_dr_vector = np.zeros(10, dtype=float) # vector to store the total data rates

    sss = [845**-sigma, 988**-sigma, 925**-sigma, 850**-sigma, 780**-sigma, 870**-sigma, 770**-sigma, 835**-sigma, 890**-sigma, 923**-sigma]
    pws = [PowerWatts[0],PowerWatts[1],PowerWatts[1], PowerWatts[1],PowerWatts[1],PowerWatts[2],PowerWatts[2],PowerWatts[3],PowerWatts[4],PowerWatts[5]]

    for i, pw, ss in zip([i for i in range(10)], pws, sss):
        total_se_and_dr = get_total_se_and_dr(pw, ss, d)
        total_se_vector[i] = total_se_and_dr[0]
        total_dr_vector[i] = total_se_and_dr[1]

    # TotalSpectralEfficiency against Bandwidth graph
    figure, axis = plt.subplots(1, 2)
    axis[0].plot(Bandwidth, total_se_vector, 'bo-')
    axis[0].set_title("Total Spectral Efficiency vs. Bandwidth")
    axis[0].set_xlabel('Bandwidth(KHz)')
    axis[0].set_ylabel('Spectral Efficiency (bits/s/Hz)')
    axis[0].grid(True, which='both')

    # TotalDR against Bandwidth graph
    axis[1].plot(Bandwidth, total_dr_vector, 'ro-')
    axis[1].set_title("Total Data Rate vs. Bandwidth")
    axis[1].set_xlabel('Bandwidth(KHz)')
    axis[1].set_ylabel('Data Rate (bits/s/Hz)')
    axis[1].grid(True, which='both')
    plt.show()

d = 10**9 # % Power decay factor(d^-?)

# Data rate calculations
DR_1 = Bandwidth[0] * np.log2(1 + (1/kb) * (PowerWatts[0] / (1e-8 * d)))
DR_2 = Bandwidth[0] * np.log2(1 + (1/kb) * (PowerWatts[1] / (1e-8 * d)))
DR_3 = Bandwidth[0] * np.log2(1 + (1/kb) * (PowerWatts[2] / (1e-8 * d)))
DR_4 = Bandwidth[0] * np.log2(1 + (1/kb) * (PowerWatts[3] / (1e-8 * d)))
DR_5 = Bandwidth[0] * np.log2(1 + (1/kb) * (PowerWatts[4] / (1e-8 * d)))
DR_6 = Bandwidth[0] * np.log2(1 + (1/kb) * (PowerWatts[5] / (1e-8 * d)))
DR_7 = Bandwidth[0] * np.log2(1 + (1/kb) * (PowerWatts[6] / (1e-8 * d)))
DR_8 = Bandwidth[0] * np.log2(1 + (1/kb) * (PowerWatts[7] / (1e-8 * d)))
DR_9 = Bandwidth[0] * np.log2(1 + (1/kb) * (PowerWatts[8] / (1e-8 * d)))
DR_10 = Bandwidth[0] * np.log2(1 + (1/kb) * (PowerWatts[9] / (1e-8 * d)))

# SE for channel 1 using Bandwidth of 34 kHz(34000) and -50 dBm (1e-8 Watts) Interference
SE_1 = Bandwidth[0]*np.log2(1 + (1/kb) * (PowerWatts[0] / (1e-8 * d)))
SE_2 = Bandwidth[0]*np.log2(1 + (1/kb) * (PowerWatts[1] / (1e-8 * d)))
SE_3 = Bandwidth[0]*np.log2(1 + (1/kb) * (PowerWatts[2] / (1e-8 * d)))
SE_4 = Bandwidth[0]*np.log2(1 + (1/kb) * (PowerWatts[3] / (1e-8 * d)))
SE_5 = Bandwidth[0]*np.log2(1 + (1/kb) * (PowerWatts[4] / (1e-8 * d)))
SE_6 = Bandwidth[0]*np.log2(1 + (1/kb) * (PowerWatts[5] / (1e-8 * d)))
SE_7 = Bandwidth[0]*np.log2(1 + (1/kb) * (PowerWatts[6] / (1e-8 * d)))
SE_8 = Bandwidth[0]*np.log2(1 + (1/kb) * (PowerWatts[7] / (1e-8 * d)))
SE_9 = Bandwidth[0]*np.log2(1 + (1/kb) * (PowerWatts[8] / (1e-8 * d)))
SE_10 = Bandwidth[0]*np.log2(1 + (1/kb) * (PowerWatts[9] / (1e-8 * d)))

# %Data rate for channel 3 using Bandwidth of 105 kHz 105000 and -50 dBm (1e-8 Watts)Interference
DR_31 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[0] / (1e-8 * d))) 
DR_32 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[1] / (1e-8 * d)))
DR_33 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[2] / (1e-8 * d)))
DR_34 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[3] / (1e-8 * d)))
DR_35 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[4] / (1e-8 * d)))
DR_36 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[5] / (1e-8 * d)))
DR_37 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[6] / (1e-8 * d)))
DR_38 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[7] / (1e-8 * d)))
DR_39 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[8] / (1e-8 * d)))
DR_310 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[9] / (1e-8 * d)))
# SE for channel 3 using Bandwidth of 105 kHz(105000) and -50 dBm (1e-8 Watts)Interference
SE_31 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[0] / (1e-8 * d)))
SE_32 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[1] / (1e-8 * d)))
SE_33 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[2] / (1e-8 * d)))
SE_34 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[3] / (1e-8 * d)))
SE_35 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[4] / (1e-8 * d)))
SE_36 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[5] / (1e-8 * d)))
SE_37 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[6] / (1e-8 * d)))
SE_38 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[7] / (1e-8 * d)))
SE_39 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[8] / (1e-8 * d)))
SE_310 = Bandwidth[1] * np.log2(1 + (1/kb) * (PowerWatts[9] / (1e-8 * d)))

# Data rate for channel 2 using Bandwidth of Watts Interference
DR_21 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[0] / (1e-8 * d)))
DR_22 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[1] / (1e-8 * d)))
DR_23 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[2] / (1e-8 * d)))
DR_24 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[3] / (1e-8 * d)))
DR_25 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[4] / (1e-8 * d)))
DR_26 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[5] / (1e-8 * d)))
DR_27 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[6] / (1e-8 * d)))
DR_28 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[7] / (1e-8 * d)))
DR_29 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[8] / (1e-8 * d)))
DR_210 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[9] / (1e-8 * d)))

SE_21 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[0] / (1e-8 * d)))
SE_22 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[1] / (1e-8 * d)))
SE_23 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[2] / (1e-8 * d)))
SE_24 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[3] / (1e-8 * d)))
SE_25 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[4] / (1e-8 * d)))
SE_26 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[5] / (1e-8 * d)))
SE_27 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[6] / (1e-8 * d)))
SE_28 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[7] / (1e-8 * d)))
SE_29 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[8] / (1e-8 * d)))
SE_210 = Bandwidth[2] * np.log2(1 + (1/kb) * (PowerWatts[9] / (1e-8 * d)))


print("Bandwidth", Bandwidth)


# Plotting for Spectral efficiency for channel two

# Chanel one
DR1 = [DR_1, DR_2, DR_3, DR_4, DR_5, DR_6, DR_7, DR_8, DR_9, DR_10]
SE1 = [SE_1, SE_2, SE_3, SE_4, SE_5, SE_6, SE_7, SE_8, SE_9, SE_10]
# Chanel two
DR2 = [DR_21, DR_22, DR_23, DR_24, DR_25, DR_26, DR_27, DR_28, DR_29, DR_210]
SE2 = [SE_21, SE_22, SE_23, SE_24, SE_25, SE_26, SE_27, SE_28, SE_29, SE_210]
# Chanel three
DR3 = [DR_31, DR_32, DR_33, DR_34, DR_35, DR_36, DR_37, DR_38, DR_39, DR_310]
SE3 = [SE_31, SE_32, SE_33, SE_34, SE_35, SE_36, SE_37, SE_38, SE_39, SE_310]

plt.figure(3)
plt.plot(Bandwidth, DR1, '-ro')
plt.plot(Bandwidth, DR2, '-go')
plt.plot(Bandwidth, DR3, '-bo')
plt.xlabel('Bandwidth(KHz)')
plt.ylabel('Data rate (bps)')
plt.title("Bandwidth against Total Data Rate")
plt.grid(True, which='both')
plt.minorticks_on()
plt.legend(['channel 1', 'channel 2', 'channel 3'], loc='upper left')
plt.show()

plt.figure(6)
plt.plot(Bandwidth, SE1, '-ro')
plt.plot(Bandwidth, SE2, '-go')
plt.plot(Bandwidth, SE3, '-bo')
plt.xlabel('Bandwidth(KHz)')
plt.ylabel('Spectral efficiency(bits/sec/hz)')
plt.title("Bandwidth against Total Spectral Efficiency")
plt.grid(True, which='both')
plt.minorticks_on()
plt.legend(['channel 1', 'channel 2', 'channel 3'], loc='upper left')
plt.show()

# calculating Datarate for users [1...10]
DDatarate1 = sg * Bandwidth[0] * np.log2(1 + (PowerWatts[0] * Hi) / (10 * (PowerWatts[0] * Hm + 2 * PowerWatts[0] * Hp + NF * No * Ws)))
DDatarate2 = sg * Bandwidth[1] * np.log2(1 + (PowerWatts[1] * Hi) / (20 * (PowerWatts[7] * Hm + 10 * PowerWattsP[8] * Hp + NF * No * Ws)))
DDatarate3 = sg * Bandwidth[2] * np.log2(1 + (PowerWatts[2] * Hi) / (30 * (PowerWatts[8] * Hm + 30 * PowerWattsP[9] * Hp + NF * No * Ws)))
DDatarate4 = sg * Bandwidth[3] * np.log2(1 + (PowerWatts[3] * Hi) / (40 * (PowerWatts[9] * Hm + 8 * PowerWattsP[9] * Hp + NF * No * Ws)))
DDatarate5 = sg * Bandwidth[4] * np.log2(1 + (PowerWatts[4] * Hi) / (50 * (PowerWatts[8] * Hm + phi * PowerWattsP[7] * Hp + NF * No * Ws)))
DDatarate6 = sg * Bandwidth[5] * np.log2(1 + (PowerWatts[5] * Hi) / (60 * (PowerWatts[5] * Hm + phi * PowerWattsP[5] * Hp + NF * No * Ws)))
DDatarate7 = sg * Bandwidth[6] * np.log2(1 + (PowerWatts[6] * Hi) / (70 * (PowerWatts[9] * Hm + phi * PowerWattsP[6] * Hp + NF * No * Ws)))
DDatarate8 = sg * Bandwidth[7] * np.log2(1 + (PowerWatts[7] * Hi) / (80 * (PowerWatts[9] * Hm + phi * PowerWattsP[7] * Hp + NF * No * Ws)))
DDatarate9 = sg * Bandwidth[8] * np.log2(1 + (PowerWatts[8] * Hi) / (90 * (PowerWatts[9] * Hm + phi * PowerWattsP[8] * Hp + NF * No * Ws)))
DDatarate10 = sg * Bandwidth[9] * np.log2(1 + (PowerWatts[9] * Hi) / (100 * (PowerWatts[9] * Hm + phi * PowerWattsP[9] * Hp + NF * No * Ws)))

# calculating Datarate for users [1...10]
specEFF1 = sg * np.log2(1 + (PowerWatts[0] * Hi) / (10 * (PowerWatts[0] * Hm + 2 * PowerWatts[0] * Hp + NF * No * Ws)))
specEFF2 = sg * np.log2(1 + (PowerWatts[1] * Hi) / (20 * (PowerWatts[7] * Hm + 10 * PowerWattsP[8] * Hp + NF * No * Ws)))
specEFF3 = sg * np.log2(1 + (PowerWatts[2] * Hi) / (30 * (PowerWatts[8] * Hm + 30 * PowerWattsP[9] * Hp + NF * No * Ws)))
specEFF4 = sg * np.log2(1 + (PowerWatts[3] * Hi) / (40 * (PowerWatts[7] * Hm + 10 * PowerWattsP[9] * Hp + NF * No * Ws)))
specEFF5 = sg * np.log2(1 + (PowerWatts[4] * Hi) / (50 * (PowerWatts[8] * Hm + 30 * PowerWattsP[8] * Hp + NF * No * Ws)))
specEFF6 = sg * np.log2(1 + (PowerWatts[5] * Hi) / (60 * (PowerWatts[7] * Hm + 10 * PowerWattsP[5] * Hp + NF * No * Ws)))
specEFF7 = sg * np.log2(1 + (PowerWatts[6] * Hi) / (70 * (PowerWatts[8] * Hm + 30 * PowerWattsP[9] * Hp + NF * No * Ws)))
specEFF8 = sg * np.log2(1 + (PowerWatts[7] * Hi) / (80 * (PowerWatts[7] * Hm + 10 * PowerWattsP[9] * Hp + NF * No * Ws)))
specEFF9 = sg * np.log2(1 + (PowerWatts[8] * Hi) / (90 * (PowerWatts[8] * Hm + 30 * PowerWattsP[9] * Hp + NF * No * Ws)))
specEFF10 = sg * np.log2(1 + (PowerWatts[9] * Hi) / (100 * (PowerWatts[7] * Hm + 10 * PowerWattsP[9] * Hp + NF * No * Ws)))

TotalDatarate = [DDatarate1, DDatarate2, DDatarate3, DDatarate4, DDatarate5, DDatarate6, DDatarate7, DDatarate8, DDatarate9, DDatarate10]
TotalspecEFF = [specEFF1, specEFF2, specEFF3, specEFF4, specEFF5, specEFF6, specEFF7, specEFF8, specEFF9, specEFF10]
# ------------------------------------------------------------------------------------------------------------------------------------------------------
DR1, SE1 = calculate_values(Bandwidth[0], PowerWatts, kb, d) # Channel 1
DR2, SE2 = calculate_values(Bandwidth[1], PowerWatts, kb, d) # Channel 2
DR3, SE3 = calculate_values(Bandwidth[2], PowerWatts, kb, d) # Channel 3
DR4, SE4 = calculate_values(Bandwidth[8], PowerWatts, kb, d) # Channel 2
# plot spectrul efficiency against power
plot_data_vs_power(
    PowerDBM, [SE1, SE2, SE3, SE4], 
    ['-ro', '-bo', '-go', "-ko"], 
    ['Scheme 1', 'Scheme 2', 'Scheme 4', "5G"], 
    'SE (bits/sec/Hz)', 
    'SE (bits/sec/Hz) vs Power (dBm)', 
    9
)

# plot data rate against power
plot_data_vs_power(
    PowerDBM, 
    [DR1, DR2, DR3, DR4], 
    ['-ro', '-bo', '-go', "-ko"], 
    ['Scheme 1', 'Scheme 2', 'Scheme 3', "5G "], 
    'Data Rate (bps)', 
    'Data Rate vs Power (dBm)', 
    10
)

print("DR4", DR4)
print("SE", SE4)
# ------------------------------------------------------------------------------------------------------------------------------------------------------