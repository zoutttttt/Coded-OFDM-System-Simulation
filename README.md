# Coded-OFDM-System-Simulation
Simulation of Forward Error Correction (FEC) applied OFDM system in a multi-path fading Rayleigh channel using MATLAB

The simulation requires the following files to be in a same directory to run: 
1. channel.mat
2. Assignment2.m (run this file for the actual simulation)
3.initialise.m


The Simulation program comprises of five sections:

1. Modelling a  path loss exponent (n) of the channel using measured recieved power and distance.
2. Designing an appropriate OFDM system given a system requirmenet of 20MHz and minimum data rate of 
120 Mbps
3. Simulating the designed OFDM system in a flat fading channel 
4. Simulating the designed OFDM system in a multipath fading channel 
5. Improving the system BER performance by incorporating a FEC (CRC and convolutional coding)

Each of the system iteration's performance was assessed using its BER peroformance for a given Eb/No range.

# Results: 
Upon implementation of OFDM system, the BER performance of flat fading channel and multipath fading channel showed almost identical shape. This underlines OFDM system's immunity towards time dispersive channel. 

OFDM effectively introduces immunity to these propagation delay by its use of guard interval. The key principle here is that low symbol rate modulation schemes suffer less from intersymbol interference, which is the result of multi-path propagation. With OFDM, the wideband channel is now split into multiple narrowband channels which turns the single high-rate stream into multiple slower rate data transmissions in parallel. This is comparable to a high-pressure water stream as opposed to wide shower head that splits into multiple streams – the total rate of flow is identical for both. Now that the duration of each symbol is long, now at the beginning of each symbol it is preceded with a guard interval. The idea here is that any delays that falls within this interval will not affect the actual data as at the receiver, we remove this guard interval. 

Additionally, with the implementation of FEC on top of the OFDM system, the BER performance improved - nonetheless, for lower Eb/No, the BER actually worsened. This trend is explained by the nature of coded system. Every convolutional code system has fixed limit to the error correcting capability, and if the number of occurred errors exceed this limit, the code system is unable to correct it and therefore the BER response will be poor. Also, it must be noted here that convolutional codes scheme adds redundant data and cause lower data rate and less energy per bit. 

It was observed that data rate and reliability of the communication system cannot easily be improved concurrently. This is an unavoidable dilemma in telecommunications design, as clearly the two factors compromise each other. Therefore, for such tasks where we are given to design a communication system, it is vitally important that we clearly identify the requirement of the system, see how much headspace we have, and apply error correcting schemes accordingly. In our case, we were unable to achieve the implementation of error control schemes while also achieving the required data rate of 120 Mbps.  
