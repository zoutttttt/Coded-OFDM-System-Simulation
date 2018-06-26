%% Part 1
clear all, close all, clc
load('channel.mat')

% Q1 - Present the measured path loss data at different distances in a scatter plot.
figure(1)
plot(d,LdBShadow, '+')  
xlabel('Distance (m)');
ylabel('Path Loss (dB)');
title('Path Loss in varying distances');
hold off 

%Q2 - For each of the distances in d, calculate the average path loss
avg_LdBShadow = mean(LdBShadow,1);

%Q3 - Plot the results of the averaged path loss
figure(2)
plot(d,avg_LdBShadow)
xlabel('Distance (m)');
ylabel('Path Loss (dB)');
title('Average Path Loss in varying distances');


%Q4 - Use the MATLAB built-in fuction fitlm() to find a linear model for the given data
ddB = 10*log10(d);  % convert from metres to logscale

%plot the average path loss over distance
figure(3)
plot(ddB,avg_LdBShadow,'b.')
xlabel('log-scale Distance (dB)');
ylabel('Path Loss (dB)');
title('Average Path Loss in varying log-scale distances');

lin_param = fitlm(ddB,avg_LdBShadow);
intercept = lin_param.Coefficients{1,1};    % y-intercept
x1 = lin_param.Coefficients{2,1};    % coefficient of x

dtemp = linspace(min(ddB),max(ddB),100);
LavgdBEstemated = intercept +x1*dtemp;

% plot the graph of both theoretical and experimenatl model and compare 
hold on 
plot(dtemp,LavgdBEstemated,'r')
legend('experimental', 'linear model')
hold off
%Q5 - Estimate the path loss exponent.
PathLossExponent = x1; 

%Q6 - Estimate the path loss at a distance of 12 km away from the transmitter
LdB12 = intercept + PathLossExponent*10*log10(12e3);

%Q7 - Assuming transmitter and receiver antenna gains of 10dB each, receiver sensitivity of -100
%dBm, and fade margin of 20 dB estimate the minimum transmitted power required in task 6.
Rx_sensitivity = -100;  %unit: dBm
fade_margin = 20;   %unit: dB
Gt = 10;    %unit: dBi
Gr = 10;    %unit: dBi

Tx_dBm = (Rx_sensitivity+fade_margin)-Gt+LdB12-Gr;  % Required Tx power in dBm
Tx_W = (10^(Tx_dBm/10))/1000;  % Required Tx Power in Watt

% Q8 - For each of the distances in d, calculate the standard deviation.
sigmadB = std(LdBShadow);

% Q9 - Use this information to find an estimate for the standard deviation of X?
sigmaXdB =  mean(sigmadB);

%% Part 2 - Designing an OFDM System
clear all, close all, clc
load('channel.mat')

% Q1 - Plot and label the multipath impulse response of the channel
figure(1)
% Create a stem plot of the relative power in dB (y) against time in ns (x)
stem(tvec,pvec,'basevalue',-30)
xlabel('Time (ns)')
ylabel('Relative Power (dB)')
title('Multipath Impulse Response')

% Q2 - Estimating the RMS delay spread of the channel
pvec_W = 10.^(pvec/10); % convert dB to mW
% Calculate the mean excess delay
t_delay = (sum(pvec_W.*tvec)/sum(pvec_W)); % unit: ns
t_delay_squared = (sum(pvec_W.*(tvec.^2))/sum(pvec_W)); % delay squared
%  Calcualte the RMS delay spread
delay_spread  = sqrt(t_delay_squared-t_delay^2)*10^-9; % unit: s

% Q3 - Estimating the coherence bandwidth of the channel 
% Assuming 0.5 frequency correlation
Bc = 1/(5*delay_spread);

% Q4 - Designing an OFDM System
% Initialise the design requirements
Rb = 120*10^6; % Bit rate (120Mbps)
BW = 20*10^6; % Channel bandwidth (20MHz)
Tmax = 1300e-9; % from tvec

% Assumptions
dirac = 0.1; % proportion of gaurd interval
Rc = 1; % code rate
beta = 0.25; 

% Using the modulating index (u) to find the no. of sub carriers (N)
% NOTE: u = log2(M)
u = (Rb/BW)*(1+beta); % raised-cosine pulse shaping
u =ceil(u); % add one for u has to be greater than (Rb/BW)*(1+beta)
N = (Rb*Tmax)/(dirac*Rc*u); % N = 223
N = 2^8; % N was chosen as 2^8 = 256 as it was closest base2 value

% Using delta_f to find the symbol duration
delta_f = BW/N;
Ts = 1/delta_f; % symbol period

% Guard interval Tg > 10*delay_spread hence set Tg to 10*delay_spread
% rounded up to the nearest 7th decimal place since dealing with micro
% seconds
Tg = round(10*delay_spread,7);
T_total = Ts + Tg; % symbol period + guard 

% Checking the following conditions
%   Condition 1: Tg > 10*delay_spread
%   Condition 2: delta_f <= Bc/10
check1 = Tg > 10*delay_spread;
check2 = delta_f <= Bc/10;

fprintf('Condition 1 - Tg > 10*delay_spread\n');
if check1
    fprintf('TRUE: %f > %f\n\n', Tg, 10*delay_spread)
end 
fprintf('Condition 2 - delta_f <= Bc/10\n');
if check2
    fprintf('TRUE: %f <= %f\n\n', delta_f, Bc/10);
end

% Q5 -  Estimate the maximum data rate and bandwidth of your system
% Evaluating the data rate of the OFDM system designed in Q4
R = (N*u)/(T_total); 

% Evaluating the BW of the system
BW_system = R/u; % 17.181 MHz (< 20MHz channel requirement)

% Bandwidth of Single Carrier system
T_single = 1/BW;
BW_single = BW/u; % BW_single = 2.8571MHz 

%% Q7 - Simulating the performance of an OFDM system in a AWGN channel
% Simulate and plot the bit error rate performance within the Eb/N0 range from 0dB to
% 15dB, and compare with the theoretical bit-error rate performance.

M = 2^(u); % alphabet size
k = u;  % bits/Symbol
Nsyms = N; % number of modulated symbols
maxNumBits = 1e5;

% Calculate OFDM Parameters
ofdmBw = BW;
deltaF = ofdmBw/Nsyms; % OFDM sub carrier separation
Tsamp = Ts/Nsyms; % sample time
Ncp = floor(Tg/Tsamp)+1;  % number of samples for Guard interval

% Use communication toolbox method for M-QAM modulation simulation.
% Instantiate two objects hMod and hDemod with specified M symbols.
hMod = comm.RectangularQAMModulator('ModulationOrder',M);
hDemod = comm.RectangularQAMDemodulator('ModulationOrder',M);
EbNoVec = (0:15)'; % range OF EbNo to be explored (0 dB to 15 dB)
snrVec = EbNoVec + 10*log10(k); % Find SNR from given EbNo

errorRate = comm.ErrorRate('ResetInputPort',true);
bervec_AWGN = zeros(length(EbNoVec),1);  % simulated BER vector
errorStats = zeros(1,3); % vector used for capturing errorStats from .errorRate function. 
% the actual BER will be calculated using biterr() function but errorStats
% will be used for controlling the while loop to ensure the data captures
% sufficient rounds of run.

% calculates the modulator scale factor for M-QAM
input = 0:M-1;
const = qammod(input,M);
scale = modnorm(const,'avpow',1);

for n=1:length(EbNoVec)
    
    % These two variables are used for averaging the BER over 
    % number of test runs in the while loop
    count = 0;
    berSUM = 0;
    
    while errorStats(3) <= maxNumBits % This will limit the number of tests ran
        dataIn = randi([0 M-1],Nsyms,1);    % Message signal generated in symbols. 
        dataIn_bin = reshape(de2bi(dataIn),k*Nsyms,1);  % converts the message signal to bits
        y =scale* hMod(dataIn);   %modulate the data
        OFDM = ifft(y)*sqrt(Nsyms); %% Generate the OFDM symbol by taking IFFT of the modulated symbols
        OFDM_CP = [OFDM(Nsyms-Ncp+1:Nsyms); OFDM];  % cyclic prefix needed for OFDM operation
        ynoisy = awgn(OFDM_CP,snrVec(n));  %add AWGM to the OFDM symbols 
        OFDM_Rx = ynoisy(Ncp+1:(length(OFDM_CP)));  % remove cyclic prefix by truncating removing the beginning bits
        OFDM_decode=fft(OFDM_Rx,Nsyms)/sqrt(Nsyms); %FFT Block at the receiver
        dataOut = step(hDemod,OFDM_decode/scale); % Demodulate to recover the message sing hDemod object
        dataOut_bin = reshape(de2bi(dataOut,k),k*Nsyms,1); % converts the received symbols to bits 
        [NUMBER,RATIO] = biterr(dataOut_bin,dataIn_bin);    % calculate bit error. use RATIO variable for BER
        errorStats = errorRate(dataIn,dataOut,0);     % Collect error statistics - used for checking maxNumbits.
        count = count +1 ;  % increase the counter - used for keeping track of how many runs the while loop iterates
        berSUM = berSUM + RATIO;    % sums the BER for each run. Later this value will be used to calcalate average BER
    end % end while loop
    errorStats = errorRate(dataIn,dataOut,1);     % reset errorStats for next EbNO value
    bervec_AWGN(n) = berSUM/count;  %add average BER to the simulated ber vector  
end % end for loop 

% Theoretical BER in AWGN setting 
theoryBer_AWGN = berawgn(EbNoVec, 'qam',M);

% Plot comparing theoretical and simulated BER
figure(5)
semilogy (EbNoVec,theoryBer_AWGN,'g','linewidth',2);
hold on
semilogy(EbNoVec,bervec_AWGN,'*')
legend('Theoretical','Simulated','Location','Best')
xlabel('Eb/No (dB)')
ylabel('Bit Error Rate')
title('OFDM Performance in AWGN - 256 QAM')
grid on
hold off

%% Part 3
%randn produces random values with a mean of 0 and a standard distribution
%of 1 variance = sqrt(standard deviation)  = 1, so randn produces values
%that match the specification by default
%randn generates numbers in a normal distribution with
%mean of 0 and variance of 1, so no modifications needs to be done
X = randn(10000,1);
Y = randn(10000,1);

%generate h and the envelope of h
h = X + i.*Y;
envelope = sqrt(X.^2 + Y.^2);

%50 bins is adequate for the interval
%this function creates the unnormalized histogram
[freq,bin] = hist(envelope,50);
figure(6);
%by dividing the frequency by the area of the histogram it sets the area of
%the function to 1, meaning it is a valid pdf, so it will match up with the
%rayleigh pdf function below
bar(bin,freq/trapz(bin,freq));

sigma = 1;
x = 0:2.5/10000:6;
%generate rayleigh distribution with given formula
fA = x/sigma^2.*exp((-x.^2)/(2*sigma^2));
figure(6)
hold on
plot(x,fA);
xlabel('Envelope')
ylabel('probability')
title('Theoretical Rayleigh Distribution vs simulated result')
hold off

%% Question 2

% Use M-QAM modulation. Same way as part 2 and part 3 
hMod = comm.RectangularQAMModulator('ModulationOrder',M);
hDemod = comm.RectangularQAMDemodulator('ModulationOrder',M);
EbNoVec_2 = (0:30)'; % EbNo vector (0 - 30 dB)
snrVec = EbNoVec_2 + 10*log10(k); % Find SNR from given EbNo

% Set the errorRate array and BER array
errorRate = comm.ErrorRate('ResetInputPort',true);
berVec_flat = zeros(length(EbNoVec_2),1);
errorStats = zeros(1,3);

% iterate through each EbNo vector and calculate average BER.
for n=1:length(EbNoVec_2),
    
    % These two variables are used for averaging the BER over 
    % number of test runs in the while loop
    berSUM = 0;
    count = 0;
    
    % The while loop ensures we capture good BER result with high enough
    % max number of bits
    while errorStats(3) <= maxNumBits 
        dataIn = randi([0 M-1],Nsyms,1);  % Message signal in symbol forms
        dataIn_bin = reshape(de2bi(dataIn),k*Nsyms,1);  % message signal in binary form
        y = scale*step(hMod,dataIn);  % modulate the data using hMod object
        OFDM = ifft(y)*sqrt(Nsyms); % Generate the OFDM symbol by taking IFFT of the modulated symbols
        OFDM_CP =  [OFDM(Nsyms-Ncp+1:Nsyms); OFDM];  % Append cyclic prefix to the beginning of the symbols
        % Genrate a mutli-path fading channel
        fade = zeros(Nsyms,1); % create empty vector with length of total symbols
        fade(1) = 1/sqrt(2)*(randn(1,1)+1i*randn(1,1)); % Generate Rayleigh fading channel
        % Transmit signal through a fading + AWGN channel.
        OFDM_fad = filter(fade,1,OFDM_CP); % apply multi-fading
        ynoisy = awgn(OFDM_fad(1:length(OFDM_CP)),snrVec(n));   % add AWGN 
        OFDM_Rx=ynoisy(Ncp+1:(length(OFDM_CP))); % remove cyclic prefix
        OFDM_decode=fft(OFDM_Rx,Nsyms)/sqrt(Nsyms); % decode the message by fft 
        fadF = fft(fade,Nsyms);    % Single tap euqalisation to remove the channel effect.
        yeq = OFDM_decode./fadF;
        dataOut = step(hDemod,yeq/scale);   % Demodulate to recover the message.  
        dataOut_bin = reshape(de2bi(dataOut),k*Nsyms,1);    % received message in binary form 
        [error1, ratio1] = biterr(dataIn_bin,dataOut_bin);  % calculate current run's BER
        berSUM = berSUM + ratio1;     % sum of the BER for each iteration. will be used to calculate average BER.
        count = count + 1;  % counter for keeping track of total number of iteration
        errorStats = errorRate(dataIn,dataOut,0);     % check bits used doesnt exceed maxNumBits. If yes, exit while loop
    end % end while loop 
    errorStats = errorRate(dataIn,dataOut,1);         % Reset the error rate calculator
    berVec_flat(n) = berSUM/count;   % calculate the average BER for each EbNo
end % end for loop

% plot BER performance in AWGN vs RAYLEIGH channel
figure(7)
theoryBER_flat = berfading(EbNoVec_2, 'qam',M,1);
semilogy (EbNoVec_2,theoryBER_flat,'g','linewidth',2);
hold on
semilogy(EbNoVec_2,berVec_flat,'*')
semilogy (EbNoVec,theoryBer_AWGN,'r','linewidth',2);
semilogy(EbNoVec,bervec_AWGN,'*')
legend('theory-flat','simulation-flat','theory-AWGN','simulated-AWGN','Location','Best')
xlabel('Eb/No (dB)')
ylabel('Bit Error Rate')
title('OFDM performance in Rayleigh Fading vs AWGN Channel')
grid on
hold off


%% Part 4 

% Calculate multi-path channel parameters
fad_pdB = pvec; % multipath delay vector in dB
fad_p = 10.^(fad_pdB/10);   % multipath delay vector in W
alpha = 1/sum(fad_p);   % estimated normalisation factor 
sigma_tau = alpha.*fad_p;   % variance of each path 
delayT = tvec*10^-9;    % Time delay with correct unit (nano-seconds)
delaySample = floor(delayT/Tsamp)+1;    % number of samples for each delay 

% Use M-QAM modulation. Same way as part 2 and part 3 
hMod = comm.RectangularQAMModulator('ModulationOrder',M);
hDemod = comm.RectangularQAMDemodulator('ModulationOrder',M);
EbNoVec_2 = (0:30)'; % EbNo vector (0 - 30 dB)
snrVec = EbNoVec_2 + 10*log10(k); % Find SNR from given EbNo

% Set the errorRate array and BER array
errorRate = comm.ErrorRate('ResetInputPort',true);
berVec_multifade = zeros(length(EbNoVec_2),1);
errorStats = zeros(1,3);

% iterate through each EbNo vector and calculate average BER.
for n=1:length(EbNoVec_2),
    
    % These two variables are used for averaging the BER over 
    % number of test runs in the while loop
    berSUM = 0;
    count = 0;
    
    % The while loop ensures we capture good BER result with high enough
    % max number of bits
    while errorStats(3) <= maxNumBits 
        dataIn = randi([0 M-1],Nsyms,1);  % Message signal in symbol forms
        dataIn_bin = reshape(de2bi(dataIn),k*Nsyms,1);  % message signal in binary form
        y = scale*step(hMod,dataIn);  % modulate the data using hMod object
        OFDM = ifft(y)*sqrt(Nsyms); % Generate the OFDM symbol by taking IFFT of the modulated symbols
        OFDM_CP =  [OFDM(Nsyms-Ncp+1:Nsyms); OFDM];  % Append cyclic prefix to the beginning of the symbols
        % Genrate a mutli-path fading channel
        fad_multipath = zeros(Nsyms,1); % create empty vector with length of total symbols
        fad_multipath(delaySample) = sqrt(sigma_tau/2)'.*(randn(length(fad_pdB),1)+1i*randn(length(fad_pdB),1)); % Generate Rayleigh fading channel
        % Transmit signal through a fading + AWGN channel.
        OFDM_fad = filter(fad_multipath,1,OFDM_CP); % apply multi-fading
        ynoisy = awgn(OFDM_fad(1:length(OFDM_CP)),snrVec(n));   % add AWGN 
        OFDM_Rx=ynoisy(Ncp+1:(length(OFDM_CP))); % remove cyclic prefix
        OFDM_decode=fft(OFDM_Rx,Nsyms)/sqrt(Nsyms); % decode the message by fft 
        fadF = fft(fad_multipath,Nsyms);    % Single tap euqalisation to remove the channel effect.
        yeq = OFDM_decode./fadF;
        dataOut = step(hDemod,yeq/scale);   % Demodulate to recover the message.  
        dataOut_bin = reshape(de2bi(dataOut),k*Nsyms,1);    % received message in binary form 
        [error1, ratio1] = biterr(dataIn_bin,dataOut_bin);  % calculate current run's BER
        berSUM = berSUM + ratio1;     % sum of the BER for each iteration. will be used to calculate average BER.
        count = count + 1;  % counter for keeping track of total number of iteration
        errorStats = errorRate(dataIn,dataOut,0);     % check bits used doesnt exceed maxNumBits. If yes, exit while loop
    end % end while loop 
    errorStats = errorRate(dataIn,dataOut,1);         % Reset the error rate calculator
    berVec_multifade(n) = berSUM/count;   % calculate the average BER for each EbNo
end % end for loop

theoryBer_multifade = berfading(EbNoVec_2,'qam',M,1);  % calculate theoretical BER under fading condition 

% plot flat fading channel and multifading channel and observe the difference 
figure(8)
semilogy(EbNoVec_2,berVec_flat,'*');   % simulated BER for multipath-fading channel
hold on 
semilogy(EbNoVec_2,theoryBER_flat,'k','linewidth',2); % theoeretical BER for multipath-fading channel
semilogy(EbNoVec_2,berVec_multifade,'*');   % simulated BER for multipath-fading channel
semilogy(EbNoVec_2,theoryBer_multifade,'g','linewidth',2); % theoeretical BER for multipath-fading channel
legend('Simulation-flat','theory-flat','simulation-multifade','Theory-multifade','Location','Best')
xlabel('Eb/No (dB)')
ylabel('Bit Error Rate')
title('BER performance in multipath fading channel vs flatfading channel')
grid on
hold off

%% Part 5 Q1

% Calculate multi-path channel parameters
fad_pdB = pvec; % multipath delay vector in dB
fad_p = 10.^(fad_pdB/10);   % multipath delay vector in W
alpha = 1/sum(fad_p);   % estimated normalisation factor 
sigma_tau = alpha.*fad_p;   % variance of each path 
delayT = tvec*10^-9;    % Time delay with correct unit (nano-seconds)
delaySample = floor(delayT/Tsamp)+1;    % number of samples for each delay 

% code properties
% Set the trellis structure and traceback length for a rate 1/2, constraint length 7, convolutional code.
trellis = poly2trellis(7,[171 133]);
tbl = 30;
rate = 1/2;

EbNoVec_2 = (0:30)'; % EbNo vector (0 - 30 dB)
snrVec = EbNoVec_2 + 10*log10(k*rate); % Find SNR from given EbNo

% Set the errorRate array and BER array
errorRate = comm.ErrorRate('ResetInputPort',true);
berVec_CC = zeros(length(EbNoVec_2),1);

% iterate through each EbNo vector and calculate average BER.
for n=1:length(EbNoVec_2),
    
    % These two variables are used for averaging the BER over 
    % number of test runs in the while loop

    [numErrsSoft,numErrsHard,numBits] = deal(0);
    % The while loop ensures we capture good BER result with high enough
    % max number of bits
    while numBits < 5e5 
        dataIn = randi([0 1],Nsyms*k,1);  % Message signal in bitsm
        % Convolutionally encode the data
        dataEnc = convenc(dataIn,trellis);
        y = qammod(dataEnc,M,'InputType','bit');  % modulate the data using hMod object
        OFDM = ifft(y)*sqrt(Nsyms); % Generate the OFDM symbol by taking IFFT of the modulated symbols
        OFDM_CP =  [OFDM(Nsyms*2-Ncp*2+1:Nsyms*2); OFDM];  % Append cyclic prefix to the beginning of the symbols
        % Genrate a mutli-path fading channel
        fad_multipath = zeros(Nsyms*2,1); % create empty vector with length of total symbols
        fad_multipath(delaySample) = sqrt(sigma_tau/2)'.*(randn(length(fad_pdB),1)+1i*randn(length(fad_pdB),1)); % Generate Rayleigh fading channel
        % Transmit signal through a fading + AWGN channel.
        OFDM_fad = filter(fad_multipath,1,OFDM_CP); % apply multi-fading
        ynoisy = awgn(OFDM_fad(1:length(OFDM_CP)),snrVec(n),'measured');   % add AWGN 
        OFDM_Rx=ynoisy(Ncp*2+1:(length(OFDM_CP))); % remove cyclic prefix
        OFDM_decode=fft(OFDM_Rx,Nsyms*2)/sqrt(Nsyms); % decode the message by fft 
        fadF = fft(fad_multipath,Nsyms*2);    % Single tap euqalisation to remove the channel effect.
        yeq = OFDM_decode./fadF;
        dataOut = qamdemod(yeq,M,'OutputType','bit'); % Demodulate to recover the message. 
        % Viterbi decode the demodulated data
        dataHard = vitdec(dataOut,trellis,tbl,'cont','hard');
        % Calculate the number of bit errors in the frame. Adjust for the
        % decoding delay, which is equal to the traceback depth.
        numErrsInFrameHard = biterr(dataIn(1:end-tbl),dataHard(tbl+1:end));
        % Increment the error and bit counters
        numErrsHard = numErrsHard + numErrsInFrameHard;
        numBits = numBits + Nsyms*k;
        
    end % end while loop 
    berVec_CC(n) = numErrsHard/numBits;   % calculate the average BER for each EbNo
end % end for loop

% plot flat fading channel and multifading channel and observe the difference 
figure(9)
semilogy(EbNoVec_2,berVec_multifade,'*');   % simulated BER for multipath-fading channel
hold on 
semilogy(EbNoVec_2,theoryBer_multifade,'k','linewidth',2); % theoeretical BER for multipath-fading channel
semilogy(EbNoVec_2,berVec_CC,'-','linewidth',2);   % simulated BER for multipath-fading channel
legend('Simulation-multifade','thoery-multifade','simulation-CC','Location','Best')
xlabel('Eb/No (dB)')
ylabel('Bit Error Rate')
title('BER performance: CC encoded vs un-encoded 256-QAM')
grid on
hold off

