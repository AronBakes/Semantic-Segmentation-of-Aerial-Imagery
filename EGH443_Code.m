%% Clearing Workspace and Importing Required Data
clc; clear all;
load('A2Data.mat');

set(0, 'DefaultFigurePosition', [550, 250, 800*0.7, 600*0.7]);

% Notes:
% - Simulating a 5G-based OFDM system.
% - System must provide reliable broadband in an urban environment.
% - 200 MHz bandwidth centred around the 24 GHz spectrum.
% - Guard interval duration is 7% of the OFDM symbol duration.
% - The IFFT/FFT size is 4096.

BW = 200e6; % Bandwidth, Hz

%% Task 1: Designing and Simulating the Performance of OFDM in an AWGN Channel
%  1.1 OFDM System Design

SCS     = 240e3;     % Sub-carrier spacing, Hz
N       = 4096;      % FFT size, 2^12
T       = 1 / (SCS); % OFDM symbol duration, s
Tg      = 0.07 * T;  % Guard interval duration, s
Ts      = T;         % Total OFDM symbol duration, s
Tsample = T / N;    % Sample duration, s

% Finding active sub-carriers and resource blocks available
N_active = floor(BW / SCS); % Number of data-carrying sub-carriers.
BW_u = N_active * SCS;      % Bandwidth used by the data-carrying sub-carriers.
RBs = floor(N_active / 12); % Resource blocks supported by system (1 RB is 12 sub-carriers across 14 OFDM symbols).

fprintf('1.1 OFDM System Design:\n');
fprintf('OFDM Symbol Duration (T)       = %.3f us\n', T * 1e6);
fprintf('Guard Interval Duration (Tg)   = %.3f us\n', Tg * 1e6);
fprintf('Total Symbol Duration (Ts)     = %.3f us\n', Ts * 1e6); 
fprintf('Sample Duration (Tsample)      = %.3f ns\n', Tsample * 1e9);
fprintf('Active Sub-carriers (N_active) = %d\n', N_active);
fprintf('Total Occupied Bandwidth (BW_u)= %.3f MHz\n', BW_u / 1e6);
fprintf('Resource Blocks (RBs)          = %d\n', RBs);
fprintf('\n');

%% 1.2 Listing data rates for 16-QAM, 64-QAM, 256-QAM and 1024-QAM

% Symbol rate and bits per symbol
sym_rate = 1 / Ts;                  % OFDM symbol rate (with CP), Hz
mod_orders = [16, 64, 256, 1024];   % Modulation orders
bits_per_symbol = log2(mod_orders);

% Data rate
base_data_rates = bits_per_symbol .* N_active .* sym_rate; % bits/sec
data_rates = 0.9 * base_data_rates; % 10% overhead 

% Applying LDPC 3/4 coding
LDPC_data_rates = data_rates * (3/4);

% Displaying table
fprintf('1.2 Comparing Modulation Schemes:\n');
T = table(mod_orders.', bits_per_symbol.', (data_rates / 1e6).', (LDPC_data_rates / 1e6).', ...
    'VariableNames', {'Modulation Order', 'Bits/Symbol', 'Data Rate [Mbps] (no LDPC)', 'Data Rate [Mpbs] (with LDPC)'});
disp(T)
fprintf('\n');

% Plotting
f1 = figure(1); clf;
plot(bits_per_symbol, (data_rates / 1e6), '-o', 'LineWidth', 2); hold on;
plot(bits_per_symbol, (LDPC_data_rates / 1e6), '-s', 'LineWidth', 2);
grid on;
xlabel('Bits per Symbol [log_2(M)]');
ylabel('Data Rate [Mbps]');
title('Data Rate vs Modulation Scheme');
legend('Without LDPC', 'With 3/4 LDPC');
saveas(gcf, 'figs/modulation_vs_datarate.png');

%% 1.3 Comparing to Equivalent Single Carrier System

base_data_rates_SC = bits_per_symbol .* sym_rate; % Single-carrier bit rate
data_rates_SC = 0.9 * base_data_rates_SC;         % 10% overhead
LDPC_data_rates_SC = data_rates_SC * (3/4);       % Applying the 3/4 LDPC

fprintf('1.3 Comparing to Equivalent Single Carrier System:\n');
T = table(mod_orders.', bits_per_symbol.', (data_rates_SC / 1e6).', (LDPC_data_rates_SC / 1e6).', ...
    'VariableNames', {'Modulation Order', 'Bits/Symbol', 'Data Rate [Mbps] (no LDPC)', 'Data Rate [Mpbs] (with LDPC)'});
disp(T)
fprintf('\n');

%% 1.4 Simulating Performance in AWGN Channel - 64-QAM

% Parameters
M = 64;                % 64-QAM
mu = log2(M);          % Bits/symbol
nloops = 1000;         % Maximum number of bits transmitted
Ncp = ceil(0.07 * N_active); % Guard interval

EbN0Vec = (0:2:50);
snrVec = EbN0Vec + 10*log10(mu);
num_berrs = zeros(1, length(EbN0Vec));

% Subcarrier mapping indices, center the active subcarriers
start_idx = floor((N - N_active)/2) + 1;
active_indices = start_idx:start_idx + N_active - 1;

for n = 1:length(EbN0Vec)
    for l = 1:nloops                           % Errors collected for a given SNR to maxBitErrors.
        dataIn = randi([0 M-1], N_active, 1);  % Message signal
        modulated_symbols = qammod(dataIn, M, 'InputType', 'integer', 'UnitAveragePower', 1); % Modulating random data
       
        fd_signal = zeros(N, 1);                       % Create frequency domain with all zeros length 4096
        fd_signal(active_indices) = modulated_symbols; % Fill in 833 spaces with the modulated symbols
        
        OFDM = ifft(fd_signal) * sqrt(N);      % Generating OFDM symbol by taking IFFT
        OFDM_CP = [OFDM(end-Ncp+1:end); OFDM]; % Add cyclic prefix
        ynoisy = awgn(OFDM_CP, snrVec(n));     % AWGN Channel

        OFDM_Rx = ynoisy(Ncp+1:end);                    % Remove cyclic prefix
        OFDM_decode = fft(OFDM_Rx, N) / sqrt(N);        % Convert back to frequency domain
        received_symbols = OFDM_decode(active_indices); % Extract active subcarriers

        dataOut = qamdemod(received_symbols, M, 'OutputType', 'integer', 'UnitAveragePower', 1); % Demodulation
        num_berrs(n) = num_berrs(n) + biterr(dataIn, dataOut, mu);                               % Collecting errors
    end
end

% Calculate BER
berS = num_berrs / (nloops * N_active * mu);
berT = berawgn(EbN0Vec, 'qam', M);

% Plot results
f2 = figure(2); clf;
semilogy(EbN0Vec, berS, '*');
hold on
semilogy(EbN0Vec, berT, 'r');
legend('Simulated', 'Theoretical', 'Location', 'Best');
xlabel('Eb/No [dB]'); 
ylabel('Bit Error Rate');
title('AWGN OFDM System');
grid on
saveas(gcf, 'figs/ber_ofdm_awgn.png');


%% 2.2 Simulating the complex Gaussain RV 

% Simulating Gaussian random variable
K = 1e6;         % Number of samples
X = randn(1, K); % Real part
Y = randn(1, K); % Imaginary part
h = X + 1i*Y;    % Complex Gaussian
envelope = abs(h);

% Plot histogram
f3 = figure(3); clf;
histogram(envelope, ...
    'Normalization', 'pdf', ...
    'DisplayName', 'Simulated Envelope', ...
    'FaceColor', [0.6, 0.8, 1]); % Light blue
hold on;

% Theoretical Rayleigh PDF
a = linspace(0, 5, 1000);
sigma = 1;
fA = (a./sigma^2) .* exp(-a.^2 / (2*sigma^2));
plot(a, fA, 'r--', 'LineWidth', 2, 'DisplayName', 'Theoretical Rayleigh PDF');

xlabel('Amplitude');
ylabel('Probability Density');
title('Simulating the complex Gaussain RV');
legend('Location', 'northeast');
grid on;
saveas(gcf, 'figs/rayleigh_envelope.png');

%% 2.3 Simulating Performance in an AWGN & Rayleigh Fading Channel

% Parameters
M = 64;              % 64-QAM
mu= log2(M);         % Bits/symbol
nloops = 1000;       % Maximum number of bits transmitted
Ncp = ceil(0.07*N); % Guard interval

EbN0Vec = (0:2:30);
snrVec = EbN0Vec + 10*log10(mu);
num_berrs = zeros(1, length(EbN0Vec));

% Subcarrier mapping indices, center the active subcarriers
start_idx = floor((N - N_active)/2) + 1;
active_indices = start_idx:start_idx + N_active - 1;

for n=1:length(EbN0Vec)
    
    for l = 1:nloops                           % Errors collected for a given SNR to maxBitErrors.
        dataIn = randi([0 M-1], N_active, 1);  % Message signal
        
        modulated_symbols =  qammod(dataIn, M, 'InputType', 'integer', 'UnitAveragePower', 1); % Modulating random data
        
        fd_signal = zeros(N, 1);                       % Create frequency domain with all zeros length 4096
        fd_signal(active_indices) = modulated_symbols; % Fill in 833 spaces with the modulated symbols

        OFDM = ifft(fd_signal) * sqrt(N);      % Generating OFDM symbol by taking IFFT
        OFDM_CP = [OFDM(end-Ncp+1:end); OFDM]; % Add cyclic prefix

        h = 1/sqrt(2) * (randn + 1i*randn);  % Flat fading coefficient (time domain)
        OFDM_fad = conv(OFDM_CP, h);
        ynoisy = awgn(OFDM_fad, snrVec(n));  % Add AWGN
        
        OFDM_Rx = ynoisy(Ncp+1:end);                    % Removing the cyclic prefix
        OFDM_decode = fft(OFDM_Rx, N) / sqrt(N);        % Convert back to frequency domain
        received_symbols = OFDM_decode(active_indices); % Extract active subcarriers
        
        yeq = received_symbols ./ h; % Flat channel, so all subcarriers equalised the same
        
        dataOut = qamdemod(yeq, M, 'OutputType', 'integer', 'UnitAveragePower', 1); % Demodulation

        num_berrs(n) = num_berrs(n) + biterr(dataIn,dataOut,mu); % Collecting errors
    end
end

% Determining the theoretical BER.
[berT,~] = berfading(EbN0Vec,'qam',M,1);
berS = num_berrs/(nloops*N_active*mu);

f4 = figure(4); clf;
semilogy(EbN0Vec,berS,'*');
hold on
semilogy(EbN0Vec,berT,'r');
legend('Simulated','Theoretical','Location','Best');
xlabel('Eb/No [dB]'); ylabel('Bit Error Rate');
title('OFDM in flat Rayleigh Channel');
grid on
saveas(gcf, 'figs/ber_ofdm_rayleigh_flat.png');




%% 2.4 Performance of OFDM system in Multipath Rayleigh Channel
% 
% % Parameters
% M = 64;              % 64-QAM
% mu= log2(M);         % Bits/symbol
% nloops = 1000;       % Maximum number of bits transmitted
% Ncp = ceil(0.07*N); % Guard interval
% 
% % Delay and power profile
% tvec_samples = round((tvec * 1e-9) / (Ts / N)); % Interested at what sample does the signal arrive
% L = length(tvec_samples);   % Number of different paths
% p_linear = 10.^(pvec / 10); % Converting from dB to linear
% alpha = 1 / sum(p_linear);  % Normalisation factor
% sigma = alpha * p_linear;   % Normalised power
% 
% EbN0Vec = (0:2:30);
% snrVec = EbN0Vec + 10*log10(mu);
% num_berrs = zeros(1, length(EbN0Vec));
% 
% % Subcarrier mapping indices, center the active subcarriers
% start_idx = floor((N - N_active)/2) + 1;
% active_indices = start_idx:start_idx + N_active - 1;
% 
% for n=1:length(EbN0Vec)
% 
%     for l = 1:nloops                           % Errors collected for a given SNR to maxBitErrors.
%         dataIn = randi([0 M-1], N_active, 1);  % Message signal
% 
%         modulated_symbols =  qammod(dataIn, M, 'InputType', 'integer', 'UnitAveragePower', 1); % Modulating random data
% 
%         fd_signal = zeros(N, 1);                       % Create frequency domain with all zeros length 4096
%         fd_signal(active_indices) = modulated_symbols; % Fill in 833 spaces with the modulated symbols
% 
%         OFDM = ifft(fd_signal) * sqrt(N);      % Generating OFDM symbol by taking IFFT
%         OFDM_CP = [OFDM(end-Ncp+1:end); OFDM]; % Add cyclic prefix
% 
%         % The receiver receives the signals spread acrosss the sample range with the normalised powers found above.
%         h_taps = zeros(max(tvec_samples)+1, 1);
%         for i = 1:L
%             delay = tvec_samples(i) + 1;
%             h_taps(delay) = h_taps(delay) + sqrt(sigma(i)/2)*(randn + 1i*randn);
%         end
% 
%         OFDM_mult = conv(OFDM_CP, h_taps);   % Transmit the OFDM sybol through the multipath channel
%         ynoisy = awgn(OFDM_mult, snrVec(n)); % Add AWGN
% 
%         OFDM_Rx = ynoisy(Ncp+1:end);                    % Removing the cyclic prefix
%         OFDM_decode = fft(OFDM_Rx, N) / sqrt(N);        % Convert back to frequency domain
%         received_symbols = OFDM_decode(active_indices); % Extract active subcarriers
% 
%         fadF = fft(h_taps, N);        % Transfer function of channel
%         fadF = fadF(active_indices);  
% 
%         yeq = received_symbols./fadF; % Channel equalisaion using one tap equalisation method
% 
%         dataOut = qamdemod(yeq, M, 'OutputType', 'integer', 'UnitAveragePower', 1); % Demodulation 
%         num_berrs(n) = num_berrs(n) + biterr(dataIn,dataOut,mu);
% 
%     end
% end
% 
% berS = num_berrs/(nloops*N_active*mu);
% 
% f5 = figure(5); clf;
% semilogy(EbN0Vec,berS,'*-');
% hold on
% legend('Simulated','Location','Best');
% xlabel('Eb/No [dB]'); ylabel('Bit Error Rate');
% title('2.4: Performance of an OFDM system in Multipath Rayleigh Channel')
% grid on
% saveas(gcf, 'figs/ber_ofdm_rayleigh_multipath.png');
% 


%% 2.4 Performance of OFDM system in Multipath Rayleigh Channel
%% Section 3.1: LDPC-Coded vs Uncoded OFDM BER in Multipath Fading

M = 64;
mu = log2(M);
R = 3/4;
N = 4096;
N_active = 833;
Z = 108;
Ncp = ceil(0.07 * N);

P = [16 17 22 24  9  3 14 -1  4  2  7 -1 26 -1  2 -1 21 -1  1  0 -1 -1 -1 -1
     25 12 12  3  3 26  6 21 -1 15 22 -1 15 -1  4 -1 -1 16 -1  0  0 -1 -1 -1
     25 18 26 16 22 23  9 -1  0 -1  4 -1  4 -1  8 23 11 -1 -1 -1  0  0 -1 -1
      9  7  0  1 17 -1 -1  7  3 -1  3 23 -1 16 -1 -1 21 -1  0 -1 -1  0  0 -1
     24  5 26  7  1 -1 -1 15 24 15 -1  8 -1 13 -1 13 -1 11 -1 -1 -1 -1  0  0
      2  2 19 14 24  1 15 19 -1 21 -1  2 -1 24 -1  3 -1  2  1 -1 -1 -1 -1  0];

H = ldpcQuasiCyclicMatrix(Z, P);
cfgEnc = ldpcEncoderConfig(H);
cfgDec = ldpcDecoderConfig(H);

EbN0Vec = (0:2:30);
snr_coded = EbN0Vec + 10*log10(mu) + 10*log10(R);
snr_uncoded = EbN0Vec + 10*log10(mu);

load('A2Data.mat', 'pvec', 'tvec');
L = length(pvec);
sigma = pvec;
tvec_samples = round(tvec);

start_idx = floor((N - N_active)/2) + 1;
active_indices = start_idx : start_idx + N_active - 1;

nloops = 200;
max_iterations = 20;
ber_ldpc = zeros(size(EbN0Vec));
ber_uncoded = zeros(size(EbN0Vec));

ber = comm.ErrorRate; 
ber2 = comm.ErrorRate;

for n = 1:length(snr_uncoded)
    disp(["SNR step " num2str(n)]);
    for l = 1:nloops
        data = randi([0 1], cfgEnc.NumInformationBits, 1, 'int8');

        encodedBits = ldpcEncode(data, cfgEnc);
        qam_coded = qammod(encodedBits, M, 'InputType', 'bit', 'UnitAveragePower', true);
        qam_uncoded = qammod(data, M, 'InputType', 'bit', 'UnitAveragePower', true);

        X_coded = zeros(N, 1);
        X_uncoded = zeros(N, 1);
        X_coded(active_indices(1:length(qam_coded))) = qam_coded;
        X_uncoded(active_indices(1:length(qam_uncoded))) = qam_uncoded;

        x_coded = [ifft(X_coded, N) * sqrt(N)];
        x_uncoded = [ifft(X_uncoded, N) * sqrt(N)];
        x_coded_cp = [x_coded(end-Ncp+1:end); x_coded];
        x_uncoded_cp = [x_uncoded(end-Ncp+1:end); x_uncoded];

        h_taps = zeros(max(tvec_samples)+1, 1);
        for i = 1:L
            h_taps(tvec_samples(i)+1) = h_taps(tvec_samples(i)+1) + sqrt(sigma(i)/2)*(randn + 1i*randn);
        end
        h_freq = fft(h_taps, N);

        y_coded = awgn(conv(x_coded_cp, h_taps), snr_coded(n), 'measured');
        y_uncoded = awgn(conv(x_uncoded_cp, h_taps), snr_uncoded(n), 'measured');

        y_c = y_coded(Ncp+1:Ncp+N);
        y_u = y_uncoded(Ncp+1:Ncp+N);
        Y_c = fft(y_c, N) / sqrt(N);
        Y_u = fft(y_u, N) / sqrt(N);

        H_c = h_freq(active_indices);
        Rx_c = Y_c(active_indices(1:length(qam_coded))) ./ H_c(1:length(qam_coded));
        Rx_u = Y_u(active_indices(1:length(qam_uncoded))) ./ H_c(1:length(qam_uncoded));

        noiseVar = 10.^(-snr_coded(n)/10);
        llr = qamdemod(Rx_c, M, 'OutputType', 'approxllr', 'UnitAveragePower', true, 'NoiseVariance', noiseVar);
        decodedBits = ldpcDecode(llr, cfgDec, max_iterations);
        errStats = ber(data, decodedBits);

        dataOut = qamdemod(Rx_u, M, 'OutputType', 'integer', 'UnitAveragePower', true);
        dataOut_bits = de2bi(dataOut, mu, 'left-msb');
        dataOut_bits = reshape(dataOut_bits.', [], 1);
        errStatsNoCoding = ber2(data, dataOut_bits);

        ber_ldpc(n) = ber_ldpc(n) + errStats(1);
        ber_uncoded(n) = ber_uncoded(n) + errStatsNoCoding(1);
    end
    ber_ldpc(n) = ber_ldpc(n) / nloops;
    ber_uncoded(n) = ber_uncoded(n) / nloops;
    reset(ber);
    reset(ber2);
end

% Plot results separately and combined
figure; semilogy(EbN0Vec, ber_uncoded, '-o');
xlabel('$E_b/N_0$ [dB]', 'Interpreter', 'latex');
ylabel('Bit Error Rate');
title('Uncoded OFDM in Multipath Fading');
grid on;
saveas(gcf, 'figs/uncoded_multipath_separate.png');

figure; semilogy(EbN0Vec, ber_ldpc, '-*');
xlabel('$E_b/N_0$ [dB]', 'Interpreter', 'latex');
ylabel('Bit Error Rate');
title('LDPC-Coded OFDM in Multipath Fading');
grid on;
saveas(gcf, 'figs/ldpc_multipath_separate.png');

figure;
semilogy(EbN0Vec, ber_ldpc, '-*'); hold on;
semilogy(EbN0Vec, ber_uncoded, '--');
xlabel('$E_b/N_0$ [dB]', 'Interpreter', 'latex');
ylabel('Bit Error Rate');
legend('LDPC-Coded', 'Uncoded', 'Location', 'southwest');
title('BER of Coded vs Uncoded OFDM in Multipath Rayleigh Fading');
grid on;
saveas(gcf, 'figs/ber_ldpc_vs_uncoded.png');

