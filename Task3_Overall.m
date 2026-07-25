% ES96T Assignment (Please do not distribute the code without agreement)
% This is a template code that you need to complete.
% Acknowledgement to L.Lanante, H.Yin, and S.Deronne (University of Washington)

clc;
clear;
close all;

%% initialize bitrate for data and ack channels in IEEE 802.11a
% %6Mbps
% data_rate = 6e6;
% ack_rate = 6e6;
% 
% %9Mbps
% data_rate = 9e6;
% ack_rate = 6e6;
% 
% %12Mbps
% data_rate = 12e6;
% ack_rate = 12e6;
% 
% %18Mbps
% data_rate = 18e6;
% ack_rate = 12e6;
% 
% %24Mbps
% data_rate = 24e6;
% ack_rate = 24e6;
% 
% %36Mbps
% data_rate = 36e6;
% ack_rate = 24e6;
% 
% %48Mbps
% data_rate = 48e6;
% ack_rate = 24e6;

%54Mbps
data_rate = 54e6;
ack_rate = 24e6;

%% saving results in a local file
fid = fopen('bianchi_11a.txt', 'wt');
node = [5:5:50];

[throughput] = CSMA11a_YUAN(data_rate, ack_rate);

% Changes to code to create plots
legend_labels = {'m=3, CWmin=32', 'm=5, CWmin=32', 'm=3, CWmin=128'};
colors        = {'-ob', '-sr', '-^g'};

fprintf(fid, "// Data rate: 54 Mbps\n");
for s = 1:3
    fprintf(fid, "// %s\n", legend_labels{s});
    for k = 1:10
        fprintf(fid, "        {%d, %.4f},\n", node(k), throughput(s, k));
    end
    fprintf(fid, "\n");
end
fclose(fid);

%% plot the results
%*** Hint: you might put multiple curves in a single graph (feel free to modify the code) ******
figure; hold on; grid on;
for s = 1:3
    plot(node, throughput(s, :), colors{s}, 'LineWidth', 1.5, ...
        'DisplayName', legend_labels{s});
end
title('Throughput Simulation of Bianchi Model');
xlabel('Number of Stations');
ylabel('Effective Throughput');
legend('Location', 'best');
hold off;

%% Task 3.3.1: generate CSV dataset for ML training
rates_ml      = [6e6, 9e6, 12e6, 18e6, 24e6, 36e6, 48e6, 54e6];
ack_ml        = [6e6, 6e6, 12e6, 12e6, 24e6, 24e6, 24e6, 24e6];
param_sets_ml = [3,  32;
                 5,  32;
                 3, 128];
payload_sizes = [1000, 2000, 4000];

SIFS = 16e-6; DIFS = 34e-6; SLOT = 9e-6; PROP = 1e-6;
SYMBOL = 4e-6; PHY_PREAM = 20e-6; SERVICE = 16; TAIL = 6;
LEN_ACK = 14*8; MAC_HEADER = (24+4)*8; APP_HEADER = 8*6;

fid_csv = fopen('dataset_task3.csv', 'w');
fprintf(fid_csv, 'n_stations,m,CWmin,data_rate_Mbps,payload_bytes,throughput\n');

for dr = 1:length(rates_ml)
    dr_val  = rates_ml(dr);
    ack_val = ack_ml(dr);
    for ps = 1:length(payload_sizes)
        LEN_DATA     = payload_sizes(ps) * 8;
        EP           = LEN_DATA / SLOT;
        BITS_SYMBOL  = dr_val * SYMBOL;
        DATA_SYMBOLS = ceil((SERVICE + MAC_HEADER + LEN_DATA + APP_HEADER + TAIL) / BITS_SYMBOL);
        T_DATA       = PHY_PREAM + (SYMBOL * DATA_SYMBOLS);
        BITS_SYMBOL  = ack_val * SYMBOL;
        DATA_SYMBOLS = ceil((SERVICE + LEN_ACK + TAIL) / BITS_SYMBOL);
        T_ACK        = PHY_PREAM + (SYMBOL * DATA_SYMBOLS);
        T_S = T_DATA + SIFS + T_ACK + DIFS + PROP;
        T_C = T_DATA + DIFS + PROP;
        for s = 1:size(param_sets_ml, 1)
            m     = param_sets_ml(s, 1);
            CWmin = param_sets_ml(s, 2);
            W     = CWmin;
            for j = 1:length(node)
                n  = node(j);
                eq = @(p) p - (1 - (1 - (2*(1-2*p)) / ...
                    ((1-2*p)*(W+1) + p*W*(1-(2*p)^m)))^(n-1));
                p   = fsolve(eq, 0.1, optimset('Display','off'));
                tau = 2*(1-2*p) / ((1-2*p)*(W+1) + p*W*(1-(2*p)^m));
                Ptr = 1 - (1-tau)^n;
                Ps  = n*tau*(1-tau)^(n-1) / Ptr;
                tp  = (Ps*Ptr*EP) / (Ps*Ptr*T_S + Ptr*(1-Ps)*T_C + (1-Ptr)*SLOT);
                fprintf(fid_csv, '%d,%d,%d,%.1f,%d,%.4f\n', ...
                    n, m, CWmin, dr_val/1e6, payload_sizes(ps), tp);
            end
        end
    end
end

fclose(fid_csv);

%% Define a function that calculates throughput
function [throughput] = CSMA11a_YUAN(data_rate, ack_rate)
node = [5:5:50];

%******* You need to change the values of CWmin and CWmax as required*******
% %initia values have been given
% CWmin = 32; % not useful since briefing provides values
% CWmax = 256; % not useful since briefing provides values
param_sets = [3,  32;
              5,  32;
              3, 128];
%******* End of your input**************

% data size in bits
LEN_DATA = 4000 * 8;
% ACK size in bits
LEN_ACK = 14 * 8;
% MAC header size in bits
MAC_HEADER = (24 + 4) * 8;
% bits added by the upper layer(s)
APP_HEADER = 8 * 6;
% duration of control slots
SIFS = 16e-6;
DIFS = 34e-6;
SLOT = 9e-6;
% propagation delay
PROP = 1e-6;

%**************** (1) Your input: initialize 802.11a parameters*******
% OFDM symbol duration
SYMBOL = 4e-6;
% PHY preamble duration in seconds
PHY_PREAM = 20e-6;
% service field length in bits
SERVICE = 16;
% tail length in bits
TAIL = 6;
%**************** (1) End of your input*******

% data bits per OFDM symbol
BITS_SYMBOL = data_rate * SYMBOL;
DATA_SYMBOLS = ceil((SERVICE + MAC_HEADER + LEN_DATA + APP_HEADER + TAIL)/BITS_SYMBOL);
%T_DATA is sum of PHY_Header+MAC_Header+Payload
T_DATA = PHY_PREAM + (SYMBOL * DATA_SYMBOLS);

% number of data bits per OFDM symbol
BITS_SYMBOL = ack_rate * SYMBOL;
DATA_SYMBOLS = ceil((SERVICE + LEN_ACK + TAIL)/BITS_SYMBOL);
T_ACK = PHY_PREAM + (SYMBOL * DATA_SYMBOLS);

%average packet payload size.
EP = LEN_DATA / SLOT;

%***** (2) Your Input: Calculate time slot for successful transmission ********
T_S = T_DATA + SIFS + T_ACK + DIFS + PROP;
%***** (2) End of your input************

%***** (3) Your Input: Calculate time slot for collided transmission ********
T_C = T_DATA + DIFS + PROP;
%***** (3) End of your input ********

% calculate Bianchi's throughput for each parameter set
throughput = zeros(size(param_sets, 1), length(node));

for s = 1:size(param_sets, 1)
    m     = param_sets(s, 1);
    CWmin = param_sets(s, 2);

    Bianchi = zeros(size(node));
    for j = 1:length(node)
        n = node(j)*1;
        W = CWmin;

        %***** (4) Your Input: calculate collision probability, p *********
        eq = @(p) p - (1 - (1 - (2*(1-2*p)) / ((1-2*p)*(W+1) + p*W*(1-(2*p)^m)))^(n-1));
        p = fsolve(eq, 0.1, optimset('Display','off'));
        %***** (4) End of your input ********

        % Equation(7) in Bianchi's paper.
        tau = 2*(1 - 2*p) / ((1 - 2*p)*(W + 1) + p*W*(1 - (2*p)^m));

        %Equation(10) in Bianchi's paper
        Ptr = 1 - (1 - tau)^n;

        %equation(11) in Bianchi's paper
        Ps = n*tau*(1 - tau)^(n - 1)/Ptr;

        %***** (5) Your Input: calculate throughput*********
        Bianchi(j) = (Ps * Ptr * EP) / (Ps * Ptr * T_S + Ptr * (1 - Ps) * T_C + (1 - Ptr) * SLOT);
        %***** (5) End of your input ********
    end

    throughput(s, :) = Bianchi;
end

end