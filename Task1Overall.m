% Task 1: Channel Fading in Wireless Communications
% Task 1.3
function Pr_dBW = hex_pathloss(fc, Pt, d, sigma)
% received power using hex cell pathloss model
% fc(GHz), Pt(W), d(m), sigma(dB)

% Convert power to dBW
Pt_dBW = 10*log10(Pt);

% Hex Path loss model (dB)
PL = 36.7*log10(d) + 22.7 + 26*log10(fc);

% Shadow fading (Gaussian random variable)
X_sigma = sigma * randn;

% Received Power 
Pr_dBW = Pt_dBW - PL + X_sigma;

% % Results
% fprintf('Received power: %.2f dBW\n', Pr_dBW);

end

% % task 1.3 - verify against task 1.1 (sigma=0 should give -96.10 dBW)
% hex_pathloss(1, 1, 100, 0)

%% Task 1.4 - 1000 Shadow Fading Realisations
clear; clc;

% Parameters (Task 1.1 values)
fc = 1;        % GHz
Pt = 1;        % Watts
d = 100;       % metres
sigma = 4;     % dB
Pmin = -104.1; % dBW (minimum acceptable signal level)
N = 1000;      % number of realisations

% Monte Carlo simulation: Generate N received power samples by calling function N times
Pr = zeros(1, N);
for i = 1:N
    Pr(i) = hex_pathloss(fc, Pt, d, sigma);
end

% Simulated coverage probability: Check the percentage of samples over the minimum received power
Pc_sim = sum(Pr >= Pmin) / N;

% Theoretical coverage probability (from Task 1.2) Q((-104.1 - (-96.1)) / 4) = Q(-2) = 0.9772
Pc_theory = qfunc((-104.1 - (-96.1)) / sigma);

% Display results
fprintf('Simulated  coverage probability: %.4f (%.2f%%)\n', Pc_sim, Pc_sim*100);
fprintf('Theoretical coverage probability: %.4f (%.2f%%)\n', Pc_theory, Pc_theory*100);

% Plot histogram. PDF of received power across 1000 realisations
figure;
histogram(Pr, 40, 'Normalization', 'pdf');
xline(Pmin, 'r--', 'LineWidth', 2);
xlabel('Received Power (dBW)');
ylabel('PDF');
title('Distribution of Received Signal Power (1000 realisations)');
legend('Simulated P_r', 'Coverage Threshold P_{min}');
grid on;
