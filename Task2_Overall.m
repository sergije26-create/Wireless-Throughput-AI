% Task 2: Radio Resource Allocation
% Based on Yu, W. "Multiuser water-filling in the presence of crosstalk", ITA 2007
close all; clear; clc;

%% Task 2.1 - 2-Cell Simulation Space

% Each cell is 1 km x 1 km
% BS placed at centre of each cell, UT placed randomly within it
cell_size = 1; % (km)
BS1 = [0.5, 0.5]; % centre of cell 1
BS2 = [1.5, 0.5]; % centre of cell 2

% Fixed seed for reproducibility - same positions used in Tasks 2.2 and 2.3
rng(42);
UT1 = [rand, rand];             % uniform random within cell 1: x,y in [0,1]
UT2 = [rand + cell_size, rand]; % uniform random within cell 2: x in [1,2], y in [0,1]

% Plot simulation space
figure;
hold on;

% Cell boundaries drawn as dashed rectangles
rectangle('Position', [0, 0, 1, 1], 'EdgeColor', 'b', 'LineWidth', 2, 'LineStyle', '--');
rectangle('Position', [1, 0, 1, 1], 'EdgeColor', 'r', 'LineWidth', 2, 'LineStyle', '--');

% Base stations use triangle markers as specified in assignment
plot(BS1(1), BS1(2), 'b^', 'MarkerSize', 12, 'MarkerFaceColor', 'b', 'DisplayName', 'BS Cell 1');
plot(BS2(1), BS2(2), 'r^', 'MarkerSize', 12, 'MarkerFaceColor', 'r', 'DisplayName', 'BS Cell 2');

% User terminals use circle markers as specified in assignment
plot(UT1(1), UT1(2), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'cyan', 'DisplayName', 'UT Cell 1');
plot(UT2(1), UT2(2), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'magenta', 'DisplayName', 'UT Cell 2');

% Text labels for each marker
text(BS1(1)+0.03, BS1(2)+0.05, 'BS1', 'FontSize', 10); % BS1 label
text(BS2(1)+0.03, BS2(2)+0.05, 'BS2', 'FontSize', 10); % BS2 label
text(UT1(1)+0.03, UT1(2)+0.05, 'UT1', 'FontSize', 10); % UT1 label
text(UT2(1)+0.03, UT2(2)+0.05, 'UT2', 'FontSize', 10); % UT2 label

xlim([0, 2]); ylim([0, 1]);
xlabel('x (km)'); ylabel('y (km)');
title('Task 2.1: 2-Cell Simulation Space');
legend('Location', 'best');
grid on; axis equal; % ensures 1 km looks the same in x and y directions
hold off;

% Direct link distances from each BS to its corresponding UT
d1 = norm(UT1 - BS1); % magnitude of vector from BS1 to UT1
d2 = norm(UT2 - BS2); % magnitude of vector from BS2 to UT2
fprintf('UT1 distance from BS1: %.4f km\n', d1);
fprintf('UT2 distance from BS2: %.4f km\n\n', d2);

%% Task 2.2 - Channel Model and Iterative Water Filling

% System parameters from assignment spec
N = 5;          % number of OFDM subchannels
K = 2;          % number of cells/users
fc = 2.4;       % carrier frequency in GHz
sigma = 4;      % shadow fading standard deviation in dB
alpha = [1, 1]; % user weighting coefficients, equal priority for both users

% Convert power and noise from dB to linear Watts
P_max = 10^(10/10);        % 10 dBW = 10 W total power budget per BS
N0_W  = 10^((-114-30)/10); % -114 dBm converted to Watts

% Iteration limits and convergence thresholds for the water filling algorithm
max_outer = 100;
max_inner = 100;
tol_inner = 1e-8;
tol_outer = 1e-8;

% Build full distance matrix d_m(j,k) = distance in metres from BS j to UT k
% Needed for both direct links (j=k) and cross-cell interference links (j~=k)
BS_pos = [BS1; BS2];
UT_pos = [UT1; UT2];
d_km = zeros(K, K);
for j = 1:K
    for k = 1:K
        d_km(j,k) = norm(BS_pos(j,:) - UT_pos(k,:));
    end
end
d_m = d_km * 1000; % path loss formula requires distance in metres

fprintf('Distances (m):\n');
fprintf('  BS1->UT1 (direct): %.2f m\n', d_m(1,1));
fprintf('  BS2->UT2 (direct): %.2f m\n', d_m(2,2));
fprintf('  BS1->UT2 (cross):  %.2f m\n', d_m(1,2));
fprintf('  BS2->UT1 (cross):  %.2f m\n\n', d_m(2,1));

% Task 2.2a - Channel Model

% Path loss (dB) for each BS-UT pair using the model from the assignment:
% PL = 36.7*log10(d) + 22.7 + 26*log10(fc), d in metres, fc in GHz
PL_dB = zeros(K, K);
for j = 1:K
    for k = 1:K
        PL_dB(j,k) = 36.7*log10(d_m(j,k)) + 22.7 + 26*log10(fc);
    end
end

% Shadow fading modelled as Gaussian in dB (equivalent to log-normal in linear)
% Independent across subchannels and across links as per assignment spec
SF_dB = sigma * randn(K, K, N);

% Channel power gain |H_jk(n)|^2 in linear scale
% Total attenuation (dB) = path loss + shadow fading, then converted to linear
% H2(j,k,n) = power gain from BS j to UT k on subchannel n
H2 = zeros(K, K, N);
for j = 1:K
    for k = 1:K
        for n = 1:N
            H2(j,k,n) = 10^(-(PL_dB(j,k) + SF_dB(j,k,n)) / 10);
        end
    end
end

% Task 2.2b - Iterative Multi-User Water Filling

% Initialise power to equal split across subchannels
P = ones(K, N) * (P_max / N);
% Crosstalk penalty t_k(n) initialised to zero - no interference assumed initially
t = zeros(K, N);

% Algorithm has two nested loops (Yu 2007):
%   Outer loop: updates t_k(n) via eq.(19) until convergence
%   Inner loop: for fixed t, each user water-fills via eq.(22)/(23)
for outer = 1:max_outer
    t_old = t;

    for inner = 1:max_inner
        P_old_inner = P;

        for k = 1:K
            % Interference at UT k from all other BSs on each subchannel
            % I_k(n) = sum_{j~=k} P_j(n)*|H_jk(n)|^2, eq.(20)
            I_k = zeros(1, N);
            for j = 1:K
                if j ~= k
                    for n = 1:N
                        I_k(n) = I_k(n) + P(j,n) * H2(j,k,n);
                    end
                end
            end

            % Water filling floor: noise + interference normalised by direct channel gain
            % Higher floor means worse conditions on that subchannel, so less power allocated
            floor_k = (I_k + N0_W) ./ squeeze(H2(k,k,:))';

            % Bisection search for water level lambda_k
            % eq.(23): P_k(n) = max(0, alpha_k/(lambda_k + t_k(n)) - floor_k(n))
            % We need sum_n P_k(n) = P_max, so bisect to find the lambda achieving this
            lam_lo = 0;
            lam_hi = alpha(k) / min(floor_k + 1e-30);
            for bisect = 1:100
                lam_mid = (lam_lo + lam_hi) / 2;
                P_k_try = max(0, alpha(k)./(lam_mid + t(k,:)) - floor_k);
                if sum(P_k_try) > P_max
                    lam_lo = lam_mid; % lambda too low, allocated power exceeds budget
                else
                    lam_hi = lam_mid; % lambda too high, not using full power budget
                end
                if (lam_hi - lam_lo) < 1e-12
                    break;
                end
            end

            % Apply eq.(22) with converged water level
            lam_k  = (lam_lo + lam_hi) / 2;
            P(k,:) = max(0, alpha(k)./(lam_k + t(k,:)) - floor_k);
        end

        if max(abs(P(:) - P_old_inner(:))) < tol_inner
            break;
        end
    end

    % Update crosstalk penalty t_k(n) using eq.(19)
    % t_k(n) captures the marginal cost of BS k transmitting on subchannel n
    % in terms of harm caused to other users' achievable rates
    % Built from two terms:
    %   frac1: SINR sensitivity - how much user j's rate changes per unit signal power
    %   frac2: cross-channel gain normalised by the interference floor at UT j
    t_new = zeros(K, N);
    for k = 1:K
        for n = 1:N
            sum_t = 0;
            for j = 1:K
                if j ~= k
                    sig_j = P(j,n) * H2(j,j,n); % signal power received at UT j from BS j

                    % Total interference at UT j from all other BSs plus noise
                    int_at_j = N0_W;
                    for l = 1:K
                        if l ~= j
                            int_at_j = int_at_j + P(l,n) * H2(l,j,n);
                        end
                    end
                    int_excl_j = int_at_j - N0_W; % interference component only

                    % First term in eq.(19)
                    frac1 = (alpha(j) * sig_j) / (sig_j + int_excl_j + N0_W);
                    % Second term in eq.(19)
                    frac2 = H2(k,j,n) / (int_excl_j + N0_W);

                    sum_t = sum_t + frac1 * frac2;
                end
            end
            t_new(k,n) = sum_t;
        end
    end
    t = t_new;

    if max(abs(t(:) - t_old(:))) < tol_outer
        fprintf('Converged after %d outer iterations.\n\n', outer);
        break;
    end
end

% Print final power allocation
fprintf('Final power allocation [W]:\n');
for k = 1:K
    fprintf('  User %d: ', k);
    fprintf('%.4f  ', P(k,:));
    fprintf('| Total: %.4f W\n', sum(P(k,:)));
end
fprintf('\n');

%% Task 2.3 - Sum Rate and Power Allocation Plot

% Task 2.3a - Sum rate per user

R = zeros(K, 1);
for k = 1:K
    % Recompute interference at UT k using the final converged power values
    I_k_final = zeros(1, N);
    for j = 1:K
        if j ~= k
            for n = 1:N
                I_k_final(n) = I_k_final(n) + P(j,n) * H2(j,k,n);
            end
        end
    end

    % SINR per subchannel: desired signal over noise plus interference
    SINR_k = (P(k,:) .* squeeze(H2(k,k,:))') ./ (N0_W + I_k_final);

    % Shannon capacity summed across all subchannels gives total rate [bits/s/Hz]
    C_k = alpha(k) * log2(1 + SINR_k);
    R(k) = sum(C_k);

    fprintf('User %d: R = %.4f bits/s/Hz | per subchannel: ', k, R(k));
    fprintf('%.3f  ', C_k);
    fprintf('\n');
end
fprintf('Network sum rate: %.4f bits/s/Hz\n\n', sum(R));

% Task 2.3b - Power allocation plot

figure;
hold on; grid on;
bar_w = 0.35;
x = 1:N;
b1 = bar(x - bar_w/2, P(1,:)*1000, bar_w, 'FaceColor', [0.2 0.45 0.8]);
b2 = bar(x + bar_w/2, P(2,:)*1000, bar_w, 'FaceColor', [0.85 0.33 0.1]);

% Dashed reference line showing what equal power allocation would look like
yline(P_max/N*1000, 'k--', 'LineWidth', 1.5, ...
    'Label', sprintf('Equal power (%.1f mW/sub)', P_max/N*1000));

xlabel('Subchannel index n');
ylabel('Allocated power (mW)');
title('Task 2.3b: Water Filling Power Allocation per Subchannel');
legend([b1, b2], {'User 1 (BS1)', 'User 2 (BS2)'}, 'Location', 'best');
xticks(1:N);
xlim([0.5, N+0.5]);
hold off;
