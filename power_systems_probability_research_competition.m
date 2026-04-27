%% Probability-Driven Statistical Modeling and Verification of Random Signals and Noise
% Competition-ready MATLAB script for the power-systems probability paper.
% The script reproduces the key figures, tables, and probability metrics
% reported in the IEEE-style paper.
%
% Main ideas demonstrated:
% 1) Stochastic waveform construction for power-system measurements
% 2) Mean-square error expectation model
% 3) Rice / noncentral chi-square probability modeling
% 4) Monte Carlo verification with confidence intervals

clear; close all; clc;
rng(20260414, 'twister');

%% Output folder
outDir = fullfile(pwd, 'power_probability_outputs');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

%% Global parameters
fs = 10000;                 % Hz
f0 = 50;                    % Hz
Ttotal = 1.0;               % s
N = round(fs * Ttotal);
t = (0:N-1).' / fs;
cycleN = fs / f0;           % 200 samples
snrList = [30 20 15 10 5];
MC_DESC = 250;
MC_MSE = 300;
MC_PROB = 5000;

A0 = 1.0;
alpha5 = 0.05;
alpha7 = 0.03;
phi5 = 0.4;
phi7 = -0.8;
transAmp = 0.20;
transFreq = 600;
transDecay = 120;
transT0 = 0.62;
rhoL = 0.9995;
stdL = 0.006;
rhoR = 0.9992;
stdR = 0.008;
gammaR = 0.8;
alphaFA = 0.05;
epsAmp = 0.01;

%% Deterministic components
s0 = A0 * sin(2*pi*f0*t);
harm = alpha5 * sin(2*pi*5*f0*t + phi5) + alpha7 * sin(2*pi*7*f0*t + phi7);
u = double(t >= transT0);
tau = transAmp * exp(-transDecay * (t - transT0)) .* sin(2*pi*transFreq*(t - transT0)) .* u;

Psignal = mean(s0.^2);
Pharm = mean(harm.^2);
Ptau = mean(tau.^2);
PdeltaTheory = 0.5 * (stdL^2 + (gammaR^2) * stdR^2);
MSEfloorTheory = Pharm + Ptau + PdeltaTheory;

%% Representative realization for waveform / PSD figures
rng(101, 'twister');
[~, ~, deltaRep] = slow_random_component(N, rhoL, stdL, rhoR, stdR, gammaR, s0);
sigma20 = noise_sigma_for_snr(20);
n20 = sigma20 * randn(N,1);
x20full = s0 + harm + tau + deltaRep + n20;

%% Fig. 1 - representative waveform
idx = (t >= 0.585) & (t <= 0.675);
fig = figure('Color', 'w', 'Position', [100 100 700 360]);
plot(t(idx), s0(idx), '--', 'LineWidth', 1.0); hold on;
plot(t(idx), x20full(idx), 'LineWidth', 1.2);
xlabel('Time (s)'); ylabel('Amplitude (pu)');
box on;
exportgraphics(fig, fullfile(outDir, 'fig1_waveform.png'), 'Resolution', 300);
close(fig);

%% Fig. 2 - noise histogram and Gaussian fit
fig = figure('Color', 'w', 'Position', [100 100 680 360]);
muN = mean(n20);
stdN = std(n20, 1);
skewN = skewness(n20, 0);
kurtN = kurtosis(n20, 0) - 3;
[counts, edges] = histcounts(n20, 48, 'Normalization', 'pdf');
centers = 0.5*(edges(1:end-1) + edges(2:end));
bar(centers, counts, 1.0, 'FaceAlpha', 0.65, 'EdgeColor', 'none'); hold on;
xpdf = linspace(min(n20), max(n20), 500);
pdfFit = (1/(stdN*sqrt(2*pi))) * exp(-0.5*((xpdf - muN)/stdN).^2);
plot(xpdf, pdfFit, 'LineWidth', 1.6);
text(0.97, 0.96, sprintf('mean=%.4f\nstd=%.4f\nskew=%.3f\nex.kurt=%.3f', ...
    muN, stdN, skewN, kurtN), 'Units', 'normalized', ...
    'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
    'BackgroundColor', 'white', 'Margin', 4);
xlabel('Noise sample value'); ylabel('Density'); box on;
exportgraphics(fig, fullfile(outDir, 'fig2_noise_hist.png'), 'Resolution', 300);
close(fig);

%% Fig. 3 - PSD of representative waveform
[pxx, fpxx] = pwelch(x20full, 2048, [], [], fs);
fig = figure('Color', 'w', 'Position', [100 100 680 360]);
semilogy(fpxx, pxx, 'LineWidth', 1.1); hold on;
for fmark = [50 250 350 600]
    xline(fmark, '--', 'LineWidth', 0.8);
end
text(58, interp1(fpxx, pxx, 50, 'linear', 'extrap') * 1.05, '50 Hz');
text(258, interp1(fpxx, pxx, 250, 'linear', 'extrap') * 1.05, '5th');
text(358, interp1(fpxx, pxx, 350, 'linear', 'extrap') * 1.05, '7th');
text(615, interp1(fpxx, pxx, 600, 'linear', 'extrap') * 1.05, 'Transient');
xlim([0 800]); xlabel('Frequency (Hz)'); ylabel('PSD (pu^2/Hz)'); box on;
exportgraphics(fig, fullfile(outDir, 'fig3_psd.png'), 'Resolution', 300);
close(fig);

%% Table I - parameters
paramNames = {
    'Nominal frequency';
    'Sampling frequency';
    'Record length';
    'Fundamental amplitude';
    'Harmonics';
    'Transient';
    'Load process';
    'Renewable process';
    'Monte Carlo trials';
    'False alarm target'};
paramValues = {
    '50 Hz';
    '10 kHz';
    '1 s (10,000 samples)';
    '1.0 pu';
    '5th = 5%, 7th = 3%';
    '0.2 pu decaying 600 Hz oscillation from 0.62 s';
    'AR(1), rho = 0.9995, std = 0.006 pu';
    'AR(1), rho = 0.9992, std = 0.008 pu';
    '250 (desc.), 300 (MSE), 5000 (prob.)';
    'alpha = 5%'};
Tparams = table(paramNames, paramValues, 'VariableNames', {'Parameter', 'Value'});
writetable(Tparams, fullfile(outDir, 'table1_parameters.csv'));

%% Table II - descriptive statistics at 15 dB
rng(222, 'twister');
sigma15 = noise_sigma_for_snr(15);
descAccum = zeros(4,4);
for k = 1:MC_DESC
    n = sigma15 * randn(N,1);
    [~, ~, delta] = slow_random_component(N, rhoL, stdL, rhoR, stdR, gammaR, s0);
    cases = {s0, s0 + n, s0 + harm + n, s0 + harm + tau + delta + n};
    for c = 1:4
        x = cases{c};
        descAccum(c,1) = descAccum(c,1) + var(x, 1) * N/(N-1);
        descAccum(c,2) = descAccum(c,2) + rms(x);
        descAccum(c,3) = descAccum(c,3) + mean((x - s0).^2);
        descAccum(c,4) = descAccum(c,4) + one_cycle_corr(x, cycleN);
    end
end
descMean = descAccum / MC_DESC;
caseNames = {'Clean fundamental'; 'AWGN only'; 'AWGN + harmonics'; 'Full stochastic case'};
Tdesc = table(caseNames, descMean(:,1), descMean(:,2), descMean(:,3), descMean(:,4), ...
    'VariableNames', {'Case', 'Variance', 'RMS', 'MSE', 'OneCycleCorr'});
writetable(Tdesc, fullfile(outDir, 'table2_descriptive_stats_15dB.csv'));

%% MSE versus SNR - empirical and theoretical
rng(333, 'twister');
meanMSE = zeros(numel(snrList),1);
ciLowMSE = zeros(numel(snrList),1);
ciHighMSE = zeros(numel(snrList),1);
meanMSEtheory = zeros(numel(snrList),1);

for i = 1:numel(snrList)
    sigma = noise_sigma_for_snr(snrList(i));
    vals = zeros(MC_MSE,1);
    for k = 1:MC_MSE
        n = sigma * randn(N,1);
        [~, ~, delta] = slow_random_component(N, rhoL, stdL, rhoR, stdR, gammaR, s0);
        x = s0 + harm + tau + delta + n;
        vals(k) = mean((x - s0).^2);
    end
    meanMSE(i) = mean(vals);
    sVals = std(vals, 1);
    halfWidth = 1.96 * sVals / sqrt(MC_MSE);
    ciLowMSE(i) = meanMSE(i) - halfWidth;
    ciHighMSE(i) = meanMSE(i) + halfWidth;
    meanMSEtheory(i) = MSEfloorTheory + sigma^2;
end
Tmse = table(snrList(:), meanMSE, ciLowMSE, ciHighMSE, meanMSEtheory, ...
    'VariableNames', {'SNR_dB', 'Mean_MSE_emp', 'CI95_low', 'CI95_high', 'Mean_MSE_theory'});
writetable(Tmse, fullfile(outDir, 'table_mse_summary.csv'));

fig = figure('Color', 'w', 'Position', [100 100 680 360]);
errorbar(snrList, meanMSE, meanMSE - ciLowMSE, ciHighMSE - meanMSE, 'o-', ...
    'LineWidth', 1.2, 'MarkerSize', 5); hold on;
plot(snrList, meanMSEtheory, '--', 'LineWidth', 1.3);
xlabel('SNR (dB)'); ylabel('Mean MSE (pu^2)'); box on;
exportgraphics(fig, fullfile(outDir, 'fig4_mse_vs_snr.png'), 'Resolution', 300);
close(fig);

%% One-cycle amplitude error probability
rng(444, 'twister');
tCycle = (0:cycleN-1).' / fs;
Hamp = [sin(2*pi*f0*tCycle), cos(2*pi*f0*tCycle)];
Gamp = (Hamp' * Hamp) \ Hamp';
sCycle = sin(2*pi*f0*tCycle);

PampEmp = zeros(numel(snrList),1);
PampTheory = zeros(numel(snrList),1);
ciLowAmp = zeros(numel(snrList),1);
ciHighAmp = zeros(numel(snrList),1);

for i = 1:numel(snrList)
    sigma = noise_sigma_for_snr(snrList(i));
    noise = sigma * randn(cycleN, MC_PROB);
    Y = sCycle + noise;
    coeff = Gamp * Y;
    Ahat = sqrt(coeff(1,:).^2 + coeff(2,:).^2);
    flag = abs(Ahat - 1.0) > epsAmp;
    PampEmp(i) = mean(flag);
    [ciLowAmp(i), ciHighAmp(i)] = wilson_interval(sum(flag), MC_PROB, 1.96);

    sigmaTheta = sqrt(2 * sigma^2 / cycleN);
    lambdaRice = (1.0 / sigmaTheta)^2;
    lo = ((1 - epsAmp) / sigmaTheta)^2;
    hi = ((1 + epsAmp) / sigmaTheta)^2;
    PampTheory(i) = ncx2cdf(lo, 2, lambdaRice) + 1 - ncx2cdf(hi, 2, lambdaRice);
end
Tamp = table(snrList(:), PampEmp, PampTheory, ciLowAmp, ciHighAmp, ...
    'VariableNames', {'SNR_dB', 'P_emp', 'P_theory', 'CI95_low', 'CI95_high'});
writetable(Tamp, fullfile(outDir, 'table_amp_probability.csv'));

fig = figure('Color', 'w', 'Position', [100 100 680 360]);
errorbar(snrList, PampEmp, PampEmp - ciLowAmp, ciHighAmp - PampEmp, 'o', ...
    'LineWidth', 1.1, 'MarkerSize', 5); hold on;
plot(snrList, PampTheory, '--', 'LineWidth', 1.3);
xlabel('SNR (dB)'); ylabel('P(|Ahat - A| > 1%)'); ylim([-0.02 1.02]); box on;
exportgraphics(fig, fullfile(outDir, 'fig5_amp_probability.png'), 'Resolution', 300);
close(fig);

%% Residual-energy detector probability
rng(555, 'twister');
cols = [];
for h = [1 5 7]
    cols = [cols, sin(2*pi*h*f0*tCycle), cos(2*pi*h*f0*tCycle)]; %#ok<AGROW>
end
Hdet = cols;
Pdet = Hdet / (Hdet' * Hdet) * Hdet';
Mdet = eye(cycleN) - Pdet;
nuDet = round(trace(Mdet));
startIdx = 81; % MATLAB 1-based; corresponds to sample 80 in zero-based indexing
uCycle = double((1:cycleN).' >= startIdx);
tauCycle = transAmp * exp(-transDecay * (tCycle - tCycle(startIdx))) .* sin(2*pi*transFreq*(tCycle - tCycle(startIdx))) .* uCycle;
rTau = Mdet * tauCycle;
hCycle = alpha5 * sin(2*pi*5*f0*tCycle + phi5) + alpha7 * sin(2*pi*7*f0*tCycle + phi7);
baseCycle = sCycle + hCycle;

PFAemp = zeros(numel(snrList),1);
PDemp = zeros(numel(snrList),1);
PDtheory = zeros(numel(snrList),1);
ciLowPD = zeros(numel(snrList),1);
ciHighPD = zeros(numel(snrList),1);
etaVals = zeros(numel(snrList),1);

for i = 1:numel(snrList)
    sigma = noise_sigma_for_snr(snrList(i));
    eta = (sigma^2 / cycleN) * chi2inv(1 - alphaFA, nuDet);
    etaVals(i) = eta;
    lambdaNC = (norm(rTau)^2) / sigma^2;
    PDtheory(i) = 1 - ncx2cdf((cycleN * eta) / sigma^2, nuDet, lambdaNC);

    noise = sigma * randn(cycleN, MC_PROB);
    X0 = baseCycle + noise;
    X1 = X0 + tauCycle;
    R0 = Mdet * X0;
    R1 = Mdet * X1;
    J0 = mean(R0.^2, 1);
    J1 = mean(R1.^2, 1);
    flagFA = J0 > eta;
    flagD = J1 > eta;
    PFAemp(i) = mean(flagFA);
    PDemp(i) = mean(flagD);
    [ciLowPD(i), ciHighPD(i)] = wilson_interval(sum(flagD), MC_PROB, 1.96);
end
Tdet = table(snrList(:), etaVals, PFAemp, PDemp, PDtheory, ciLowPD, ciHighPD, ...
    'VariableNames', {'SNR_dB', 'eta', 'P_FA_emp', 'P_D_emp', 'P_D_theory', 'CI95_low', 'CI95_high'});
writetable(Tdet, fullfile(outDir, 'table_detector_probability.csv'));

fig = figure('Color', 'w', 'Position', [100 100 680 360]);
errorbar(snrList, PDemp, PDemp - ciLowPD, ciHighPD - PDemp, 's', ...
    'LineWidth', 1.1, 'MarkerSize', 5); hold on;
plot(snrList, PDtheory, '--', 'LineWidth', 1.3);
xlabel('SNR (dB)'); ylabel('Detection probability P_D'); ylim([-0.02 1.02]); box on;
exportgraphics(fig, fullfile(outDir, 'fig6_detector_probability.png'), 'Resolution', 300);
close(fig);

%% Summary table and design rule
Tsummary = table(snrList(:), meanMSE, meanMSEtheory, PampEmp, PampTheory, PDemp, PDtheory, PFAemp, ...
    'VariableNames', {'SNR_dB', 'MSE_emp', 'MSE_theory', 'P_amp_emp', 'P_amp_theory', 'P_D_emp', 'P_D_theory', 'P_FA_emp'});
writetable(Tsummary, fullfile(outDir, 'table3_probability_summary.csv'));

beta = 0.10;
q = -sqrt(2) * erfcinv(beta); % same as norminv(1 - beta/2)
snrMinApprox = 10 * log10((q^2) / (cycleN * epsAmp^2));

summaryTxt = fullfile(outDir, 'results_summary.txt');
fid = fopen(summaryTxt, 'w');
fprintf(fid, 'Probability-Driven Power-System Signal Analysis\n\n');
fprintf(fid, 'Theoretical MSE floor = %.6f pu^2\n', MSEfloorTheory);
fprintf(fid, 'Approximate minimum SNR for 10%% exceedance risk = %.3f dB\n\n', snrMinApprox);
fprintf(fid, 'Summary table:\n');
for i = 1:numel(snrList)
    fprintf(fid, 'SNR=%2d dB | MSE emp=%.6f, theory=%.6f | P_amp emp=%.4f, theory=%.4f | P_D emp=%.4f, theory=%.4f | P_FA emp=%.4f\n', ...
        snrList(i), meanMSE(i), meanMSEtheory(i), PampEmp(i), PampTheory(i), PDemp(i), PDtheory(i), PFAemp(i));
end
fclose(fid);

disp('MATLAB reproduction completed successfully.');
disp(Tsummary);

%% Local functions
function sigma = noise_sigma_for_snr(snrDb)
    sigma = sqrt(0.5 / (10^(snrDb/10)));
end

function [L, R, delta] = slow_random_component(N, rhoL, stdL, rhoR, stdR, gammaR, s0)
    L = ar1_process(N, rhoL, stdL);
    R = ar1_process(N, rhoR, stdR);
    delta = (L - gammaR * R) .* s0;
end

function x = ar1_process(N, rho, stdTarget)
    sigmaW = stdTarget * sqrt(1 - rho^2);
    w = sigmaW * randn(N,1);
    x = zeros(N,1);
    for k = 2:N
        x(k) = rho * x(k-1) + w(k);
    end
    x = x - mean(x);
end

function r = one_cycle_corr(x, cycleN)
    x0 = x - mean(x);
    r = (x0(1:end-cycleN)' * x0(1+cycleN:end)) / ((numel(x0) - cycleN) * var(x0, 1));
end

function [low, high] = wilson_interval(k, n, z)
    phat = k / n;
    denom = 1 + z^2 / n;
    center = (phat + z^2 / (2*n)) / denom;
    half = z * sqrt(phat*(1-phat)/n + z^2/(4*n^2)) / denom;
    low = center - half;
    high = center + half;
end
