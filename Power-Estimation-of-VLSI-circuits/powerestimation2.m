clc;
clear;
close all;

% Load the ARM-A15 dataset
data = readtable('traindata.csv');

% Extract features (X) and the target (Y)
X = table2array(data(:, {'Frequency', 'Voltage', 'Core_Count', 'EPH_0x11_Frequency_A15_Voltage_A15_Squared_Percentage', 'EPH_0x1b_Frequency_A15_Voltage_A15_Squared_Percentage', 'EPH_0x50_Frequency_A15_Voltage_A15_Squared_Percentage', 'EPH_0x14_Frequency_A15_Voltage_A15_Squared_Percentage', 'EPH_0x19_Frequency_A15_Voltage_A15_Squared_Percentage'}));
Y = table2array(data(:, 'Actual')); % Target: Actual Power (W)

% Get feature and data dimensions
[num_samples, num_features] = size(X);

rng(1); % for reproducibility
cv = cvpartition(num_samples, 'HoldOut', 0.3);
idx_train = training(cv);
idx_test = test(cv);

X_train = X(idx_train, :);
Y_train = Y(idx_train);
X_test = X(idx_test, :);
Y_test = Y(idx_test);

fprintf('Data loaded. Training set: %d samples, Test set: %d samples.\n\n', size(X_train, 1), size(X_test, 1));


%% GMM-BASED POWER PREDICTION
fprintf('Performing GMM-based Power Prediction...\n');

% Normalize Training Data
[X_train_norm, mu_X, sigma_X] = zscore(X_train);
[Y_train_norm, mu_Y, sigma_Y] = zscore(Y_train);
XY_train_norm = [X_train_norm, Y_train_norm];

% Find Optimal Number of Components (K) using BIC
fprintf('Finding optimal K using BIC... ');
Ks = 1:8;
bic_scores = zeros(1, length(Ks));
options = statset('MaxIter', 500);
for i = 1:length(Ks)
    gmm_temp = fitgmdist(XY_train_norm, Ks(i), 'RegularizationValue', 1e-5, 'Options', options);
    bic_scores(i) = gmm_temp.BIC;
end
[min_bic, best_K] = min(bic_scores);
fprintf('Optimal K = %d.\n', best_K);

figure;
plot(Ks, bic_scores, '-o', 'LineWidth', 2);
xlabel('Number of Components (K)');
ylabel('Bayesian Information Criterion (BIC)');
title('GMM Model Selection');
grid on;

% Fit Final GMM to Joint Training Data
fprintf('   Fitting final GMM with K = %d...\n', best_K);
gmm_final = fitgmdist(XY_train_norm, best_K, 'RegularizationValue', 1e-5, 'Options', options);

%Predict on Test Set using Conditional GMM
fprintf('Predicting power on the test set...\n');
X_test_norm = (X_test - mu_X) ./ sigma_X;
num_test_samples = size(X_test_norm, 1);
Y_pred_gmm_norm = zeros(num_test_samples, 1);
for i = 1:num_test_samples
    x_i = X_test_norm(i, :);
    weights = zeros(best_K, 1);
    mu_cond = zeros(best_K, 1);
    for k = 1:best_K
        mu_k = gmm_final.mu(k, :)';
        Sigma_k = gmm_final.Sigma(:, :, k);
        pi_k = gmm_final.ComponentProportion(k);
        mu_x = mu_k(1:num_features);
        mu_y = mu_k(num_features+1);
        S_xx = Sigma_k(1:num_features, 1:num_features);
        S_xy = Sigma_k(1:num_features, num_features+1);
        S_yx = Sigma_k(num_features+1, 1:num_features);
        S_yy = Sigma_k(num_features+1, num_features+1);
        mu_c = mu_y + S_yx * (S_xx \ (x_i' - mu_x));
        weights(k) = pi_k * mvnpdf(x_i, mu_x', S_xx);
        mu_cond(k) = mu_c;
    end
    weights = weights / sum(weights);
    Y_pred_gmm_norm(i) = sum(weights .* mu_cond);
end

% De-normalize the prediction to get the actual power value
Y_pred_gmm = Y_pred_gmm_norm * sigma_Y + mu_Y;

% Evaluate GMM
MAE_gmm = mean(abs(Y_test - Y_pred_gmm));
RMSE_gmm = sqrt(mean((Y_test - Y_pred_gmm).^2));
R2_gmm = 1 - sum((Y_test - Y_pred_gmm).^2) / sum((Y_test - mean(Y_test)).^2);
fprintf('\n[GMM Results]\n');
fprintf('   MAE  = %.4f W\n', MAE_gmm);
fprintf('   RMSE = %.4f W\n', RMSE_gmm);
fprintf('   R2   = %.4f\n\n', R2_gmm);

%% VISUALIZATION OF RESULTS
fprintf('Generating result visualizations...\n');

% GMM: True vs. Predicted Power 
% Create the 9-dimensional matrix for clustering
XY_test_for_clustering = [X_test_norm, Y_pred_gmm_norm];
cluster_indices = cluster(gmm_final, XY_test_for_clustering);

figure;
gscatter(Y_test, Y_pred_gmm, cluster_indices);
hold on;
refline(1, 0); % Add y=x line for reference
hold off;
grid on;
xlabel('True Power (W)');
ylabel('Predicted Power (W)');
title(sprintf('GMM Power Prediction (K=%d)', best_K));
legend('Location', 'best');

% GMM Prediction Error Distribution
figure;
histogram(Y_test - Y_pred_gmm);
xlabel('Prediction Error (W)');
ylabel('Frequency');
title('GMM Prediction Error Distribution');
grid on;

fprintf('   Done. All tasks complete.\n');
