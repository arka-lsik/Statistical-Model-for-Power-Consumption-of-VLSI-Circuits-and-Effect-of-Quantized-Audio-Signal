clc; 
clear; 
close all;

data = readtable('power consumption of ISCAS89 benchmark circuits.csv');  

% extract numeric features and power
X = table2array(data(:, 2:10));   % Features
Y = table2array(data(:, end));   % MC simulated Power (mW)

% Remove rows with missing values
valid_rows = all(~isnan([X Y]), 2);
X = X(valid_rows, :);
Y = Y(valid_rows);

% Normalize features and power
[Xnorm, mu_X, sigma_X] = zscore(X);
Ynorm = (Y - mean(Y)) / std(Y);
XYnorm = [Xnorm Ynorm];

% Quantization of Inputs
%delta = 0.1;  
Xq = round(Xnorm / delta) * delta;
XYq = [Xq Ynorm];

% Fit GMM to Joint Data (Xq + Y)
K = 3;  
gmm = fitgmdist(XYnorm, K, 'RegularizationValue', 1e-5, ...
    'Options', statset('MaxIter', 500, 'Display', 'final'));

% Power Prediction using Conditional GMM
n = size(Xq, 1);
Y_pred = zeros(n, 1);

for i = 1:n
    x_i = Xq(i, :);
    weights = zeros(K,1);
    mu_cond = zeros(K,1);

    for k = 1:K
        mu_k = gmm.mu(k, :)';
        Sigma_k = gmm.Sigma(:,:,k);
        pi_k = gmm.ComponentProportion(k);

        mu_x = mu_k(1:9);
        mu_y = mu_k(10);
        S_xx = Sigma_k(1:9, 1:9);
        S_xy = Sigma_k(1:9, 10);
        S_yx = Sigma_k(10, 1:9);
        S_yy = Sigma_k(10, 10);

        mu_c = mu_y + S_yx / S_xx * (x_i' - mu_x);
        weights(k) = pi_k * mvnpdf(x_i, mu_x', S_xx);
        mu_cond(k) = mu_c;
    end

    weights = weights / sum(weights);
    y_pred_norm = sum(weights .* mu_cond);
    Y_pred(i) = y_pred_norm * std(Y) + mean(Y);  
end

% Output Quantization
delta_p = 1;  % Power quantization step
Y_pred_q = round(Y_pred / delta_p) * delta_p;

% Evaluation Metrics
MAE = mean(abs(Y - Y_pred));
RMSE = sqrt(mean((Y - Y_pred).^2));
R2 = 1 - sum((Y - Y_pred).^2) / sum((Y - mean(Y)).^2);

fprintf('MAE  = %.4f mW\n', MAE);
fprintf('RMSE = %.4f mW\n', RMSE);
fprintf('R²   = %.4f\n', R2);


% Visualization
figure;
scatter(Y, Y_pred, 'filled');
xlabel('True Power (mW)');
ylabel('Predicted Power (mW)');
title('GMM Power Prediction'); grid on; refline(1,0);


figure;
histogram(Y - Y_pred);
xlabel('Prediction Error (mW)');
ylabel('Frequency');
title('Prediction Error Distribution'); grid on;

figure;
Ks = 1:6;
AIC = zeros(1,length(Ks));
BIC = zeros(1,length(Ks));
for i = Ks
    g = fitgmdist(XYq, i, 'RegularizationValue', 1e-5);
    AIC(i) = g.AIC; BIC(i) = g.BIC;
end
plot(Ks, AIC, '-o'); hold on;
plot(Ks, BIC, '-x');
legend('AIC', 'BIC');
xlabel('K'); ylabel('Score');
title('Model Selection with AIC/BIC'); grid on;

figure;
scatter(1:n, Y_pred_q, 'filled');
xlabel('Sample Index');
ylabel('Quantized Predicted Power');
title('Quantized Output Power'); grid on;

%model comparison
%% LINEAR REGRESSION 
% Train/Test split
n = size(X,1);
rng(1); 
idx = randperm(n);
n_train = round(0.7 * n);
train_idx = idx(1:n_train);
test_idx  = idx(n_train+1:end);

X_train = X(train_idx, :);
Y_train = Y(train_idx);
X_test  = X(test_idx, :);
Y_test  = Y(test_idx);
mdl_lr = fitlm(X_train, Y_train);
Y_pred_lr = predict(mdl_lr, X_test);

MAE_lr = mean(abs(Y_test - Y_pred_lr));
RMSE_lr = sqrt(mean((Y_test - Y_pred_lr).^2));
R2_lr = 1 - sum((Y_test - Y_pred_lr).^2) / sum((Y_test - mean(Y_test)).^2);

fprintf('\n[Linear Regression]\n');
fprintf('MAE  = %.4f mW\n', MAE_lr);
fprintf('RMSE = %.4f mW\n', RMSE_lr);
fprintf('R²   = %.4f\n', R2_lr);


%% RANDOM FOREST 
nTrees = 100;
rf_model = TreeBagger(nTrees, X_train, Y_train, 'Method', 'regression');
Y_pred_rf = predict(rf_model, X_test);

MAE_rf = mean(abs(Y_test - Y_pred_rf));
RMSE_rf = sqrt(mean((Y_test - Y_pred_rf).^2));
R2_rf = 1 - sum((Y_test - Y_pred_rf).^2) / sum((Y_test - mean(Y_test)).^2);

fprintf('\n[Random Forest]\n');
fprintf('MAE  = %.4f mW\n', MAE_rf);
fprintf('RMSE = %.4f mW\n', RMSE_rf);
fprintf('R²   = %.4f\n', R2_rf);


%% NEURAL NETWORK (shallow)
mu_Y = mean(Y_train);
sigma_Y = std(Y_train);

X_nn_train = (X_train - mu_X) ./ sigma_X;
X_nn_test  = (X_test - mu_X) ./ sigma_X;
Y_nn_train = (Y_train - mu_Y) / sigma_Y;

net = fitrnet(X_nn_train, Y_nn_train, 'Standardize', false); 
Y_pred_nn_norm = predict(net, X_nn_test);
Y_pred_nn = Y_pred_nn_norm * sigma_Y + mu_Y;

MAE_nn = mean(abs(Y_test - Y_pred_nn));
RMSE_nn = sqrt(mean((Y_test - Y_pred_nn).^2));
R2_nn = 1 - sum((Y_test - Y_pred_nn).^2) / sum((Y_test - mean(Y_test)).^2);

fprintf('\n[Neural Network]\n');
fprintf('MAE  = %.4f mW\n', MAE_nn);
fprintf('RMSE = %.4f mW\n', RMSE_nn);
fprintf('R²   = %.4f\n', R2_nn);

%visualize the results
models = {'Linear', 'RF', 'NN','GMM'};
mae_all = [MAE, MAE_lr, MAE_rf, MAE_nn];
rmse_all = [RMSE, RMSE_lr, RMSE_rf, RMSE_nn];
r2_all = [R2, R2_lr, R2_rf, R2_nn];

figure;
 bar(mae_all); title('MAE'); set(gca, 'xticklabel', models); ylabel('mW'); 
 figure;
 bar(rmse_all); title('RMSE'); set(gca, 'xticklabel', models); ylabel('mW'); grid on;
 figure;
 bar(r2_all); title('R^2'); set(gca, 'xticklabel', models); ylim([0 1.1]); grid on;

% Get cluster assignments for each point
cluster_indices = cluster(gmm, XYq);

 figure;
gscatter(Y, Y_pred, cluster_indices);
xlabel('True Power (mW)');
ylabel('Predicted Power (mW)');
title('GMM Clustering and Prediction Results');
legend('Cluster 1', 'Cluster 2', 'Cluster 3', 'Location', 'best');
grid on;

% Add a reference line to show perfect predictions
hold on; % Keep the current plot
refline(1,0); % Add a y=x line
hold off;

