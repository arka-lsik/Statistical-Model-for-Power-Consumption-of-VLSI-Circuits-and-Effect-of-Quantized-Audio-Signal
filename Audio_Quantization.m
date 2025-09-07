% Code is for mainly 1. fit the data as GMM and clustering
%                    2. Used Quantization(Uniform and ECQ(entropy coded quantization) effect of GMM modeled data after qunatization with specific levels.
%                    3. After effects of quantizaation to reconsturuct the original one.
%                    4. Applied EM Algorithm to better estimate, comapre the noise/error results.

% Parameters for GMM - Dataset 1
mu1_1 = 0; sigma1_1 = 1; alpha1_1 = 0.5;
mu2_1 = 3; sigma2_1 = 1; alpha2_1 = 0.5;

% Parameters for GMM - Dataset 2
mu1_2 = -2; sigma1_2 = sqrt(2); alpha1_2 = 0.3;
mu2_2 = 3; sigma2_2 = 1; alpha2_2 = 0.7;

% Quantization and EM parameters
eta_values = [1, 0.7];  % Scaling factors for quantization
L_values = [4, 8, 16];  % Quantization levels
num_samples = 10000;

% Generate data for both datasets
data1 = [alpha1_1 * randn(num_samples, 1) * sigma1_1 + mu1_1; ...
         alpha2_1 * randn(num_samples, 1) * sigma2_1 + mu2_1];
data2 = [alpha1_2 * randn(num_samples, 1) * sigma1_2 + mu1_2; ...
         alpha2_2 * randn(num_samples, 1) * sigma2_2 + mu2_2];


% Fit GMM to each dataset and perform clustering
datasets = {data1, data2};
titles = {'Dataset 1', 'Dataset 2'};

for i = 1:2
    % Select dataset
    data = datasets{i};
    
    % Fit a 2-component GMM to the dataset
    options = statset('MaxIter', 1000);
    GMModel = fitgmdist(data, 2, 'Options', options);
    
    % Assign clusters based on the GMM model
    cluster_assignments = cluster(GMModel, data);
    
    % Plot the data with clusters
    figure;
    scatter(data, zeros(size(data)), 10, cluster_assignments, 'filled');
    xlabel('Data');
    ylabel('Cluster Assignment');
    title(['GMM Clustering of ', titles{i}]);
    colorbar;
end


% Uniform scalar quantization function
quantize = @(x, eta, L, xmin, xmax) eta * xmin + floor((x - eta * xmin) / ...
           ((eta * (xmax - xmin)) / L) + 0.5) * ((eta * (xmax - xmin)) / L);


% Loop over each dataset to create separate figures
for dataset_idx = 1:2
    % Select the dataset
    data = eval(['data', num2str(dataset_idx)]);
    
    % Create a new figure for each dataset
    figure;
    tiledlayout(2, 1);
    
    for eta_idx = 1:length(eta_values)
        eta = eta_values(eta_idx);
        xmin = min(data); xmax = max(data);

        % Select subplot for current eta value
        nexttile;

        % Plot original data PDF
[f, xi] = ksdensity(data);
plot(xi, f, 'k-', 'LineWidth', 2.5, 'DisplayName', 'Without quantization'); 
hold on;


color_array = [
    0.1, 0.1, 0.44; % Dark Blue
    0.13, 0.55, 0.13; % Dark Green
    0.55, 0, 0; % Maroon
    0.25, 0.25, 0.25; % Dark Gray
    0.54, 0.17, 0.89; % Dark Violet
];


plot(xi, f, 'Color', color_array(1, :), 'LineWidth', 2.5, 'DisplayName', 'Dark Blue');

        % Loop over quantization levels
        for k = 1:length(L_values)
            L = L_values(k);

            % Quantize data
            quantized_data = quantize(data, eta, L, xmin, xmax);
            % Calculate distortion between original data and quantized data (without EM)
            distortion_without_EM = mean((data - quantized_data).^2);
            distortions_without_EM(eta_idx, k) = distortion_without_EM;

            % Apply EM algorithm on quantized data to estimate GMM
            options = statset('MaxIter', 1000);
            GMModel = fitgmdist(quantized_data, 2, 'Options', options);
            gamma = mean(GMModel.ComponentProportion);

            % Generate data using GMM model to estimate the clustered data (fitted data)
            clustered_data = pdf(GMModel, quantized_data);

         

            % Calculate distortion between original data and EM clustered data (with EM)
            distortion_with_EM = mean((cluster_assignments - clustered_data).^2);
            distortions_with_EM(eta_idx, k) = distortion_with_EM;

            % Calculate PDF after EM clustering
            [f_quantized, xi_quantized] = ksdensity(quantized_data);
            plot(xi_quantized, f_quantized, '--','Color', color_array(k), 'LineWidth', 1.5, ...
                 'DisplayName', sprintf('L = %d', L));

            % Display distortion results
            fprintf('Dataset %d, eta = %.1f, L = %d\n', dataset_idx, eta, L);
            fprintf('Distortion (with EM): %.4f\n', distortion_without_EM);
            fprintf('Distortion (without EM): %.4f\n', distortion_with_EM);
        end

        % Label plot
        xlabel('x', 'FontWeight', 'bold');
        ylabel('p(x)', 'FontWeight', 'bold');
        legend('show');
        title(['\eta = ', num2str(eta)]);
        hold off;
    end
    
    % Set title for each dataset figure
    sgtitle(['PDFs of GMM Estimated for Various L and \eta on Dataset ', num2str(dataset_idx)]);

end



