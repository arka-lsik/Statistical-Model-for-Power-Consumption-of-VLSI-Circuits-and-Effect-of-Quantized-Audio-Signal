% Synthetic data generation creates new, artificial datasets that statistically mirror real-world data (my case I taken this data from one of Lab mate who
% is crrently working on this type of similar data recorded audio form 10 different places)
% without copying specific observations, making it similar to real data in its patterns and structure but not identical

clc;
clear;

N = 10000; 

mu1_1 = 0; sigma1_1 = 1; alpha1_1 = 0.5;
mu2_1 = 3; sigma2_1 = 1; alpha2_1 = 0.5;

N1 = round(alpha1_1 * N);
N2 = N - N1;

% Generate data and labels
X1_1 = mu1_1 + sigma1_1 * randn(N1, 1);   
X2_1 = mu2_1 + sigma2_1 * randn(N2, 1);   
data1 = [X1_1; X2_1];
labels1 = [ones(N1, 1); 2*ones(N2, 1)];

% Shuffle both together
perm1 = randperm(length(data1));
data1 = data1(perm1);
labels1 = labels1(perm1);
y1 = 0.1 * randn(length(data1), 1);  % vertical jitter


mu1_2 = -2; sigma1_2 = sqrt(2); alpha1_2 = 0.3;
mu2_2 = 3; sigma2_2 = 1; alpha2_2 = 0.7;

N1 = round(alpha1_2 * N);
N2 = N - N1;

% Generate data and labels
X1_2 = mu1_2 + sigma1_2 * randn(N1, 1);   
X2_2 = mu2_2 + sigma2_2 * randn(N2, 1);   
data2 = [X1_2; X2_2];
labels2 = [ones(N1, 1); 2*ones(N2, 1)];

% Shuffle both together
perm2 = randperm(length(data2));
data2 = data2(perm2);
labels2 = labels2(perm2);
y2 = 0.1 * randn(length(data2), 1);  % vertical jitter


figure;

subplot(1,2,1);
gscatter(data1, y1, labels1, 'rb', 'ox');
title('GMM Dataset 1');
xlabel('x'); ylabel('~');
legend('Component 1', 'Component 2');

subplot(1,2,2);
gscatter(data2, y2, labels2, 'mg', 'ox');
title('GMM Dataset 2');
xlabel('x'); ylabel('~');
legend('Component 1', 'Component 2');
