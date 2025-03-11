% Sampling rate
sampling_rate = 200;

% Load the CSV file
file_name = 'S001_G01_D01_B01_T01.csv';
dat = readtable(file_name);

% % Display first few rows of the data
% disp(dat(1:5, :));
% 
% % Find foot contact and toe-off events
% right_contact_locs = findpeaks(dat{:, 244}, 'MinPeakDistance', 50, 'MinPeakHeight', 0.5, 'MinPeakWidth', 70);
% right_toeoff_locs = findpeaks(-dat{:, 244}, 'MinPeakWidth', 50);
% left_contact_locs = findpeaks(dat{:, 153}, 'MinPeakDistance', 50, 'MinPeakHeight', 0.5, 'MinPeakWidth', 70);
% left_toeoff_locs = findpeaks(-dat{:, 153}, 'MinPeakWidth', 50);
% 
% if length(left_contact_locs) <= 10 || length(right_contact_locs) <= 10 || abs(length(left_contact_locs) - length(right_contact_locs)) > 20
%     disp("Datos inválidos para análisis espaciotemporal.");
% else
%     % Trajectories
%     leftXTraj = dat{:, 317};
%     leftYTraj = dat{:, 318};
%     rightXTraj = dat{:, 319};
%     rightYTraj = dat{:, 320};
% 
%     right_stride_length = [];
%     left_step_width = [];
%     left_step_length = [];
%     left_stride_length = [];
%     right_step_width = [];
%     right_step_length = [];
% 
%     % Left leg calculations
%     for i = 1:length(right_contact_locs) - 1
%         R1x = rightXTraj(right_contact_locs(i));
%         R1y = rightYTraj(right_contact_locs(i));
%         R2x = rightXTraj(right_contact_locs(i + 1));
%         R2y = rightYTraj(right_contact_locs(i + 1));
%         L1x = leftXTraj(left_contact_locs(i));
%         L1y = leftYTraj(left_contact_locs(i));
% 
%         right_stride = sqrt((R2x - R1x)^2 + (R2y - R1y)^2) / 10;
%         right_stride_length(end + 1) = right_stride;
% 
%         R1L1 = sqrt((L1x - R1x)^2 + (L1y - R1y)^2) / 10;
%         R2L1 = sqrt((L1x - R2x)^2 + (L1y - R2y)^2) / 10;
%         semi_perimeter = (R1L1 + R2L1 + right_stride) / 2;
%         width = (2 * sqrt(semi_perimeter * (semi_perimeter - R1L1) * ...
%                  (semi_perimeter - R2L1) * (semi_perimeter - right_stride))) / R1L1;
%         left_step_width(end + 1) = width;
%         left_step_length(end + 1) = sqrt(R1L1^2 - width^2);
%     end
% 
%     % Right leg calculations
%     for i = 1:length(left_contact_locs) - 1
%         L1x = leftXTraj(left_contact_locs(i));
%         L1y = leftYTraj(left_contact_locs(i));
%         L2x = leftXTraj(left_contact_locs(i + 1));
%         L2y = leftYTraj(left_contact_locs(i + 1));
%         R1x = rightXTraj(right_contact_locs(i));
%         R1y = rightYTraj(right_contact_locs(i));
% 
%         left_stride = sqrt((L2x - L1x)^2 + (L2y - L1y)^2) / 10;
%         left_stride_length(end + 1) = left_stride;
% 
%         L1R1 = sqrt((R1x - L1x)^2 + (R1y - L1y)^2) / 10;
%         L2R1 = sqrt((R1x - L2x)^2 + (R1y - L2y)^2) / 10;
%         semi_perimeter = (L1R1 + L2R1 + left_stride) / 2;
%         width = (2 * sqrt(semi_perimeter * (semi_perimeter - L1R1) * ...
%                  (semi_perimeter - L2R1) * (semi_perimeter - left_stride))) / L1R1;
%         right_step_width(end + 1) = width;
%         right_step_length(end + 1) = sqrt(L1R1^2 - width^2);
%     end
% 
%     % Pad lists to the same length
%     max_length = max([length(right_stride_length), length(left_step_width), ...
%                       length(left_step_length), length(left_stride_length), ...
%                       length(right_step_width), length(right_step_length)]);
% 
%     right_stride_length = [right_stride_length, nan(1, max_length - length(right_stride_length))];
%     left_step_width = [left_step_width, nan(1, max_length - length(left_step_width))];
%     left_step_length = [left_step_length, nan(1, max_length - length(left_step_length))];
%     left_stride_length = [left_stride_length, nan(1, max_length - length(left_stride_length))];
%     right_step_width = [right_step_width, nan(1, max_length - length(right_step_width))];
%     right_step_length = [right_step_length, nan(1, max_length - length(right_step_length))];
% 
%     % Additional Calculations: Cadence, stride time
%     step_locs = sort([right_contact_locs; left_contact_locs]);
%     cadence = length(step_locs) / (4 / 60);  % Steps per minute (4 minutes walking)
% 
%     step_time = diff(step_locs) / sampling_rate;
%     left_stride_time = diff(left_contact_locs) / sampling_rate;
%     right_stride_time = diff(right_contact_locs) / sampling_rate;
% 
%     left_stance_time = [];
%     for j = 1:length(left_contact_locs) - 1
%         if length(left_contact_locs) == length(left_toeoff_locs)
%             left_stance_time(end + 1) = abs((left_toeoff_locs(j + 1) - left_contact_locs(j)) / sampling_rate);
%         end
%     end
% 
%     % Export results to Excel
%     spatiotemporal_variables = table(left_step_width', right_step_width', ...
%         left_step_length', right_step_length', right_stride_length', ...
%         left_stride_length', left_stride_time', right_stride_time', ...
%         [cadence; nan(max_length - 1, 1)], 'VariableNames', ...
%         {'left_step_width', 'right_step_width', 'left_step_length', ...
%         'right_step_length', 'right_stride_length', 'left_stride_length', ...
%         'left_stride_time', 'right_stride_time', 'cadence'});
% 
%     writetable(spatiotemporal_variables, 'spatiotemporal_variables.xlsx');
%     disp('Variables espaciotemporales exportadas con éxito.');
% end

% --- Nonlinear Analysis ---

% Average Mutual Information (AMI)
addpath('C:\Users\diana\OneDrive - UNIVERSIDAD INDUSTRIAL DE SANTANDER\2024_II\Tesis\Pruebas algoritmo\libreriasNONAn\matlab');
data1 = dat.KneeFlexionLT_deg_;
L = 50; % Lag to analyze
[tau, ami] = AMI_Stergiou(data1, L); % Replace with your AMI calculation function
fprintf('Primer mínimo en AMI (tau): %d\n', tau);

% False Nearest Neighbors (FNN)
tau=5;
MaxDim = 30;
Rtol = 15;
Atol = 2;
speed = 0;
[dim, dE] = FNN(data1, tau, MaxDim, Rtol, Atol, speed); % Replace with your FNN function
fprintf('Embedding Dimension (dE): %d\n', dE);

% Phase Space Reconstruction
numpoints = length(data1);
timelag = tau;
embeddingdimension = dim;
numsamples = numpoints - (embeddingdimension - 1) * timelag;
stateSpace = zeros(embeddingdimension, numsamples);

for i = 1:embeddingdimension
    stateSpace(i, :) = data1((1:numsamples) + (i - 1) * timelag);
end

% Plot Phase Space Reconstruction
figure;
plot3(stateSpace(1, :), stateSpace(2, :), stateSpace(3, :), 'r', 'LineWidth', 1.5);
title('Reconstructed State Space (3D)');
xlabel('x(t)'); ylabel('x(t - \tau)'); zlabel('x(t - 2*\tau)');
grid on;

% Lyapunov Exponent (LyE)
[out, LyE] = LyE_W(data1, sampling_rate, tau, dim, 50 * sampling_rate); % Replace with your LyE function
fprintf('Lyapunov Exponent (LyE): %.4f\n', LyE);

% Correlation Dimension (CoD)
CoD = corrdim(data1, tau, dim, 0); % Replace with your CoD function
fprintf('Correlation Dimension (CoD): %.4f\n', CoD);

% Sample entropy 
R=0.20;%20% Standard deviation 
%m=2*dim; %vector length
m=5; 
SE = Ent_Samp(data1,m,R);
%0.0535 %0.1057

%8. Approximate entropy ApEn
r=0.2; 
AE = Ent_Ap(data1, dim, r ); %0.1153