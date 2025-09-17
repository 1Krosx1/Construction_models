% =========================================================================
% main_script.m (REVISED & UPGRADED)
%
% Main script to load, process, train, and evaluate the ANN model for
% architectural data. This version generates and saves all analytical plots.
% =========================================================================

% =========================================================================
% Section 1: Setup
% =========================================================================
clear; 
clc; 
close all;

set(groot, 'defaultfigurerenderer', 'painters');
fprintf("Workspace cleared and ready.\n");


% =========================================================================
% Section 2: Data Loading, Cleaning, and Merging
% =========================================================================
fprintf('\n--- Section 2: Loading, Cleaning & Merging Data ---\n');

opts_qty = detectImportOptions('Thesis Data - Architectural Quantity Cost.csv', 'VariableNamingRule', 'preserve');
opts_cost = detectImportOptions('Thesis Data - Achitectural Unit Cost.csv', 'VariableNamingRule', 'preserve');

try
    T_quantity = readtable('Thesis Data - Architectural Quantity Cost.csv', opts_qty);
    fprintf('Architectural Quantity data loaded.\n');
    T_unit_cost = readtable('Thesis Data - Achitectural Unit Cost.csv', opts_cost);
    fprintf('Architectural Unit Cost data loaded.\n');
catch ME
    fprintf('Error: %s\nMake sure both CSV files are in the correct directory.\n', ME.message);
    return;
end

T_quantity_cleaned = clean_table(T_quantity);
T_unit_cost_cleaned = clean_table(T_unit_cost);

budgets = rowfun(@extract_budget, T_quantity_cleaned(:, 'Year/Budget'), 'OutputFormat', 'uniform');
T_quantity_cleaned.Budget = budgets;
T_quantity_cleaned.('Year/Budget') = [];
T_unit_cost_cleaned.('Year/Budget') = [];

T_merged = innerjoin(T_quantity_cleaned, T_unit_cost_cleaned, 'Keys', 'Join_Key');

T_merged = T_merged(~isnan(T_merged.Budget), :);
T_merged = T_merged(T_merged.Budget > 100000, :);
fprintf('Tables merged successfully. Working with %d common projects.\n', height(T_merged));


% =========================================================================
% Section 3: Granular Feature Engineering & Visualization
% =========================================================================
fprintf('\n--- Section 3: Engineering Granular Features & Analysis ---\n');

% --- Step 1: Create Granular Cost Features ---
individual_cost_features = {};
base_feature_cols = {
    'Quantity of plaster (sq.m.)', 'Quantity of glazed tiles (sq.m.)', ...
    'Painting masonry (sq.m.)', 'painting wood (sq.m.)', ...
    'painting metal (sq.m.)', 'Area of CHB 100mm (sq.m.)', ...
    'Area of CHB 150mm (sq.m.)'
};

for i = 1:numel(base_feature_cols)
    col = base_feature_cols{i};
    qty_col = [col, '_T_quantity_cleaned'];
    cost_col = [col, '_T_unit_cost_cleaned'];
    
    new_cost_feature = regexprep(col, {' \(sq\.m\.\)', 'Quantity of ', 'Area of ', ' '}, {'', '', '', '_'});
    new_cost_feature = [new_cost_feature, '_Est_Cost'];

    if ismember(qty_col, T_merged.Properties.VariableNames) && ismember(cost_col, T_merged.Properties.VariableNames)
        T_merged.(new_cost_feature) = T_merged.(qty_col) .* T_merged.(cost_col);
    else
        T_merged.(new_cost_feature) = zeros(height(T_merged),1);
        fprintf('Warning: missing columns for %s; filled with zeros.\n', new_cost_feature);
    end
    individual_cost_features{end+1} = new_cost_feature;
end
fprintf('Created new granular cost features.\n');

% --- Step 2: Create Contextual and Target Features ---
project_description_col = 'Project_Name_T_quantity_cleaned';
storeys_cell = regexp(T_merged.(project_description_col), '(\d+)\s*sty', 'tokens', 'once');
num_rows = height(T_merged);
num_storeys = NaN(num_rows, 1);
for i = 1:num_rows
    if ~isempty(storeys_cell{i})
        num_storeys(i) = str2double(storeys_cell{i}{1});
    end
end
T_merged.Num_Storeys = num_storeys;
T_merged.Num_Storeys(isnan(T_merged.Num_Storeys)) = median(T_merged.Num_Storeys, 'omitnan');

classrooms_cell = regexp(T_merged.(project_description_col), '(\d+)\s*cl', 'tokens', 'once');
num_classrooms = NaN(num_rows, 1);
for i = 1:num_rows
    if ~isempty(classrooms_cell{i})
        num_classrooms(i) = str2double(classrooms_cell{i}{1});
    end
end
T_merged.Num_Classrooms = num_classrooms;
T_merged.Num_Classrooms(isnan(T_merged.Num_Classrooms)) = median(T_merged.Num_Classrooms, 'omitnan');
T_merged.Budget_log = log1p(T_merged.Budget);

% --- Step 3: Visualization and Analysis ---
fprintf('\n--- Generating and Saving Visualizations for Analysis ---\n');

% Create the directory for images if it doesn't exist
output_dir = 'visualization images';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% 3.1: Justifying Log-Transformation
fig1 = figure('Visible', 'off', 'Position', [100, 100, 1400, 500]);
subplot(1, 2, 1);
histogram(T_merged.Budget, 'NumBins', 30);
title('Distribution of Original Budget (Right-Skewed)');
xlabel('Budget (PHP)');
ylabel('Count');
ax = gca; ax.XAxis.Exponent = 0; % Prevent scientific notation

subplot(1, 2, 2);
histogram(T_merged.Budget_log, 'NumBins', 30, 'FaceColor', 'g');
title('Distribution of Log-Transformed Budget (Normalized)');
xlabel('Log(1 + Budget)');
ylabel('Count');
sgtitle('Effect of Log-Transformation on Target Variable', 'FontSize', 16); % Main title
exportgraphics(fig1, fullfile(output_dir, 'architectural_log_transformation_effect.png'), 'Resolution', 300);
close(fig1);
fprintf('Saved: %s\n', fullfile(output_dir, 'architectural_log_transformation_effect.png'));

% 3.2: Correlation Matrix
fig2 = figure('Visible', 'off');
heatmap_cols = [individual_cost_features, 'Num_Storeys', 'Num_Classrooms', 'Budget'];
correlation_matrix = corrcoef(T_merged{:, heatmap_cols}, 'Rows', 'pairwise');
h = heatmap(heatmap_cols, heatmap_cols, correlation_matrix, 'Colormap', parula);
h.Title = 'Correlation Matrix of Granular Architectural Features';
exportgraphics(fig2, fullfile(output_dir, 'architectural_features_correlation_matrix.png'), 'Resolution', 300);
close(fig2);
fprintf('Saved: %s\n', fullfile(output_dir, 'architectural_features_correlation_matrix.png'));

% 3.3: Justifying Standardization
final_feature_columns = [individual_cost_features, 'Num_Storeys', 'Num_Classrooms'];
X_for_viz = T_merged{:, final_feature_columns};
X_for_viz(any(isnan(X_for_viz), 2), :) = []; % Remove rows with NaNs for plotting

fig3 = figure('Visible', 'off', 'Position', [100, 100, 1000, 800]);
boxplot(X_for_viz, 'Labels', final_feature_columns, 'Orientation', 'horizontal');
set(gca, 'XScale', 'log'); % Use log scale to handle wide range
title('Architectural Feature Scales Before Standardization');
xlabel('Original Feature Values (Varying Scales)');
grid on;
exportgraphics(fig3, fullfile(output_dir, 'architectural_feature_scales_before_standardization.png'), 'Resolution', 300);
close(fig3);
fprintf('Saved: %s\n', fullfile(output_dir, 'architectural_feature_scales_before_standardization.png'));

% =========================================================================
% Section 4: Data Preparation for the ANN Model
% =========================================================================
fprintf('\n--- Section 4: Preparing Data for the ANN Model ---\n');

X = T_merged{:, final_feature_columns};
y = T_merged.Budget_log;

fprintf('Training model with %d input features.\n', size(X, 2));

rng(42);
num_data = size(X, 1);
shuffled_indices = randperm(num_data);
split_point = floor(0.8 * num_data);
idxTrain = shuffled_indices(1:split_point);
idxTest = shuffled_indices(split_point+1:end);

X_train = X(idxTrain, :);
y_train = y(idxTrain, :);
X_test = X(idxTest, :);
y_test = y(idxTest, :);

[X_train_scaled, scaler_X_mu, scaler_X_sigma] = zscore(X_train);
zeroSigmaIdx = scaler_X_sigma == 0;
if any(zeroSigmaIdx)
    scaler_X_sigma(zeroSigmaIdx) = 1; 
    warning('Some features had zero standard deviation; sigma set to 1.');
end
X_test_scaled = (X_test - scaler_X_mu) ./ scaler_X_sigma;

scaler_y_min = min(y_train);
scaler_y_max = max(y_train);
y_range = scaler_y_max - scaler_y_min;
if y_range == 0
    y_train_scaled = zeros(size(y_train));
    warning('All training targets are identical.');
else
    y_train_scaled = (y_train - scaler_y_min) / y_range;
end
y_train_scaled = y_train_scaled(:);

fprintf('Data has been split and scaled.\n');


% =========================================================================
% Section 5: Build and Train the Artificial Neural Network
% =========================================================================
fprintf('\n--- Section 5: Building and Training the ANN ---\n');

input_size = size(X_train_scaled, 2);

layers = [
    featureInputLayer(input_size, 'Name', 'input')
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu1')
    dropoutLayer(0.3, 'Name', 'dropout1')
    fullyConnectedLayer(64, 'Name', 'fc2')
    reluLayer('Name', 'relu2')
    dropoutLayer(0.2, 'Name', 'dropout2')
    fullyConnectedLayer(32, 'Name', 'fc3')
    reluLayer('Name', 'relu3')
    fullyConnectedLayer(1, 'Name', 'output')
    regressionLayer('Name', 'regression')
];

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 200, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'none', ...
    'Verbose', false);

X_train_scaled = single(X_train_scaled);
X_test_scaled  = single(X_test_scaled);
y_train_scaled = single(y_train_scaled);

[net, trainInfo] = trainNetwork(X_train_scaled, y_train_scaled, layers, options);
fprintf('Neural Network training complete.\n');

% --- Plot and Save Training Loss Curve ---
fig_train = figure('Visible', 'off');
plot(trainInfo.TrainingLoss);
title('Architectural Model Training Loss Over Epochs');
ylabel('Mean Squared Error Loss');
xlabel('Iteration');
grid on;
exportgraphics(fig_train, fullfile(output_dir, 'architectural_model_training_loss_curve.png'), 'Resolution', 300);
close(fig_train);
fprintf('Saved: %s\n', fullfile(output_dir, 'architectural_model_training_loss_curve.png'));

% =========================================================================
% Section 6: Model Evaluation
% =========================================================================
fprintf('\n--- Section 6: Evaluating the ANN Model ---\n');

scaled_predictions = predict(net, X_test_scaled);

if y_range == 0
    log_predictions = scaled_predictions * 0 + scaler_y_min;
else
    log_predictions = scaled_predictions .* y_range + scaler_y_min;
end

final_predictions = expm1(log_predictions);
y_test_actual = expm1(y_test);

r2_ann = 1 - sum((y_test_actual - final_predictions).^2) / sum((y_test_actual - mean(y_test_actual)).^2);
mae_ann = mean(abs(y_test_actual - final_predictions));
rmse_ann = sqrt(mean((y_test_actual - final_predictions).^2));

fprintf('\n--- Final Model Performance ---\n');
fprintf('R-squared (R²): %.4f\n', r2_ann);
fprintf('Mean Absolute Error (MAE): ₱%,.2f\n', mae_ann);
fprintf('Root Mean Squared Error (RMSE): ₱%,.2f\n', rmse_ann);

% --- Visualization: Actual vs. Predicted Budget ---
fig_results = figure('Visible', 'off');
scatter(y_test_actual, final_predictions, 50, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
plot([min(y_test_actual), max(y_test_actual)], [min(y_test_actual), max(y_test_actual)], 'r--', 'LineWidth', 2);
hold off;
title('Actual vs. Predicted Project Budget (Architectural Model)');
xlabel('Actual Budget (PHP)');
ylabel('Predicted Budget (PHP)');
legend('Predictions', 'Perfect Fit', 'Location', 'northwest');
grid on;
ax = gca;
ax.XAxis.Exponent = 0;
ax.YAxis.Exponent = 0;
exportgraphics(fig_results, fullfile(output_dir, 'architectural_actual_vs_predicted_budget.png'), 'Resolution', 300);
close(fig_results);
fprintf('Saved: %s\n', fullfile(output_dir, 'architectural_actual_vs_predicted_budget.png'));

% --- Visualization: Residuals Plot ---
residuals = y_test_actual - final_predictions;
fig_residuals = figure('Visible', 'off');
scatter(final_predictions, residuals, 50, 'g', 'filled', 'MarkerFaceAlpha', 0.6);
hold on;
yline(0, 'r--', 'LineWidth', 2);
hold off;
title('Residuals Plot (Architectural Model)');
xlabel('Predicted Budget (PHP)');
ylabel('Residuals (Actual - Predicted)');
grid on;
ax = gca;
ax.XAxis.Exponent = 0;
ax.YAxis.Exponent = 0;
exportgraphics(fig_residuals, fullfile(output_dir, 'architectural_residuals_plot.png'), 'Resolution', 300);
close(fig_residuals);
fprintf('Saved: %s\n', fullfile(output_dir, 'architectural_residuals_plot.png'));

% =========================================================================
% Section 7: Save Final Assets
% =========================================================================
fprintf('\n--- Section 7: Saving Final Model and Scalers ---\n');

save('ann_granular_model.mat', 'net');
fprintf("ANN model saved as 'ann_granular_model.mat'\n");

save('scalers_granular.mat', 'scaler_X_mu', 'scaler_X_sigma', 'scaler_y_min', 'scaler_y_max');
fprintf("Feature and target scalers saved as 'scalers_granular.mat'.\n");

fprintf('\nProcess finished successfully.\n');